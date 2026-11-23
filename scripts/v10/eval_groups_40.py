#!/usr/bin/env python
"""V10 group eval (2026-11-21 user spec): 3 groups x 32-class taxonomy x CPs.

Groups (data/eval_groups/*.jsonl, built by v10_eval_taxonomy.py):
  group1_unseen  positions >=2000  — never trained through CP2000
  group2_cp250   positions <250    — trained once by CP250
  group3_late    [750,1000)        — trained once only by CP1000 (recency)

Tokenization is TRAINING-FAITHFUL: ids = train_ds[ds_idx]["input_ids"]
(load_lf_data pipeline, cutoff 1024, WITH the system template) — NOT the
ce16p-style tokenization, which lacks the system segment (ledger #24: the
d15 matrix numbers are internally consistent but used the no-system form;
group-eval numbers and d15 numbers are not cross-comparable).

Metric kernel = the d15 recurrent chain (DSv3 convention + no-roll caller):
  depth k: input e(roll(ids, -(k-1))) + chained draft hidden -> logits_k
  targets ids[t+k], valid iff t+k < true_len (right-pad is CAUSAL-safe:
  padded rows sit at the END, real positions never attend them).
  accept = P(argmax(logits_k[t]) == ids[t+k]); CE at horizon k.

Aggregation: per (group, class, depth) mean with n_cases labels; per
(group, depth) pooled. Canary: train_ds[0] alone through the same kernel.

Run (.40 mtpV8, 16 cards free):
  torchrun --nproc_per_node=16 eval_groups_40.py
"""
import json
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
V8 = os.path.join(os.path.dirname(HERE), "v8")
if V8 not in sys.path:
    sys.path.insert(0, V8)

import torch  # noqa: E402

import time


def _stamp():
    """Wall-clock stamp for [geval] lines (same convention as the trainer)."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


DEPTHS = 5
CHUNK = 256


def _gather_logits(x):
    """Vocab-parallel gather (lazy import keeps the kernel CPU-testable;
    monkeypatch target for the fake-model unit test)."""
    from v10_diag import gather_full_logits
    return gather_full_logits(x, dim=-1)


def batch_metrics(model, batches, mtp_num_layers, depths=DEPTHS):
    """One batch list per CP pass. Each batch: dict(input_ids [B,S] long npu,
    true_lens [B]). Returns {(case_id): {"accept": [D], "ce": [D]}}."""
    out = {}
    with torch.no_grad():
        for batch in batches:
            torch.npu.empty_cache()     # cap cache growth: post-load baseline
            ids = batch["input_ids"]    # +AdamW-free still sits near the edge                       # [B, S]
            B, S = ids.shape
            tl = batch["true_lens"]                        # list[int]
            emb = model.embedding(input_ids=ids,
                                  position_ids=torch.arange(S, device=ids.device)
                                  .unsqueeze(0).expand(B, S).contiguous())
            emb = emb[0] if isinstance(emb, tuple) else emb
            dec = model.decoder(hidden_states=emb, attention_mask=None)
            dec = dec[0] if isinstance(dec, tuple) else dec
            h_prev = dec
            for k in range(1, depths + 1):
                ids_k = torch.roll(ids, shifts=-(k - 1), dims=1)   # e(t+k-1)
                mtp_out = model.mtp(input_ids=ids_k,
                                    position_ids=torch.arange(S, device=ids.device)
                                    .unsqueeze(0).expand(B, S).contiguous(),
                                    hidden_states=h_prev, attention_mask=None,
                                    inference_params=None, embedding=model.embedding)
                ch = torch.chunk(mtp_out, 1 + mtp_num_layers, dim=0)
                d_k = ch[-1]
                logits_k, _ = model.output_layer(d_k)
                logits_k = logits_k[0] if isinstance(logits_k, tuple) else logits_k
                # [S, B, V] (seq-first); vectorized per-row metrics — NO
                # python row loop, NO per-row .item() (80 device syncs/batch
                # killed the async pipeline: 90+ min per CP, probed
                # 2026-11-21); ONE .tolist() sync per depth instead.
                acc_t = torch.zeros(B, device=ids.device)
                ces_t = torch.zeros(B, device=ids.device)
                n_t = torch.zeros(B, device=ids.device)
                n_pos = S - k
                tgt_all = torch.roll(ids, shifts=-k, dims=1)       # [B,S]: ids[t+k]
                for s0 in range(0, n_pos, CHUNK):
                    end = min(s0 + CHUNK, n_pos)
                    seg = logits_k[s0:end]                          # [c, B, V] (seq-first)
                    full = _gather_logits(seg).float()              # [c, B, V]
                    lp = torch.log_softmax(full, dim=-1)
                    tg = tgt_all[:, s0:end]                         # [B, c] (batch-first!)
                    # valid[b, j] = target position j+s0 < tl[b]-k
                    pos = torch.arange(s0, end, device=ids.device)
                    mask = (pos.unsqueeze(0) < (torch.tensor(
                        tl, device=ids.device).unsqueeze(1) - k))   # [B, c]
                    lpT = lp.transpose(0, 1)                        # [B, c, V]
                    ce_vec = (-lpT.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
                              ) * mask                              # [B, c]
                    acc_vec = ((full.argmax(-1).transpose(0, 1) == tg) & mask
                               ).sum(-1)                            # [B]
                    ces_t += ce_vec.sum(-1)
                    acc_t += acc_vec.float()
                    n_t += mask.sum(-1).float()
                acc_l = (acc_t / n_t.clamp(min=1)).tolist()
                ce_l = (ces_t / n_t.clamp(min=1)).tolist()
                for b, cid in enumerate(batch["case_ids"]):
                    if n_t[b] == 0:
                        continue                                    # depth > len
                    d = out.setdefault(cid, {"accept": [], "ce": []})
                    d["accept"].append(acc_l[b])
                    d["ce"].append(ce_l[b])
                h_prev = d_k.detach()                               # chain
    return out


def aggregate(results, case_meta, depths=DEPTHS):
    """results: {cp: {case_id: {accept[D], ce[D]}}}; case_meta: {id: {group,
    cls}}. -> {"per_class": {f"{cp}|{g}|{cls}|D{k}": {accept,ce,n}},
    "per_group": {f"{cp}|{g}|D{k}": ...}}. Pooling = plain mean over cases
    (NOT class-weighted); n = cases contributing that depth."""
    agg = {}
    for cp, m in results.items():
        for cid, d in m.items():
            c = case_meta[cid]
            for k, (a, ce) in enumerate(zip(d["accept"], d["ce"]), 1):
                for key in ((c["group"], c["cls"], k), (c["group"], "*", k)):
                    e = agg.setdefault((cp,) + key, {"a": 0.0, "ce": 0.0, "n": 0})
                    e["a"] += a
                    e["ce"] += ce
                    e["n"] += 1
    return {
        "per_class": {f"{cp}|{g}|{cls}|D{k}": {"accept": v["a"] / v["n"],
                                               "ce": v["ce"] / v["n"], "n": v["n"]}
                      for (cp, g, cls, k), v in agg.items() if cls != "*"},
        "per_group": {f"{cp}|{g}|D{k}": {"accept": v["a"] / v["n"],
                                         "ce": v["ce"] / v["n"], "n": v["n"]}
                      for (cp, g, cls, k), v in agg.items() if cls == "*"},
    }


def main():
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    from launch_ce_gate import npu_shim
    npu_shim()
    import torch.distributed as dist
    dist.init_process_group(backend="hccl", init_method="env://")
    from megatron.core import parallel_state
    world = int(os.environ["WORLD_SIZE"])
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    rank = dist.get_rank()

    from v10_config import vendor_config
    from v10_fill import fill_model
    from v10_model import build_v10_model
    from v10_trainer import build_trainable, load_cp

    from v8_data import load_lf_data
    _tok, _tpl, train_ds = load_lf_data("configs/glm53_v10_train.yaml")
    cutoff = 1024

    cases = []
    for grp in ("group1_unseen", "group2_cp250", "group3_late"):
        path = f"data/eval_groups/{grp}.jsonl"
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            cases.append({"id": f"{grp}#{r['ds_idx']}", "group": grp,
                          "cls": r["class"], "ds_idx": r["ds_idx"]})
    for c in cases:
        ids = train_ds[c["ds_idx"]]
        ids = ids["input_ids"] if isinstance(ids, dict) else ids
        c["ids"] = list(ids[:cutoff])
    if rank == 0:
        print(f"[geval {_stamp()}] {len(cases)} cases; lens "
              f"min={min(len(c['ids']) for c in cases)} "
              f"max={max(len(c['ids']) for c in cases)}", flush=True)

    # sort by length desc -> batches with similar lens (less pad waste)
    cases.sort(key=lambda c: -len(c["ids"]))
    Bsz = int(os.environ.get("V10_GEVAL_BS", "4"))  # 16/8 OOM-probed (44.5G weights + activations): 4 is the proven profile
    batches = []
    for s0 in range(0, len(cases), Bsz):
        part = cases[s0:s0 + Bsz]
        S = max(len(c["ids"]) for c in part)
        ids_t = torch.zeros((len(part), S), dtype=torch.int64)
        for b, c in enumerate(part):
            ids_t[b, :len(c["ids"])] = torch.tensor(c["ids"], dtype=torch.int64)
        batches.append({"input_ids": ids_t.to("npu"),
                        "true_lens": [len(c["ids"]) for c in part],
                        "case_ids": [c["id"] for c in part]})

    vc = vendor_config()
    model, cfg = build_v10_model(vc=vc, device="npu", mtp=True,
                                 max_seq_len=cutoff, tp_size=world)
    ledger = fill_model(model, os.environ.get(
        "V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16"), vc, mtp=True,
        tp_rank=rank, tp_size=world)
    assert not ledger["unwritten"] and not ledger["unconsumed"]
    build_trainable(model)
    model.eval()

    params_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    master = {n: p.detach().clone().float() for n, p in params_named}
    for t in master.values():
        t.requires_grad_(False)
    optimizer = torch.optim.AdamW([master[n] for n, _ in params_named], lr=5e-6)

    cp_dir = os.environ.get("V10_CP_DIR", "saves/v10_t4_long")
    cp_list = [("base", None)] + [(f"cp{s}", s) for s in
                                  (250, 500, 750, 1000, 1250, 1500, 1750, 2000)]
    # V10_GEVAL_ONLY=cp1000: single-CP mode for a second machine running the
    # trained-checkpoint eval in parallel (shard dir separate via
    # V10_GEVAL_SHARD_DIR); the merged view tolerates shards from both.
    only = os.environ.get("V10_GEVAL_ONLY", "")
    if only:
        cp_list = [x for x in cp_list if x[0] == only]
    shard_dir = os.environ.get("V10_GEVAL_SHARD_DIR",
                               "h0-test/eval35/v10_group_eval_shards")
    os.makedirs(shard_dir, exist_ok=True)
    results = {}
    case_meta = {c["id"]: c for c in cases}
    for name, _ in cp_list:
        p = os.path.join(shard_dir, f"{name}.json")
        if not os.path.exists(p):
            continue
        try:
            shard = json.load(open(p))
            if len(shard.get("cases", {})) >= len(case_meta):
                results[name] = shard["cases"]
                if rank == 0:
                    print(f"[geval {_stamp()}] {name}: shard complete — "
                          f"inherited ({len(shard['cases'])} cases)", flush=True)
        except Exception as e:
            if rank == 0:
                print(f"[geval {_stamp()}] {name}: shard unreadable ({e!r}) "
                      f"— will recompute", flush=True)

    def dump_shard(name):
        """Write-once per-CP shard: single small JSON, created once, never
        rewritten — write-race is structurally absent."""
        if rank != 0:
            return
        p = os.path.join(shard_dir, f"{name}.json")
        with open(p, "w") as f:
            json.dump({"cases": results[name]}, f, indent=1)
        print(f"[geval {_stamp()}] shard {name} -> {p}", flush=True)

    for name, step_n in cp_list:
        if name in results:
            continue
        if step_n is not None:
            # release allocator cache before the CP-load transients: the
            # eval sits at ~98.5% HBM and the shard load tipped it over
            # (crash after base, probed 2026-11-21)
            torch.npu.empty_cache()
            cp_file = os.path.join(cp_dir, f"cp{step_n}_rank{rank}.pt")
            if not os.path.exists(cp_file):
                if rank == 0:
                    print(f"[geval {_stamp()}] SKIP {name}: missing", flush=True)
                continue
            load_cp(model, optimizer, cp_file, master=master, with_opt=False)
        m = batch_metrics(model, batches, cfg.mtp_num_layers)
        results[name] = m
        done = sum(1 for v in m.values() if len(v["accept"]) == DEPTHS)
        if rank == 0:
            print(f"[geval {_stamp()}] {name}: {done}/{len(case_meta)} cases "
                  f"x {DEPTHS} depths", flush=True)
        dump_shard(name)

    # final merged view = DERIVED from the shards (shards are the source of
    # truth); written once at the end for convenience
    if rank == 0:
        dump = {"depths": DEPTHS,
                "tokenization": "train_ds(cutoff1024,system)"}
        dump.update(aggregate(results, case_meta))
        with open("h0-test/eval35/v10_group_eval_40.json", "w") as f:
            json.dump(dump, f, indent=1)
        for g in ("group1_unseen", "group2_cp250", "group3_late"):
            print(f"[geval {_stamp()}] {g} pooled accept/CE (rows=CP, cols=D1..D5):",
                  flush=True)
            for cp in results:
                row_a = [dump["per_group"].get(f"{cp}|{g}|D{k}", {}).get("accept")
                         for k in range(1, DEPTHS + 1)]
                row_c = [dump["per_group"].get(f"{cp}|{g}|D{k}", {}).get("ce")
                         for k in range(1, DEPTHS + 1)]
                if any(v is not None for v in row_a):
                    fa = " ".join("  .  " if v is None else f"{v:.4f}"
                                  for v in row_a)
                    fc = " ".join("  .  " if v is None else f"{v:.4f}"
                                  for v in row_c)
                    print(f"  {cp:8s} acc[{fa}] ce[{fc}]", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
