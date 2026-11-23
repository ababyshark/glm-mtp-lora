#!/usr/bin/env python
"""Accuracy gate: B=1 unpadded ground truth vs the batched vectorized kernel.

The group eval's batch path (right-pad + 3D mask + per-depth .tolist syncs)
was never cross-checked against a simple per-case loop. This script runs the
SAME 6 cases both ways and demands |d_accept| < 1e-4 / |d_ce| < 1e-3.
Path A = d15-style B=1 chain (the audited pattern: [S,1,V] sharded logits,
2D chunk gather); Path B = eval_groups_40.batch_metrics (B=6 padded).
Run on .40 (TP16): python xval_kernel.py
"""
import json
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

from v10_diag import gather_full_logits  # noqa: E402

DEPTHS = 5


def chain_b1(model, ids, mtp_layers, depths=DEPTHS, chunk=256):
    """ids: 1-D list, UNPADDED [1, S_true] B=1 — the reference pattern."""
    t = torch.tensor([ids], dtype=torch.int64, device="npu")
    S = t.shape[1]
    res = {"accept": [], "ce": []}
    with torch.no_grad():
        emb = model.embedding(
            input_ids=t,
            position_ids=torch.arange(S, device=t.device).unsqueeze(0))
        emb = emb[0] if isinstance(emb, tuple) else emb
        dec = model.decoder(hidden_states=emb, attention_mask=None)
        dec = dec[0] if isinstance(dec, tuple) else dec
        h_prev = dec
        for k in range(1, depths + 1):
            ids_k = torch.roll(t, shifts=-(k - 1), dims=1)
            mtp_out = model.mtp(
                input_ids=ids_k,
                position_ids=torch.arange(S, device=t.device).unsqueeze(0),
                hidden_states=h_prev, attention_mask=None,
                inference_params=None, embedding=model.embedding)
            d_k = torch.chunk(mtp_out, 1 + mtp_layers, dim=0)[-1]
            logits_k, _ = model.output_layer(d_k)
            logits_k = logits_k[0] if isinstance(logits_k, tuple) else logits_k
            n_pos = S - k
            tgt = t[0, k:k + n_pos]
            acc = 0
            ce = 0.0
            for s0 in range(0, n_pos, chunk):
                end = min(s0 + chunk, n_pos)
                seg = logits_k[s0:end, 0, :]                 # [c, V/tp] (B=1)
                full = gather_full_logits(seg, dim=-1).float()
                lp = torch.log_softmax(full, dim=-1)
                tt = tgt[s0:end]
                ce += (-lp.gather(-1, tt.unsqueeze(-1)).squeeze(-1)
                       ).sum().item()
                acc += (full.argmax(-1) == tt).sum().item()
            res["accept"].append(acc / n_pos)
            res["ce"].append(ce / n_pos)
            h_prev = d_k.detach()
    return res


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
    from v10_trainer import build_trainable
    from v10_data import load_lf_data
    import eval_groups_40 as eg

    _tok, _tpl, train_ds = load_lf_data("configs/glm53_v10_train.yaml")
    cases = []
    for grp in ("group1_unseen", "group2_cp250", "group3_late"):
        for line in open(f"data/eval_groups/{grp}.jsonl", encoding="utf-8"):
            r = json.loads(line)
            cases.append({"id": f"{grp}#{r['ds_idx']}", "group": grp,
                          "ds_idx": r["ds_idx"]})
    picks = [cases[i] for i in (0, 50, 120, 200, 280, 340)]
    for c in picks:
        ids = train_ds[c["ds_idx"]]
        ids = ids["input_ids"] if isinstance(ids, dict) else ids
        c["ids"] = list(ids[:1024])
    if rank == 0:
        print(f"[xval] picks: {[(c['id'], len(c['ids'])) for c in picks]}",
              flush=True)

    vc = vendor_config()
    model, cfg = build_v10_model(vc=vc, device="npu", mtp=True,
                                 max_seq_len=1024, tp_size=world)
    ledger = fill_model(model, os.environ.get(
        "V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16"), vc, mtp=True,
        tp_rank=rank, tp_size=world)
    assert not ledger["unwritten"] and not ledger["unconsumed"]
    build_trainable(model)
    model.eval()

    # Path A: B=1 unpadded, one at a time (rank-0 only computes; others idle
    # but MUST stay in the collectives — run on all ranks, compare on rank0)
    ref = {}
    for i, c in enumerate(picks):
        ref[f"c{i}"] = chain_b1(model, c["ids"], cfg.mtp_num_layers)
        if rank == 0:
            print(f"[xval] A c{i} done ({c['id']}, S={len(c['ids'])})",
                  flush=True)

    # Path B: the batched kernel (B=4 chunks — B=6 OOMs in the MoE dispatch)
    got = {}
    for s0 in range(0, len(picks), 4):
        part = picks[s0:s0 + 4]
        Smax = max(len(c["ids"]) for c in part)
        ids_t = torch.zeros((len(part), Smax), dtype=torch.int64,
                            device="npu")
        for b, c in enumerate(part):
            ids_t[b, :len(c["ids"])] = torch.tensor(c["ids"],
                                                    dtype=torch.int64)
        batches = [{
            "input_ids": ids_t,
            "true_lens": [len(c["ids"]) for c in part],
            "case_ids": [f"c{s0 + i}" for i in range(len(part))],
        }]
        got.update(eg.batch_metrics(model, batches, cfg.mtp_num_layers))

    if rank == 0:
        ok = True
        for i, c in enumerate(picks):
            cid = f"c{i}"
            for k in range(DEPTHS):
                da = abs(ref[cid]["accept"][k] - got[cid]["accept"][k])
                dc = abs(ref[cid]["ce"][k] - got[cid]["ce"][k])
                flag = "OK" if (da < 1e-4 and dc < 1e-3) else "FAIL"
                ok = ok and flag == "OK"
                if k < 2 or flag == "FAIL":
                    print(f"  {c['id']:22s} D{k+1} d_acc={da:.2e} "
                          f"d_ce={dc:.2e} [{flag}]", flush=True)
        print(f"[xval] {'PASS' if ok else 'FAIL'} — batched kernel "
              f"{'validated' if ok else 'BROKEN'} vs B=1 ground truth",
              flush=True)
        json.dump({"ref": ref, "got": got,
                   "ids": [c["id"] for c in picks]},
                  open("h0-test/eval35/v10_xval_kernel.json", "w"), indent=1)
    dist.barrier()


if __name__ == "__main__":
    main()
