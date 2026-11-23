#!/usr/bin/env python
"""V10 Phase 2c/4 trainer: LF data + mcore 0.18.2 — the thin glue (V8 skeleton
port; adds NO framework). Flash judgment training = MTP-subtree-only optimizer
(shared draft head = frozen main output_layer; base decoder frozen).

Flow per step (mcore 0.18.2 contract): forward_step -> fwdbwd -> optimizer.step.
Two modes:
  native  : model(labels) -> mcore process_mtp_loss (draft CE at +1)
  recipe  : no_grad decoder -> h0 -> v10_recipe.serving_chain (CE+margin+KL+agree)
            (GLM53_V10_RECIPE=1; the h0-quality diagnostic rides along)

CP = rank-sharded torch.save of the optimizer-managed MTP shard (V8 pattern).
"""
import argparse
import json
import math
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from v10_config import vendor_config  # noqa: E402
from v10_fill import fill_model  # noqa: E402
from v10_model import build_v10_model  # noqa: E402
from v10_data import McoreBatcher, load_lf_data  # noqa: E402  (iron rules §6 reuse, V8 verbatim)
from launch_ce_gate import npu_shim  # noqa: E402


def _stamp():
    """Wall-clock stamp for every [v10-trainer] log line (restart/duration
    forensics: the 2000-step run's per-step lines previously carried NO
    timestamp — step-time-vs-wallclock could not be reconciled post hoc)."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def fixed_diag_kernel(logits, tgt, gather_fn, chunk=256):
    """EVAL-FIXED CE/agree over the TARGET horizon (len(tgt) = S-1 rows).
    The chunk loop clamps the LAST chunk to len(tgt) — a range() start bound
    alone lets seg overrun tgt by one and (am == tgt) explodes S vs S-1
    (ledger #23, introduced by the #19 squeeze fix). gather_fn is injected
    (mcore vocab-parallel gather in prod; identity in tests)."""
    n_pos = tgt.shape[0]
    ce_v = 0.0
    n = 0
    am = []
    for s0 in range(0, n_pos, chunk):
        end = min(s0 + chunk, n_pos)
        seg = logits[s0:end]
        full = gather_fn(seg.contiguous()).float()
        lp = torch.log_softmax(full, dim=-1)
        if lp.dim() == 3:
            lp = lp[:, 0]          # B=1: [chunk,1,V] -> [chunk,V]
        t = tgt[s0:end]
        ce_v = ce_v + (-lp.gather(-1, t.unsqueeze(-1)).squeeze(-1)).sum()
        n += t.numel()
        am.append(full.argmax(-1).reshape(-1))
    am = torch.cat(am)
    return (ce_v / max(n, 1)).item(), (am == tgt).float().mean().item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="LF data config (V8 keys)")
    p.add_argument("--max-steps", type=int, default=150)
    p.add_argument("--cp-interval", type=int, default=50)
    p.add_argument("--tp-size", type=int, default=16)
    p.add_argument("--num-layers", type=int, default=None, help="truncation (smoke)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=None,
                   help="batcher/fwdbwd seq length (default: yaml cutoff_len)")
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--load-cp", default=None)
    p.add_argument("--determinism", action="store_true")
    p.add_argument("--diag-every", type=int, default=10,
                   help="native mode: every N steps, log manual full-vocab "
                   "main/draft CE (the raw loss is the self-paired main term "
                   "+ backward-only mtp term — NOT a meaningful curve)")
    p.add_argument("--skip-batches", type=int, default=0)
    p.add_argument("--vendor", default=os.environ.get(
        "V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16"))
    return p.parse_args()


def _opt_param_names(model, optimizer):
    managed = set()
    for g in optimizer.param_groups:
        for p in g["params"]:
            managed.add(id(p))
    return {n: p for n, p in model.named_parameters() if id(p) in managed}


def save_cp(model, optimizer, step, out_dir, master=None):
    rank = int(os.environ.get("RANK", "0"))
    os.makedirs(out_dir, exist_ok=True)
    mine = master if master is not None else _opt_param_names(model, optimizer)
    shard = {n: t.detach().cpu() for n, t in mine.items()}
    torch.save({"step": step, "shard": shard,
                "opt": optimizer.state_dict()}, f"{out_dir}/cp{step}_rank{rank}.pt")
    if rank == 0:
        print(f"[v10-trainer {_stamp()}] CP saved: step={step} -> {out_dir}/cp{step}_rank{{0..}}.pt "
              f"({len(shard)} params)", flush=True)


def load_cp(model, optimizer, cp_file, master=None, with_opt=True):
    """with_opt=False: eval-only resume — skips optimizer.load_state_dict
    (AdamW moments would materialize ~3.6GB on GPU for NOTHING: the metric
    pass never calls optimizer.step; the weights still restore via shard →
    master → model)."""
    blob = torch.load(cp_file, weights_only=False)
    shard = blob["shard"]
    mine = master if master is not None else _opt_param_names(model, optimizer)
    missing = [n for n in mine if n not in shard]
    if missing:
        raise KeyError(f"CP shard missing trainable params: {missing[:6]}")
    with torch.no_grad():
        for n, t in mine.items():
            if tuple(t.shape) != tuple(shard[n].shape):
                raise ValueError(f"CP shape mismatch {n}")
            t.copy_(shard[n].to(t.device))
    if with_opt:
        optimizer.load_state_dict(blob["opt"])
    if master is not None:
        pm = dict(model.named_parameters())
        with torch.no_grad():
            for n, t in master.items():
                pm[n].data.copy_(t.to(pm[n].device).to(pm[n].dtype))
    return blob["step"]


def _mtp_frozen(name):
    """V10.3 freeze surface (user verdict 2026-11-23: narrow-domain 100-step
    overwrites scrambled the fragile MTP structures first — router routing
    collapsed, norm drift compounds through the chain). Frozen inside mtp.*:
      router (mlp.gate)  — GLM53_V10_FREEZE_MTP_ROUTER=1 (default)
      every RMSNorm      — GLM53_V10_FREEZE_MTP_NORMS=1 (default: enorm/hnorm/
                           final/input/pre_mlp/q_a_ln/kv_a_ln/indexer.k_norm)
      eh_proj            — GLM53_V10_FREEZE_EH_PROJ=1 (default OFF: it is the
                           draft's entry projection and one of the real learners)
    Trainable in mtp.*: attention linears + expert MLPs (fc1/fc2). Frozen keys
    are absent from the CP (optimizer-managed set) — the exporter treats
    missing keys as frozen and keeps the training-base value (V10.3 contract)."""
    if os.environ.get("GLM53_V10_FREEZE_MTP_ROUTER", "0") == "1" \
            and ".mlp.router." in name:
        return True
    if os.environ.get("GLM53_V10_FREEZE_MTP_NORMS", "0") == "1" \
            and "norm" in name:
        return True
    if os.environ.get("GLM53_V10_FREEZE_EH_PROJ", "0") == "1" \
            and "eh_proj" in name:
        return True
    return False


def _kl_reduce(main_det, draft, tk):
    """KL impl selector (V10.3 2026-11-23): tk>0 → legacy top-k renormalized
    truncation (V10.2 compatibility, unchanged bytes); tk=0 → FULL-vocab
    chunked_vocab_kl. The top-128 renormalization leaks probability mass
    off-distribution (the draft never learns behavior outside the top-128
    support — serving showed exactly that gap). main is the caller-detached
    fixed reference; gradient flows into draft only."""
    if tk > 0:
        from v10_diag import chunked_topk_kl
        return chunked_topk_kl(main_det, draft, topk=tk)
    from v10_diag import chunked_vocab_kl
    return chunked_vocab_kl(main_det, draft)


def build_trainable(model):
    """Trainable whitelist (user contract 2026-11-20: 主权重+MTP 都微调;
    V10.3 2026-11-23: mtp router/norms frozen per _mtp_frozen):
    LoRA adapters on the BF16 base decoder (v10_lora.apply_lora) + the MTP
    subtree minus the frozen surface. Base weights themselves stay frozen —
    full-param AdamW on ~300B is physically infeasible (~225G/card); adapters
    merge into the base at export. Returns (handled lora names, n_frozen)."""
    from v10_lora import apply_lora
    handled = apply_lora(model.decoder)
    n_frozen = 0
    for n, p in model.named_parameters():
        if n.startswith("mtp.") and _mtp_frozen(n):
            p.requires_grad_(False)
            n_frozen += 1
        elif n.startswith("mtp.") or "lora_" in n:
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
            n_frozen += 1
    return handled, n_frozen


def main():
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
    dev = torch.device("npu", local_rank)
    npu_shim()

    import torch.distributed as dist
    dist.init_process_group(backend="hccl", init_method="env://")
    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=args.tp_size)
    world = dist.get_world_size()
    assert world == args.tp_size, f"V10 = TP{args.tp_size}/DP1 (world={world})"
    rank = dist.get_rank()

    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(1234)

    recipe_mode = os.environ.get("GLM53_V10_RECIPE", "0") == "1"
    if rank == 0:
        print(f"[v10-trainer {_stamp()}] config: recompute="
              f"{os.environ.get('GLM53_V10_RECOMPUTE', 'full(default)')} "
              f"ce=chunked-gather(no-ckpt) lora_r="
              f"{os.environ.get('GLM53_V10_LORA_R', '16')} "
              f"warmup={os.environ.get('GLM53_V10_WARMUP', '20')} "
              f"clip={os.environ.get('GLM53_V10_CLIP', '1.0')} "
              f"draft_w={os.environ.get('GLM53_V10_DRAFT_W', '1.0')}",
              flush=True)
    if os.environ.get("GLM53_V10_ANOMALY", "0") == "1":
        torch.autograd.set_detect_anomaly(True)
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            print(f"[v10-trainer {_stamp()}] anomaly detection ON (GLM53_V10_ANOMALY=1)",
                  flush=True)
    vc = vendor_config()
    nl = args.num_layers or vc["num_hidden_layers"]
    model, cfg = build_v10_model(vc=vc, num_layers=nl, device="npu", mtp=True,
                                 max_seq_len=2048, tp_size=args.tp_size)
    ledger = fill_model(model, args.vendor, vc, mtp=True, tp_rank=rank,
                        tp_size=args.tp_size, verbose=(rank == 0))
    if ledger["unwritten"] or ledger["unconsumed"]:
        raise RuntimeError(f"fill ledger not clean: "
                           f"unwritten={ledger['unwritten'][:5]} "
                           f"unconsumed={ledger['unconsumed'][:5]}")
    lora_names, n_frozen = build_trainable(model)
    if not lora_names:
        raise RuntimeError("LoRA injected 0 targets — whitelist/filter bug "
                           "(silent no-op would train mtp-only)")

    # nan-birth locator: per-layer backward tracer — the FIRST layer (walking
    # upstream from the loss) whose incoming grad carries NaN localizes the
    # culprit's backward. Rank0-only, first two hits.
    if os.environ.get("GLM53_V10_BKWD_TRACE", "1") == "1":
        _trace = {"n": 0}

        def _mk_bkwd_hook(idx):
            def hook(m, gi, go):
                if _trace["n"] >= 2:
                    return
                g = go[0] if go else None
                if g is None:
                    return
                if not bool(torch.isfinite(g).all()):
                    _trace["n"] += 1
                    print(f"[v10-trainer {_stamp()}] BKWD-NAN layer{idx}: "
                          f"go has-nan (first hit walking upstream)",
                          flush=True)
            return hook
        for li, layer in enumerate(model.decoder.layers):
            layer.register_full_backward_hook(_mk_bkwd_hook(li))
    if rank == 0:
        print(f"[v10-trainer rank0 {_stamp()}] fill clean ({ledger['n_written']} targets, "
              f"{ledger['n_deferred']} deferred); frozen {n_frozen} base params; "
              f"lora injected: {len(lora_names)} targets; "
              f"mode={'recipe' if recipe_mode else 'native'}", flush=True)
    model.eval()
    meta_leak = [n for n, t in list(model.named_parameters())
                 if t.device.type == "meta"]
    if meta_leak:
        raise RuntimeError(f"meta tensors leaked past fill: {meta_leak[:8]}")
    torch.npu.empty_cache()

    # Mixed-precision optimizer (user decision 2026-11-20; NPU fused adam
    # REJECTS mixed dtypes — bf16 param + fp32 moment → CheckDtypeValid):
    # fp32 MASTER weights own the true state, fp32 Adam moments step on them,
    # and each step round-trips bf16 model params ← masters. Masters are the
    # CP payload (load restores masters + syncs the bf16 model).
    params_named = [(n, p) for n, p in model.named_parameters()
                    if p.requires_grad]
    params = [p for _, p in params_named]
    n_train = sum(p.numel() for p in params)
    master = {n: p.detach().clone().float() for n, p in params_named}
    for t in master.values():
        t.requires_grad_(False)
    # LR 分组（专家铁律⑤，2026-11-22）：MTP（学生）的学习率必须高于主模型
    # LoRA（老师），学生适应快于老师变化——否则老师漂移时学生追不上。
    # GLM53_V10_MTP_LR 默认 2e-5（专家建议区间 2e-5~5e-5）。
    mtp_lr = float(os.environ.get("GLM53_V10_MTP_LR", "2e-5"))
    lora_master = [master[n] for n, _ in params_named if "lora_" in n]
    mtp_master = [master[n] for n, _ in params_named if n.startswith("mtp.")]
    other_master = [master[n] for n, _ in params_named
                    if "lora_" not in n and not n.startswith("mtp.")]
    groups = [{"params": lora_master, "lr": args.lr, "base_lr": args.lr}]
    if mtp_master:
        groups.append({"params": mtp_master, "lr": mtp_lr,
                       "base_lr": mtp_lr})
    if other_master:
        groups.append({"params": other_master, "lr": args.lr,
                       "base_lr": args.lr})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.0)
    n_lora_t = sum(1 for n, _ in params_named if "lora_" in n)
    n_lora_p = sum(p.numel() for n, p in params_named if "lora_" in n)
    n_mtp_t = sum(1 for n, _ in params_named if n.startswith("mtp."))
    n_mtp_p = sum(p.numel() for n, p in params_named if n.startswith("mtp."))
    if rank == 0:
        print(f"[v10-trainer rank0 {_stamp()}] trainable: {n_train} ({len(params)} tensors) = "
              f"LoRA {n_lora_p/1e6:.1f}M/{n_lora_t}t + MTP {n_mtp_p/1e6:.1f}M/{n_mtp_t}t "
              f"lr=lora {args.lr} / mtp {mtp_lr} optimizer=fp32-master", flush=True)

    tokenizer, template, train_ds = load_lf_data(args.yaml,
                                                 max_samples=args.max_samples)
    import yaml as _yaml
    seq_len = args.seq_len or int(_yaml.safe_load(open(args.yaml)).get("cutoff_len", 2048))
    if rank == 0:
        tname = getattr(template, "name", None) or \
            getattr(template, "template_name", "?")
        print(f"[v10-trainer rank0 {_stamp()}] LF dataset: {len(train_ds)} samples "
              f"(template={tname}, seq_len={seq_len})", flush=True)
    batcher = McoreBatcher(train_ds, args.batch_size, seq_len, dev)
    # FIXED eval batch (diag on the same tokens every time — kills batch noise
    # so the long-run main/draft CE curves are readable)
    row0 = train_ds[0]
    ids0 = row0["input_ids"] if "input_ids" in row0 else row0
    t0 = torch.tensor(ids0[:seq_len], dtype=torch.long)
    eb = torch.full((1, seq_len), 0, dtype=torch.long)
    eb[0, :t0.shape[0]] = t0
    eval_batch = {"input_ids": eb.to(dev), "labels": eb.to(dev).clone(),
                  "position_ids": torch.arange(seq_len, device=dev)
                  .unsqueeze(0).contiguous()}
    if rank == 0:
        print(f"[v10-trainer rank0 {_stamp()}] eval batch fixed: {t0.shape[0]}/{seq_len} tokens",
              flush=True)
    for _ in range(args.skip_batches):
        batcher.next_batch()

    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    fwdbwd = get_forward_backward_func()
    batcher_state = {"last": None}

    _phase_t = {"dec_fwd": 0.0, "mtp_fwd": 0.0, "ce": 0.0}

    def native_loss(model, batch):
        """V10-owned loss (2026-11-20 contract: LoRA(BF16 base) + full-MTP).

        Both terms are next-token CE on the gate-proven DIFFERENTIABLE path
        (mcore gather_from_tensor_model_parallel_region + fp32 softmax — the
        internal vocab_parallel_cross_entropy is blacklisted on the .40 stack,
        error-ledger #10):
          main  = CE(main_logits[t], ids[t+1])   — trains decoder LoRA
          draft = CE(draft_logits[t], ids[t+1])  — vendor convention (h[t], e[t])
        replaces process_mtp_loss (self-paired main term + chained target).
        """
        t_dec0 = time.time()
        ids_b = batch["input_ids"]
        emb = model.embedding(input_ids=ids_b,
                              position_ids=batch["position_ids"])
        emb = emb[0] if isinstance(emb, tuple) else emb
        # reentrant-checkpoint early-out: if NO checkpoint input requires
        # grad, mcore's full-recompute region drops EVERYTHING inside it from
        # autograd (R8: lora-grads=0 mtp-grads=0). The embedding is frozen so
        # emb is a no-grad leaf — mark it to anchor the graph (~67MB emb.grad
        # cost). Only needed WITH recompute; without it the graph builds
        # naturally (torch_npu's recompute free-mechanism is broken —
        # GLM53_V10_RECOMPUTE=off is the default as of 2026-11-20).
        if os.environ.get("GLM53_V10_RECOMPUTE", "full") != "off":
            emb.requires_grad_(True)
        dec = model.decoder(hidden_states=emb, attention_mask=None)
        dec = dec[0] if isinstance(dec, tuple) else dec          # [S, B, H] grad ✓
        _phase_t["dec_fwd"] += time.time() - t_dec0
        main_logits, _ = model.output_layer(dec)
        main_logits = main_logits[0] if isinstance(main_logits, tuple) \
            else main_logits                                     # [S, B, V/tp]
        t_mtp0 = time.time()
        # Teacher/student split (V10.2 audit): the draft-side losses
        # (draft-CE/KL/agree/margin) must NOT backprop into the main LoRA via
        # the shared hidden state — otherwise the "LoRA = slow teacher / MTP =
        # fast student" LR grouping is violated (the MTP alignment loss would
        # drag LoRA too). detach() severs the grad_fn (no path back to LoRA);
        # requires_grad_(True) then marks the detached leaf so mcore's
        # full-recompute checkpoint still builds the backward graph — WITHOUT
        # it the checkpoint sees a no-grad input and drops the whole MTP region
        # from autograd (R8: mtp-grads=0, the .34 04:10 run). main_ce still
        # trains LoRA via main_logits <- dec (untouched). The recipe path
        # detaches base_hidden the same way (v10_recipe.py:116).
        mtp_out = model.mtp(input_ids=ids_b,
                            position_ids=batch["position_ids"],
                            hidden_states=dec.detach().requires_grad_(True),
                            attention_mask=None,
                            inference_params=None, embedding=model.embedding)
        chunks = torch.chunk(mtp_out, 1 + cfg.mtp_num_layers, dim=0)
        draft_logits, _ = model.output_layer(chunks[-1])
        draft_logits = draft_logits[0] if isinstance(draft_logits, tuple) \
            else draft_logits
        _phase_t["mtp_fwd"] += time.time() - t_mtp0

        tgt = ids_b[:, 1:].transpose(0, 1).contiguous()          # [S-1, B]
        t_ce0 = time.time()
        from v10_diag import chunked_full_vocab_ce

        main_ce = chunked_full_vocab_ce(main_logits[:-1].contiguous(), tgt)
        draft_ce = chunked_full_vocab_ce(draft_logits[:-1].contiguous(), tgt)
        _phase_t["ce"] += time.time() - t_ce0
        # T5 proxy (the REAL metric per the contract): draft-vs-main argmax
        # agreement. CHUNKED bf16, no_grad — the fp32 full-vocab materialization
        # here was the B=4 OOM hog (~14.5GB transient, diag 2026-11-20); argmax
        # needs no fp32 and 256-row chunks keep the gather tiny.
        from megatron.core import tensor_parallel
        with torch.no_grad():
            agree_n = agree_d = 0
            CH = 256
            for s0 in range(0, main_logits.shape[0] - 1, CH):
                lm = main_logits[s0:s0 + CH].contiguous()
                ld = draft_logits[s0:s0 + CH].contiguous()
                fm = tensor_parallel.gather_from_tensor_model_parallel_region(lm)
                fd = tensor_parallel.gather_from_tensor_model_parallel_region(ld)
                am_m = fm.argmax(-1)                       # [ch, B] over full V
                am_d = fd.argmax(-1)
                agree_d += (am_m == am_d).sum()
                agree_n += am_m.numel()
            agree = (agree_d / max(agree_n, 1)).item()
        w = float(os.environ.get("GLM53_V10_DRAFT_W", "0"))  # V10.2 专家裁决: gold-CE=0(KL+agree 全词表监督已足够)
        loss = main_ce + w * draft_ce
        # KL(main‖draft) alignment (serving verdict 2026-11-22, §11.4): the
        # corpus-CE-only recipe pulled the draft OFF the target model's
        # distribution — serving K̄ base 3.52 → cp500 2.49 (−29%). The KL
        # anchors the draft to the DETACHED main model's distribution (trains
        # the draft only). Weight = the V5.4 distill weight; 0 = legacy off.
        kl_w = float(os.environ.get("GLM53_MTP_DISTILL_WEIGHT", "0"))
        if kl_w > 0:
            # default 128 = V10.2 replay safety (legacy path byte-identical);
            # the V10.3 launcher pins GLM53_V10_TOPK_KL=0 for full-vocab KL.
            tk = int(os.environ.get("GLM53_V10_TOPK_KL", "128"))
            kl = _kl_reduce(main_logits[:-1].contiguous().detach(),
                            draft_logits[:-1].contiguous(), tk)
            loss = loss + kl_w * kl
            model._last_kl = float(kl.detach())
        # argmax-shaping terms (V8 frozen recipe, §11.4): agree-CE chases the
        # DETACHED main argmax; margin pushes the GOLD above the runner-up.
        # Weights = the V8 verdict (0.5 / 0.3, m=0.5); 0 = off (legacy).
        aw = float(os.environ.get("GLM53_V10_AGREE_W", "0"))
        mw = float(os.environ.get("GLM53_V10_MARGIN_W", "0"))
        mm = float(os.environ.get("GLM53_V10_MARGIN_M", "0.5"))
        if aw > 0 or mw > 0:
            from megatron.core import tensor_parallel as _tp_am
            from v10_diag import chunked_agree_margin
            va = _tp_am.gather_from_tensor_model_parallel_region(
                main_logits[:-1].contiguous()).argmax(-1).detach()  # [S-1, B]
            agree_loss, margin_loss = chunked_agree_margin(
                draft_logits[:-1].contiguous(), va, margin=mm)
            if aw > 0:
                loss = loss + aw * agree_loss
                model._last_agree = float(agree_loss.detach())
            if mw > 0:
                loss = loss + mw * margin_loss
                model._last_margin = float(margin_loss.detach())
        model._last_main_ce = float(main_ce.detach().mean())
        model._last_draft_ce = float(draft_ce.detach().mean())
        model._last_argmax_agree = agree
        return loss

    def forward_step(data_iterator, model):
        batch, _ = next(data_iterator)
        batcher_state["last"] = batch
        if recipe_mode:
            S_b = batch["input_ids"].shape[1]  # batch-first [B, S]: shape[1] = S (the [1,1,1,1]-mask bug was silent in V8)
            # bool KEEP-mask (True=keep, tril) per causal_ends_from_mask contract
            causal_keep = torch.ones(S_b, S_b, dtype=torch.bool,
                                     device=dev).tril().unsqueeze(0).unsqueeze(0)
            emb = model.embedding(input_ids=batch["input_ids"],
                                  position_ids=batch["position_ids"])
            with torch.no_grad():
                dec = model.decoder(hidden_states=emb,
                                    attention_mask=causal_keep)
            h0 = dec[0] if isinstance(dec, tuple) else dec  # [S, B, H]
            from v10_recipe import serving_chain_recipe_loss
            loss = serving_chain_recipe_loss(model, batch["input_ids"],
                                             batch["labels"], h0)
            with torch.no_grad():
                from megatron.core.tensor_parallel.cross_entropy import (
                    vocab_parallel_cross_entropy)
                lg = model.output_layer(h0)
                lg = lg[0] if isinstance(lg, tuple) else lg
                tgt = batch["input_ids"][:, 1:].transpose(0, 1).contiguous()
                ce_diag = vocab_parallel_cross_entropy(
                    lg[:-1].contiguous(), tgt).mean().item()
            print(f"[v10-trainer {_stamp()}] h0-quality diag: main-head CE={ce_diag:.4f} "
                  f"(~2-4 = h0 OK; ~11 = h0 broken)", flush=True)
        else:
            loss = native_loss(model, batch)
        return loss, lambda l: (l, {"loss": float(l.detach().mean())})

    out_dir = args.output_dir or os.path.join(os.path.dirname(HERE),
                                              "saves", "v10_phase2")
    start_step = 0
    if args.load_cp:
        cp_file = args.load_cp.replace("{rank}", os.environ.get("RANK", "0"))
        restored = load_cp(model, optimizer, cp_file, master=master)
        start_step = restored
        args.skip_batches = max(args.skip_batches, restored)
        if rank == 0:
            print(f"[v10-trainer {_stamp()}] CP restored: step={restored} <- {cp_file} "
                  f"(loop resumes {start_step + 1}/{args.max_steps})", flush=True)
    model.train()
    for step in range(start_step + 1, args.max_steps + 1):
        t0 = time.time()
        t_fwd_bwd0 = time.time()
        losses = fwdbwd(forward_step_func=forward_step, data_iterator=[batcher],
                        model=[model], num_microbatches=1,
                        seq_length=seq_len, micro_batch_size=args.batch_size)
        t_fwdbwd = time.time() - t_fwd_bwd0
        loss_val = None
        if isinstance(losses, (list, tuple)) and losses:
            e0 = losses[0]
            v = e0.get("loss") if isinstance(e0, dict) else e0
            loss_val = float(v) if not hasattr(v, "item") else v.item()
        elif hasattr(losses, "item"):
            loss_val = float(losses)
        leaks = [n for n, p in model.named_parameters()
                 if not p.requires_grad and p.grad is not None]
        if leaks:
            raise RuntimeError(f"frozen-param grad leaks: {leaks[:6]}")
        det_note = ""
        if args.determinism and step == 1:
            model.eval()
            with torch.no_grad():
                batch = batcher_state["last"]
                if recipe_mode:
                    S_b = batch["input_ids"].shape[1]  # batch-first [B, S]: shape[1] = S (the [1,1,1,1]-mask bug was silent in V8)
                    causal_keep = torch.ones(S_b, S_b, dtype=torch.bool,
                                             device=dev).tril() \
                        .unsqueeze(0).unsqueeze(0)
                    emb = model.embedding(input_ids=batch["input_ids"],
                                          position_ids=batch["position_ids"])
                    dec = model.decoder(hidden_states=emb,
                                        attention_mask=causal_keep)
                    h0 = dec[0] if isinstance(dec, tuple) else dec
                    from v10_recipe import serving_chain_recipe_loss
                    l2 = serving_chain_recipe_loss(model, batch["input_ids"],
                                                   batch["labels"], h0)
                else:
                    l2 = native_loss(model, batch)
                d = abs(float(l2) - loss_val)
            det_note = f" determinism:|d|={d:.3e}({'OK' if d == 0 else 'DRIFT'})"
            model.train()
        t_opt0 = time.time()
        with torch.no_grad():
            sq_sum = 0.0
            nonfinite = 0
            g_lora_sq = g_mtp_sq = 0.0
            g_lora_n = g_mtp_n = 0
            for n, p in params_named:
                g = p.grad
                if g is None:
                    master[n].grad = None
                    continue
                if not bool(torch.isfinite(g).all()):
                    nonfinite += 1
                gf = g.detach().float()
                sq = float(gf.pow(2).sum())
                sq_sum += sq
                if "lora_" in n:
                    g_lora_sq += sq
                    g_lora_n += 1
                elif n.startswith("mtp."):
                    g_mtp_sq += sq
                    g_mtp_n += 1
                master[n].grad = gf
                p.grad = None      # BUG FIX: autograd ACCUMULATES p.grad across
                                   # steps — without this the optimizer trains
                                   # on SUM(historical grads) (gnorm 6.4→46
                                   # linear climb, error-ledger #16)
        if step == 1 and rank == 0:
            pm = dict(model.named_parameters())
            frozen_with_grad = [n for n, p in pm.items()
                                if not p.requires_grad and p.grad is not None]
            print(f"[v10-trainer {_stamp()}] GRAD AUDIT step1: lora-grads={g_lora_n} "
                  f"mtp-grads={g_mtp_n} frozen-with-grad={len(frozen_with_grad)} "
                  f"(must be 0) — this run IS LoRA(base)+MTP", flush=True)
        if nonfinite and step <= 2 and rank == 0:
            bad = [n for n, _ in params_named
                   if master[n].grad is not None
                   and not bool(torch.isfinite(master[n].grad).all())]
            print(f"[v10-trainer {_stamp()}] NONFINITE grad owners (first 8): {bad[:8]} "
                  f"total={len(bad)}", flush=True)
        t_opt = time.time() - t_opt0
        gnorm = sq_sum ** 0.5
        # nan-guard (audit recommendation): a non-finite gnorm means the
        # backward poisoned the grads — applying them (clip scale=nan) would
        # destroy ALL masters in one step. Skip the update, keep training.
        if not math.isfinite(gnorm):
            optimizer.zero_grad(set_to_none=True)
            print(f"[v10-trainer {_stamp()}] step {step}: NONFINITE gnorm — "
                  f"optimizer.step SKIPPED (params preserved)", flush=True)
            continue
        warmup = int(os.environ.get("GLM53_V10_WARMUP", "20"))
        min_lr = float(os.environ.get("GLM53_V10_MIN_LR", "1e-6"))
        if warmup > 0 and step <= warmup:
            for g in optimizer.param_groups:
                g["lr"] = g["base_lr"] * step / warmup   # 分组 warmup
        elif os.environ.get("GLM53_V10_COSINE", "1") == "1":
            prog = min(1.0, max(0.0, (step - warmup) / max(1, args.max_steps - warmup)))
            import math as _m
            for g in optimizer.param_groups:
                g["lr"] = min_lr + (g["base_lr"] - min_lr) * 0.5 * (1 + _m.cos(_m.pi * prog))
        clip = float(os.environ.get("GLM53_V10_CLIP", "1.0"))
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [master[n] for n, _ in params_named if master[n].grad is not None],
                clip)
        optimizer.step()
        with torch.no_grad():
            for n, p in params_named:
                p.data.copy_(master[n].data.to(p.dtype))
        optimizer.zero_grad(set_to_none=True)
        dt = time.time() - t0
        if rank == 0:
            extra = ""
            if os.environ.get("GLM53_LOSS_DECOMP", "1") == "1":
                extra = f" kl={getattr(model, '_last_kl', None)} " \
                        f"agree={getattr(model, '_last_agree', None)} " \
                        f"margin={getattr(model, '_last_margin', None)}"
            batch_s = batcher_state["last"]["input_ids"].shape[1]
            print(f"[v10-trainer {_stamp()}] step {step}/{args.max_steps} "
                  f"loss={loss_val:.4f} ({dt:.1f}s, {args.batch_size * batch_s / dt:.0f} tok/s) "
                  f"S={batch_s} gnorm={gnorm:.3f} "
                  f"fwdbwd={t_fwdbwd:.1f}s opt={t_opt:.2f}s "
                  + (f" NONFINITE_GRADS={nonfinite}" if nonfinite else "")
                  + f" alloc={torch.npu.memory_allocated()/2**30:.1f}G "
                  f"resv={torch.npu.memory_reserved()/2**30:.1f}G "
                  f"ph[dec={_phase_t['dec_fwd']:.1f} "
                  f"mtp={_phase_t['mtp_fwd']:.1f} ce={_phase_t['ce']:.1f}s]"
                  f"{det_note}{extra}", flush=True)
            _phase_t["dec_fwd"] = _phase_t["mtp_fwd"] = _phase_t["ce"] = 0.0
        # native-mode diagnostic: the raw loss pairs logits[t] with labels[t]
        # (self-paired main term + backward-only mtp term) — the MEANINGFUL
        # curves are the manual full-vocab main/draft CEs (diag R5 verdict).
        if not recipe_mode and args.diag_every and step % args.diag_every == 0 \
                and getattr(model, "_last_main_ce", None) is not None:
            # ---- diag (NON-FATAL: any failure here must never kill training) ----
            def _run_diag():
                from v10_diag import lm_head_ce
                # GROUP ACCOUNTING: per-group grad norms from the cast loop
                # (pre-clip, pre-zero_grad) + the lora_B off-zero hard check.
                b_max = max(float(p.detach().abs().max()) for n, p in params_named
                            if n.endswith("lora_B"))
                if rank == 0:
                    print(f"[v10-trainer {_stamp()}] diag step {step}: "
                          f"main CE={model._last_main_ce:.4f} "
                          f"draft CE={model._last_draft_ce:.4f} "
                          f"argmax-agree={getattr(model, '_last_argmax_agree', float('nan')):.4f} "
                          f"(train batch — the signal)", flush=True)
                    print(f"[v10-trainer {_stamp()}] diag step {step}: B_moved="
                          f"{'YES' if b_max > 0 else 'NO (INJECTION/WIRING BUG — STOP)'} "
                          f"group-grads(pre-clip): lora={g_lora_sq ** 0.5:.4f} "
                          f"mtp={g_mtp_sq ** 0.5:.4f}", flush=True)
                # EVAL-FIXED (same tokens every diag — the trend), with the
                # argmax agreement (the cleanest T5 proxy, free from the logits)
                with torch.no_grad():
                    emb_e = model.embedding(input_ids=eval_batch["input_ids"],
                                            position_ids=eval_batch["position_ids"])
                    emb_e = emb_e[0] if isinstance(emb_e, tuple) else emb_e
                    dec_e = model.decoder(hidden_states=emb_e, attention_mask=None)
                    dec_e = dec_e[0] if isinstance(dec_e, tuple) else dec_e
                    from megatron.core import tensor_parallel as _tp

                    def _fixed_diag(logits):
                        tgt = eval_batch["input_ids"][0, 1:]
                        return fixed_diag_kernel(
                            logits, tgt,
                            _tp.gather_from_tensor_model_parallel_region)
                    main_logits_e, _ = model.output_layer(dec_e)
                    main_logits_e = main_logits_e[0] if isinstance(main_logits_e, tuple) \
                        else main_logits_e
                    mtp_e = model.mtp(input_ids=eval_batch["input_ids"],
                                      position_ids=eval_batch["position_ids"],
                                      hidden_states=dec_e, attention_mask=None,
                                      inference_params=None, embedding=model.embedding)
                    ch_e = torch.chunk(mtp_e, 1 + cfg.mtp_num_layers, dim=0)
                    draft_logits_e, _ = model.output_layer(ch_e[-1])
                    draft_logits_e = draft_logits_e[0] if isinstance(draft_logits_e, tuple) \
                        else draft_logits_e
                    ce_main_e, _ = _fixed_diag(main_logits_e)
                    ce_draft_e, agree_e = _fixed_diag(draft_logits_e)
                if rank == 0:
                    print(f"[v10-trainer {_stamp()}] diag step {step}: EVAL-FIXED "
                          f"main CE={ce_main_e:.4f} draft CE={ce_draft_e:.4f} "
                          f"argmax-agree={agree_e:.4f} "
                          f"(same tokens every diag — the trend)", flush=True)

            if not recipe_mode and args.diag_every and step % args.diag_every == 0 \
                    and getattr(model, "_last_main_ce", None) is not None:
                try:
                    _run_diag()
                except Exception as _e:
                    print(f"[v10-trainer {_stamp()}] diag FAILED (non-fatal, training "
                          f"continues): {type(_e).__name__}: {_e}", flush=True)
        if step % args.cp_interval == 0 or step == args.max_steps:
            try:
                save_cp(model, optimizer, step, out_dir, master=master)
            except Exception as _e:
                print(f"[v10-trainer {_stamp()}] CP save FAILED (non-fatal, training "
                      f"continues): {type(_e).__name__}: {_e}", flush=True)

    if rank == 0:
        print(f"[v10-trainer {_stamp()}] done", flush=True)


if __name__ == "__main__":
    main()
