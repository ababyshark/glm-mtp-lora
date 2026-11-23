#!/usr/bin/env python
"""Step0 baseline probe — anchor the fixed-token (train_ds[0]) diag BEFORE any
training, so a V10.2 run's health can be read at a glance from the trainer's
every-10-step EVAL-FIXED line, without waiting for the ~2.5h vLLM serving round.

Background (docs/V10_TRAIN.md §11): the CE-only run silently degraded serving
acceptance (K̄ 3.52→2.49) while the mcore probes reported "improvement" — the
probes measured the training objective itself (tautological). The EVAL-FIXED
diag is the fixed-token probe: SAME tokens every call, so its TREND (not its
absolute value) is the health signal. This script captures the STEP0 anchor of
that trend — the number the trend starts from.

Two argmax metrics, kept DISTINCT (the trainer's "argmax-agree" label conflates
them across the two diag lines):
  draft-vs-gold  = P(argmax(draft) == gold)         — CE-hit-rate flavor; what
                   the trainer's EVAL-FIXED "argmax-agree" currently prints.
  draft-vs-main  = P(argmax(draft) == argmax(main)) — the acceptance-relevant
                   metric (V10.2 trains this: acceptance counts MAIN's argmax).

Usage (.34/.40, 16 NPU, same env as the trainer):
  export V10_VENDOR_DIR=/root/serve_v101_cp100   # or the real vendor BF16 dir
  torchrun --nproc_per_node=16 --master_port=29xxx scripts/v10/eval_step0_baseline.py \
      --yaml configs/glm53_v10_train.yaml
  # optional: also snapshot a trained CP and diff against step0
  torchrun ... eval_step0_baseline.py --yaml ... \
      --load-cp saves/v10_t4/cp100_rank{rank}.pt

Rank0 prints:
  STEP0 (fresh fill @ $V10_VENDOR_DIR): main CE / draft CE / draft-vs-gold / draft-vs-main
  [LOADED (cp{N})]: same four + per-metric delta vs STEP0.
"""
import argparse
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

from launch_ce_gate import npu_shim  # noqa: E402


def argmax_agree_draft_vs_main(main_logits, draft_logits, gather_fn, chunk=256):
    """P(argmax(draft) == argmax(main)) over rows 0..S-2, chunked full-vocab.
    main_logits/draft_logits [S, B, V/tp]; returns a float (0..1)."""
    with torch.no_grad():
        hit = n = 0
        for s0 in range(0, main_logits.shape[0] - 1, chunk):
            fm = gather_fn(main_logits[s0:s0 + chunk].contiguous()).float()
            fd = gather_fn(draft_logits[s0:s0 + chunk].contiguous()).float()
            am_m = fm.argmax(-1).reshape(-1)
            am_d = fd.argmax(-1).reshape(-1)
            hit += (am_m == am_d).sum()
            n += am_m.numel()
        return (hit / max(n, 1)).item()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--yaml", required=True, help="LF data config (same as trainer)")
    p.add_argument("--seq-len", type=int, default=None,
                   help="override cutoff_len (default: yaml cutoff_len)")
    p.add_argument("--vendor", default=os.environ.get(
        "V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16"))
    p.add_argument("--load-cp", default=None,
                   help="optional trained CP glob (e.g. saves/v10_t4/cp100_rank{rank}.pt)")
    p.add_argument("--tp-size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    dev = torch.device("npu", int(os.environ.get("LOCAL_RANK", "0")))
    npu_shim()

    import torch.distributed as dist
    dist.init_process_group(backend="hccl", init_method="env://")
    from megatron.core import parallel_state
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=args.tp_size)
    rank = dist.get_rank()
    world = dist.get_world_size()
    assert world == args.tp_size, f"Step0 probe = TP{args.tp_size}/DP1 (world={world})"

    from megatron.core import tensor_parallel as _tp

    from v10_config import vendor_config
    from v10_fill import fill_model
    from v10_model import build_v10_model
    from v10_trainer import build_trainable, fixed_diag_kernel, load_cp
    from v10_data import load_lf_data

    import yaml as _yaml
    seq_len = args.seq_len or int(_yaml.safe_load(open(args.yaml)).get("cutoff_len", 2048))

    vc = vendor_config()
    model, cfg = build_v10_model(vc=vc, device="npu", mtp=True,
                                 max_seq_len=2048, tp_size=args.tp_size)
    ledger = fill_model(model, args.vendor, vc, mtp=True, tp_rank=rank,
                        tp_size=args.tp_size, verbose=(rank == 0))
    if ledger["unwritten"] or ledger["unconsumed"]:
        raise RuntimeError(f"fill ledger not clean: unwritten={ledger['unwritten'][:5]} "
                           f"unconsumed={ledger['unconsumed'][:5]}")
    build_trainable(model)
    model.eval()

    # fixed eval tokens = train_ds[0], EXACTLY the trainer's eval_batch (pad to seq_len)
    _tok, _tpl, train_ds = load_lf_data(args.yaml)
    row0 = train_ds[0]
    ids0 = row0["input_ids"] if "input_ids" in row0 else row0
    t0 = torch.tensor(ids0[:seq_len], dtype=torch.long)
    eb = torch.full((1, seq_len), 0, dtype=torch.long, device=dev)
    eb[0, :t0.shape[0]] = t0
    if rank == 0:
        print(f"[step0-baseline] vendor={args.vendor} seq_len={seq_len} "
              f"real_tokens={t0.shape[0]}", flush=True)

    gather_fn = _tp.gather_from_tensor_model_parallel_region

    def run_diag(tag):
        with torch.no_grad():
            emb = model.embedding(input_ids=eb,
                                  position_ids=torch.arange(seq_len, device=dev)
                                  .unsqueeze(0).contiguous())
            emb = emb[0] if isinstance(emb, tuple) else emb
            dec = model.decoder(hidden_states=emb, attention_mask=None)
            dec = dec[0] if isinstance(dec, tuple) else dec
            main_logits, _ = model.output_layer(dec)
            main_logits = main_logits[0] if isinstance(main_logits, tuple) \
                else main_logits
            mtp_out = model.mtp(input_ids=eb,
                                position_ids=torch.arange(seq_len, device=dev)
                                .unsqueeze(0).contiguous(),
                                hidden_states=dec, attention_mask=None,
                                inference_params=None, embedding=model.embedding)
            ch = torch.chunk(mtp_out, 1 + cfg.mtp_num_layers, dim=0)
            draft_logits, _ = model.output_layer(ch[-1])
            draft_logits = draft_logits[0] if isinstance(draft_logits, tuple) \
                else draft_logits

            tgt = eb[0, 1:]
            main_ce, _ = fixed_diag_kernel(main_logits, tgt, gather_fn)
            draft_ce, draft_vs_gold = fixed_diag_kernel(draft_logits, tgt, gather_fn)
            draft_vs_main = argmax_agree_draft_vs_main(main_logits, draft_logits, gather_fn)
        return main_ce, draft_ce, draft_vs_gold, draft_vs_main

    s0 = run_diag("STEP0")
    if rank == 0:
        print(f"[step0-baseline] STEP0 (fresh fill @ {args.vendor}): "
              f"main CE={s0[0]:.4f} draft CE={s0[1]:.4f} "
              f"draft-vs-gold={s0[2]:.4f} draft-vs-main={s0[3]:.4f}", flush=True)

    if args.load_cp:
        params_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        master = {n: p.detach().clone().float() for n, p in params_named}
        for t in master.values():
            t.requires_grad_(False)
        optimizer = torch.optim.AdamW([master[n] for n, _ in params_named], lr=5e-6)
        cp_file = args.load_cp.replace("{rank}", os.environ.get("RANK", "0"))
        step = load_cp(model, optimizer, cp_file, master=master, with_opt=False)
        s1 = run_diag("LOADED")
        if rank == 0:
            print(f"[step0-baseline] LOADED (cp{step}): "
                  f"main CE={s1[0]:.4f} draft CE={s1[1]:.4f} "
                  f"draft-vs-gold={s1[2]:.4f} draft-vs-main={s1[3]:.4f}", flush=True)
            print(f"[step0-baseline] DELTA: main {s1[0]-s0[0]:+.4f} "
                  f"draft {s1[1]-s0[1]:+.4f} "
                  f"draft-vs-gold {s1[2]-s0[2]:+.4f} "
                  f"draft-vs-main {s1[3]-s0[3]:+.4f}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
