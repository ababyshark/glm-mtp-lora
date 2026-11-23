#!/usr/bin/env python
"""CP50 effect evaluator — the T4 probe verdict (user directive 2026-11-20).

Baseline: fresh fill → the fixed-token diag (main/draft CE + argmax-agree).
Trained:  load_cp(cp50) (the mtp+LoRA fp32 masters restored, model synced)
          → the SAME diag → the deltas.

The CP is the 50-step LoRA(base)+full-MTP probe (clip=5.0, p.grad-accumulation
fixed mid-run — the CP holds the post-fix state at step 50).

Run (.40 mtpV8): torchrun --nproc_per_node=16 eval_cp50_40.py
"""
import json
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

from launch_ce_gate import npu_shim  # noqa: E402
from v10_diag import gather_full_logits  # noqa: E402


def fixed_diag(model, ids, cfg, chunk=256):
    """The fixed-token diag: (main CE, draft CE, argmax-agree), no_grad.
    tgt [S-1, 1] aligns with the logits [S-1, 1, V] (B=1)."""
    with torch.no_grad():
        emb = model.embedding(input_ids=ids, position_ids=torch.zeros_like(ids))
        emb = emb[0] if isinstance(emb, tuple) else emb
        dec = model.decoder(hidden_states=emb, attention_mask=None)
        dec = dec[0] if isinstance(dec, tuple) else dec
        main_logits, _ = model.output_layer(dec)
        main_logits = main_logits[0] if isinstance(main_logits, tuple) \
            else main_logits
        mtp_out = model.mtp(input_ids=ids, position_ids=torch.zeros_like(ids),
                            hidden_states=dec, attention_mask=None,
                            inference_params=None, embedding=model.embedding)
        chunks = torch.chunk(mtp_out, 1 + cfg.mtp_num_layers, dim=0)
        draft_logits, _ = model.output_layer(chunks[-1])
        draft_logits = draft_logits[0] if isinstance(draft_logits, tuple) \
            else draft_logits
        tgt = ids[0, 1:].unsqueeze(-1)                     # [S-1, 1]
        ces, ams = [], []
        for lg in (main_logits[:-1], draft_logits[:-1]):
            ce_v = n = 0
            am = []
            for s0 in range(0, lg.shape[0], chunk):
                seg = lg[s0:s0 + chunk]
                full = gather_full_logits(seg, dim=-1).float()
                lp = torch.log_softmax(full, dim=-1)
                ce_v += (-lp.gather(-1, tgt[s0:s0 + chunk].unsqueeze(-1))
                         .squeeze(-1)).sum()
                n += tgt[s0:s0 + chunk].numel()
                am.append(full.argmax(-1))
            ces.append((ce_v / max(n, 1)).item())
            ams.append(torch.cat(am))
        agree = (ams[0] == ams[1]).float().mean().item()
    return ces[0], ces[1], agree


def main():
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
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

    vc = vendor_config()
    model, cfg = build_v10_model(vc=vc, device="npu", mtp=True,
                                 max_seq_len=2048, tp_size=world)
    ledger = fill_model(model, os.environ.get(
        "V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16"), vc, mtp=True,
        tp_rank=rank, tp_size=world)
    assert not ledger["unwritten"] and not ledger["unconsumed"]
    build_trainable(model)
    model.eval()

    with open("h0-test/eval35/v10_mcore_ce16p_40.json") as f:
        ids = torch.tensor([json.load(f)["results"][0]["ids"]],
                           dtype=torch.int64, device="npu")

    b_main, b_draft, b_agree = fixed_diag(model, ids, cfg)
    if rank == 0:
        print(f"[cp50-eval] BASELINE (fresh fill): main CE={b_main:.4f} "
              f"draft CE={b_draft:.4f} argmax-agree={b_agree:.4f}", flush=True)

    params_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    master = {n: p.detach().clone().float() for n, p in params_named}
    for t in master.values():
        t.requires_grad_(False)
    optimizer = torch.optim.AdamW([master[n] for n, _ in params_named], lr=5e-6)
    cp_dir = os.environ.get("V10_CP_DIR", "saves/v10_t4")
    cp_file = os.path.join(cp_dir, f"cp50_rank{rank}.pt")
    step = load_cp(model, optimizer, cp_file, master=master)
    if rank == 0:
        print(f"[cp50-eval] CP loaded: step={step} <- {cp_file}", flush=True)

    t_main, t_draft, t_agree = fixed_diag(model, ids, cfg)
    if rank == 0:
        print(f"[cp50-eval] TRAINED (cp{step}):          main CE={t_main:.4f} "
              f"draft CE={t_draft:.4f} argmax-agree={t_agree:.4f}", flush=True)
        print(f"[cp50-eval] DELTA: main {t_main-b_main:+.4f} "
              f"draft {t_draft-b_draft:+.4f} agree {t_agree-b_agree:+.4f}", flush=True)
        verdict = []
        if t_main < b_main:
            verdict.append("main-improved")
        if t_draft < b_draft:
            verdict.append("draft-improved")
        if t_agree > b_agree:
            verdict.append("agree-improved")
        print(f"[cp50-eval] VERDICT: {', '.join(verdict) if verdict else 'NO improvement detected'}",
              flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
