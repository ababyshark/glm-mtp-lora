#!/usr/bin/env python
"""eh_proj A/B + wiring diagnostic — settles WHERE the native 17.09 lives.

One run prints the decision matrix:
  1. wiring facts:  hasattr(mtp), mtp_process, cfg.mtp_num_layers
  2. main-head CE   (labels=None forward; expect ~2.7)
  3. native loss    (labels path; baseline 17.09)
  4. eh checksum + toggle -> checksum -> native loss again (in-graph?)
  5. zero eh_proj   -> native loss again (definitive in/out-of-graph)
  6. DIRECT draft CE: manual mtp call -> last chunk -> output_layer -> CE vs
     rolled labels (bypasses all wiring questions)

Run (.40 mtpV8): torchrun --nproc_per_node=16 eh_zero_shot_probe.py
"""
import argparse
import json
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

from launch_ce_gate import npu_shim  # noqa: E402


def _scalar(out):
    loss = out[0] if isinstance(out, tuple) else out
    if loss.dim() > 0:
        loss = loss.float().mean()
    return float(loss)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="h0-test/eval35/v10_mcore_ce16p_40.json")
    ap.add_argument("--text", type=int, default=0)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_rank)
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

    vc = vendor_config()
    model, cfg = build_v10_model(vc=vc, device="npu", mtp=True,
                                 max_seq_len=2048, tp_size=world)
    ledger = fill_model(model, os.environ.get(
        "V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16"), vc, mtp=True,
        tp_rank=rank, tp_size=world)
    assert not ledger["unwritten"] and not ledger["unconsumed"], \
        f"fill ledger not clean: {ledger['unwritten'][:3]}"
    model.eval()

    with open(args.dump) as f:
        ids = torch.tensor([json.load(f)["results"][args.text]["ids"]],
                           dtype=torch.int64, device="npu")
    S = ids.shape[1]

    facts = {
        "has_mtp": hasattr(model, "mtp"),
        "mtp_process": bool(getattr(model, "mtp_process", False)),
        "cfg_mtp_num_layers": getattr(cfg, "mtp_num_layers", "MISSING"),
    }

    # main-head CE (no labels path)
    with torch.no_grad():
        out = model(input_ids=ids, position_ids=torch.zeros_like(ids),
                    attention_mask=None)
        logits = out[0] if isinstance(out, tuple) else out
        logits = logits.float()
        tgt = ids[0, 1:]
        main_ce = -(torch.log_softmax(logits[0, :-1], dim=-1)
                    .gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).mean().item()

    if rank == 0:
        print(f"[eh-diag] facts={facts}", flush=True)
        print(f"[eh-diag] main-head CE = {main_ce:.4f}   (gate said ~2.69)",
              flush=True)
    with torch.no_grad():
        native0 = _scalar(model(input_ids=ids, position_ids=torch.zeros_like(ids),
                                attention_mask=None, labels=ids))

    eh = model.mtp.layers[0].eh_proj.weight
    c0 = eh.detach().float().sum().item()
    h2 = eh.shape[1] // 2
    with torch.no_grad():
        eh.data.copy_(torch.cat([eh[:, h2:], eh[:, :h2]], dim=1))
    c1 = eh.detach().float().sum().item()
    with torch.no_grad():
        native1 = _scalar(model(input_ids=ids, position_ids=torch.zeros_like(ids),
                                attention_mask=None, labels=ids))
    saved = eh.detach().clone()
    with torch.no_grad():
        eh.data.zero_()
    with torch.no_grad():
        native_zero = _scalar(model(input_ids=ids, position_ids=torch.zeros_like(ids),
                                    attention_mask=None, labels=ids))
    with torch.no_grad():
        eh.data.copy_(saved)

    # DIRECT draft CE: manual mtp-block call, mirror gpt_model forward wiring
    with torch.no_grad():
        emb = model.embedding(input_ids=ids, position_ids=torch.zeros_like(ids))
        emb = emb[0] if isinstance(emb, tuple) else emb
        dec = model.decoder(hidden_states=emb, attention_mask=None)
        dec = dec[0] if isinstance(dec, tuple) else dec        # [S, B, H]
        mtp_out = model.mtp(input_ids=ids, position_ids=torch.zeros_like(ids),
                            hidden_states=dec, attention_mask=None,
                            inference_params=None, embedding=model.embedding)
        chunks = torch.chunk(mtp_out, 1 + cfg.mtp_num_layers, dim=0)
        lg, _ = model.output_layer(chunks[-1])
        lg = lg[0] if isinstance(lg, tuple) else lg
        lg = lg.float()
        if lg.dim() == 3:
            lg = lg[:, 0]                       # [S, B, V] -> [S, V] (B=1)
        tgt_r = torch.roll(ids[0], shifts=-1, dims=-1)
        direct_ce = -(torch.log_softmax(lg, dim=-1)
                      .gather(-1, tgt_r.unsqueeze(-1)).squeeze(-1)).mean().item()

    if rank == 0:
        print(f"[eh-diag] native loss baseline = {native0:.4f}", flush=True)
        print(f"[eh-diag] eh checksum {c0:.4f} -> {c1:.4f} "
              f"(changed={abs(c1-c0) > 1e-6})  native after toggle = {native1:.4f}",
              flush=True)
        print(f"[eh-diag] native with eh ZEROED = {native_zero:.4f} "
              f"(== baseline => eh OUT of graph)", flush=True)
        print(f"[eh-diag] DIRECT draft CE (manual mtp call) = {direct_ce:.4f}",
              flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
