#!/usr/bin/env python
"""D1-D5 depth-acceptance evaluation — 5 CP points × 5 depths.

For each CP (baseline/cp250/cp500/cp750/cp1000): the recurrent MTP chain
(DSv3 convention) unrolled 5 depths:
  depth k input: [h_prev(t); e(t+k-1)]   (h_prev chains: depth output → next
  depth's hidden; e(t+k-1) = embedding(roll(ids, -(k-1))) — the no-roll patch
  makes the caller own the roll)
  output: logits_k predicting ids[t+k]
Metrics per (CP, depth): the accept proxy = P(argmax == ids[t+k]) + the CE.

Output: the accept matrix + the CE matrix + the JSON dump.
NOTE: D2-D5 are UNTRAINED horizons in the depth-1 training run — this
evaluation QUANTIFIES the multi-depth training gap (the V11 recipe
migration's motivation data).

Run (.40 mtpV8, after the training frees the 16 cards):
  torchrun --nproc_per_node=16 eval_d15_40.py
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

DEPTHS = 5


def depth_chain_diag(model, ids, cfg, depths=DEPTHS, chunk=256):
    """The recurrent MTP chain, D1..D_depths.

    Returns (accept[D], ce[D]) — the accept proxy = P(argmax==ids[t+k]),
    the CE at the horizon k. Positions t ∈ [0, S-k) per depth."""
    with torch.no_grad():
        S = ids.shape[1]
        dev = ids.device
        h_prev = None
        accept = []
        ces = []
        emb = model.embedding(input_ids=ids, position_ids=torch.zeros_like(ids))
        emb = emb[0] if isinstance(emb, tuple) else emb
        dec = model.decoder(hidden_states=emb, attention_mask=None)
        dec = dec[0] if isinstance(dec, tuple) else dec
        h_prev = dec                                        # h^(0) = the main hidden
        for k in range(1, depths + 1):
            ids_k = torch.roll(ids, shifts=-(k - 1), dims=-1)   # e(t+k-1)
            mtp_out = model.mtp(input_ids=ids_k,
                                position_ids=torch.zeros_like(ids),
                                hidden_states=h_prev, attention_mask=None,
                                inference_params=None, embedding=model.embedding)
            chunks = torch.chunk(mtp_out, 2, dim=0)
            d_k = chunks[-1]                                # the draft hidden
            logits_k, _ = model.output_layer(d_k)
            logits_k = logits_k[0] if isinstance(logits_k, tuple) else logits_k
            if logits_k.dim() == 3:
                logits_k = logits_k[:, 0]                   # B=1
            # the targets: ids[t+k] for t ∈ [0, S-k)
            n_pos = S - k
            tgt = ids[0, k:k + n_pos]
            acc_v = 0
            ce_v = 0.0
            for s0 in range(0, n_pos, chunk):
                end = min(s0 + chunk, n_pos)
                seg = logits_k[s0:end]
                full = gather_full_logits(seg, dim=-1).float()
                lp = torch.log_softmax(full, dim=-1)
                t = tgt[s0:end]
                ce_v += (-lp.gather(-1, t.unsqueeze(-1)).squeeze(-1)).sum()
                acc_v += (full.argmax(-1) == t).sum()
            accept.append((acc_v / max(n_pos, 1)).item())
            ces.append((ce_v / max(n_pos, 1)).item())
            h_prev = d_k.detach()                       # the chain ✓
    return accept, ces


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
    S = ids.shape[1]

    params_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    master = {n: p.detach().clone().float() for n, p in params_named}
    for t in master.values():
        t.requires_grad_(False)
    optimizer = torch.optim.AdamW([master[n] for n, _ in params_named], lr=5e-6)

    cp_list = [("baseline", None), ("cp250", 250), ("cp500", 500),
               ("cp750", 750), ("cp1000", 1000)]
    cp_dir = os.environ.get("V10_CP_DIR", "saves/v10_t4_long")

    acc_matrix, ce_matrix = [], []
    labels = []
    for name, step_n in cp_list:
        if step_n is not None:
            cp_file = os.path.join(cp_dir, f"cp{step_n}_rank{rank}.pt")
            if not os.path.exists(cp_file):
                if rank == 0:
                    print(f"[d15] SKIP {name}: {cp_file} not found", flush=True)
                labels.append(f"{name}(missing)")
                acc_matrix.append([float("nan")] * DEPTHS)
                ce_matrix.append([float("nan")] * DEPTHS)
                continue
            load_cp(model, optimizer, cp_file, master=master)
        acc, ces = depth_chain_diag(model, ids, cfg, depths=DEPTHS)
        acc_matrix.append(acc)
        ce_matrix.append(ces)
        labels.append(name)
        if rank == 0:
            print(f"[d15] {name}: accept=" +
                  "[" + ", ".join(f"{a:.4f}" for a in acc) + "] " +
                  "ce=[" + ", ".join(f"{c:.4f}" for c in ces) + "]", flush=True)

    if rank == 0:
        print(f"[d15] DEPTHS: D1..D{DEPTHS} (the horizon k; "
              f"the accept proxy = P(argmax_draft_k == ids[t+k]))", flush=True)
        print("[d15] accept matrix (rows=CP, cols=D1..D5):", flush=True)
        for lab, row in zip(labels, acc_matrix):
            print(f"  {lab:10s} [" + ", ".join(f"{v:.4f}" for v in row) + "]",
                  flush=True)
        with open("h0-test/eval35/v10_d15_matrix_40.json", "w") as f:
            json.dump({"labels": labels, "accept": acc_matrix,
                       "ce": ce_matrix, "S": S, "depths": DEPTHS}, f, indent=1)
        print("[d15] dumped -> h0-test/eval35/v10_d15_matrix_40.json", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
