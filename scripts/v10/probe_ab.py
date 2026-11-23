import json, os, sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch
from launch_ce_gate import npu_shim
from v10_diag import gather_full_logits


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

    vc = vendor_config()
    model, cfg = build_v10_model(vc=vc, device="npu", mtp=True,
                                 max_seq_len=2048, tp_size=world)
    ledger = fill_model(model, os.environ.get(
        "V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16"), vc, mtp=True,
        tp_rank=rank, tp_size=world)
    assert not ledger["unwritten"] and not ledger["unconsumed"]
    model.eval()

    with open("h0-test/eval35/v10_mcore_ce16p_40.json") as f:
        ids = torch.tensor([json.load(f)["results"][0]["ids"]],
                           dtype=torch.int64, device="npu")
    S = ids.shape[1]
    layer0 = model.mtp.layers[0]
    tgt = ids[0, 1:]                                   # [S-1]

    with torch.no_grad():
        # ---- path A: eval_cp50_40.py's fixed_diag draft computation ----
        emb_a = model.embedding(input_ids=ids, position_ids=torch.zeros_like(ids))
        emb_a = emb_a[0] if isinstance(emb_a, tuple) else emb_a
        dec_a = model.decoder(hidden_states=emb_a, attention_mask=None)
        dec_a = dec_a[0] if isinstance(dec_a, tuple) else dec_a
        mtp_out_a = model.mtp(input_ids=ids,
                              position_ids=torch.zeros_like(ids),
                              hidden_states=dec_a, attention_mask=None,
                              inference_params=None, embedding=model.embedding)
        chunks_a = torch.chunk(mtp_out_a, 2, dim=0)
        d_a = chunks_a[-1]
        lg_a, _ = model.output_layer(d_a)
        lg_a = lg_a[0] if isinstance(lg_a, tuple) else lg_a
        if lg_a.dim() == 3:
            lg_a = lg_a[:, 0]
        full_a = gather_full_logits(lg_a[:S - 1], dim=-1).float()
        lp_a = torch.log_softmax(full_a, dim=-1)
        ce_a = (-lp_a.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).mean().item()
        am_a = full_a.argmax(-1)
        agree_a = (am_a == tgt).float().mean().item()

        # ---- path B: depth_chain's D1 (identical math, per the script) ----
        emb_b = model.embedding(input_ids=ids, position_ids=torch.zeros_like(ids))
        emb_b = emb_b[0] if isinstance(emb_b, tuple) else emb_b
        dec_b = model.decoder(hidden_states=emb_b, attention_mask=None)
        dec_b = dec_b[0] if isinstance(dec_b, tuple) else dec_b
        ids_k = torch.roll(ids, shifts=0, dims=-1)     # k=1: no roll
        mtp_out_b = model.mtp(input_ids=ids_k,
                              position_ids=torch.zeros_like(ids),
                              hidden_states=dec_b, attention_mask=None,
                              inference_params=None, embedding=model.embedding)
        chunks_b = torch.chunk(mtp_out_b, 2, dim=0)
        d_b = chunks_b[-1]
        lg_b, _ = model.output_layer(d_b)
        lg_b = lg_b[0] if isinstance(lg_b, tuple) else lg_b
        if lg_b.dim() == 3:
            lg_b = lg_b[:, 0]
        full_b = gather_full_logits(lg_b[:S - 1], dim=-1).float()
        lp_b = torch.log_softmax(full_b, dim=-1)
        ce_b = (-lp_b.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).mean().item()

        # ---- the intermediates ----
        e_a = model.embedding(input_ids=ids, position_ids=torch.zeros_like(ids))
        e_a = e_a[0] if isinstance(e_a, tuple) else e_a
        x_a = layer0.eh_proj(torch.cat([layer0.enorm(e_a),
                                        layer0.hnorm(dec_a)], dim=-1))
        x_a = x_a[0] if isinstance(x_a, tuple) else x_a
        e_b = model.embedding(input_ids=ids_k, position_ids=torch.zeros_like(ids_k))
        e_b = e_b[0] if isinstance(e_b, tuple) else e_b
        x_b = layer0.eh_proj(torch.cat([layer0.enorm(e_b),
                                        layer0.hnorm(dec_b)], dim=-1))
        x_b = x_b[0] if isinstance(x_b, tuple) else x_b

    if rank == 0:
        print(f"[probe] path A (cp50-eval draft) CE = {ce_a:.4f}", flush=True)
        print(f"[probe] path B (depth_chain D1)  CE = {ce_b:.4f}", flush=True)
        print(f"[probe] dec norm: A={dec_a.float().norm():.2f} B={dec_b.float().norm():.2f} "
              f"same-object={dec_a is dec_b}", flush=True)
        print(f"[probe] x norm: A={x_a.float().norm():.2f} B={x_b.float().norm():.2f} "
              f"maxdiff={(x_a.float()-x_b.float()).abs().max():.4f}", flush=True)
        print(f"[probe] d norm: A={d_a.float().norm():.2f} B={d_b.float().norm():.2f} "
              f"maxdiff={(d_a.float()-d_b.float()).abs().max():.4f}", flush=True)
        print(f"[probe] lg maxdiff A-vs-B: {(lg_a.float()-lg_b.float()).abs().max():.4f}",
              flush=True)
        print(f"[probe] mtp block type: {type(model.mtp).__name__} "
              f"layers[0] type: {type(model.mtp.layers[0]).__name__}", flush=True)
    dist.barrier()


if __name__ == "__main__":
    main()
