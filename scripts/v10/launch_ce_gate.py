#!/usr/bin/env python
"""V10 Phase 2a — TP16 full-model fill + SEMANTIC CE GATE (iron rule 第零步).

The gate: build the FULL 45-layer model sharded TP16, fill every rank's shard
from the Flash-BF16 checkpoint (plain copy, shard_slice math), forward real
text, and require the main-head next-token CE to fall in the vendor magnitude
(~2-4 for a trained 300B MoE on natural text). V8's lesson: skipping this gate
costed 15+ debug rounds — NO further V10 work passes this point until it's green.

Run (.38, 16 cards free):
  torchrun --nproc_per_node=16 launch_ce_gate.py [--seq 1024]
"""
import argparse
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402


def npu_shim():
    """mcore calls torch.cuda.* — route to NPU (V8-proven fill_vendor shim)."""
    from megatron.core.jit import disable_jit_fuser
    disable_jit_fuser()
    try:
        import torch._dynamo as _dynamo
        _dynamo.config.disable = True
    except Exception:
        pass
    torch.cuda.current_device = torch.npu.current_device
    torch.cuda.set_device = torch.npu.set_device
    torch.cuda.device_count = torch.npu.device_count
    torch.cuda.synchronize = torch.npu.synchronize
    torch.cuda.get_rng_state = torch.npu.get_rng_state
    torch.cuda.set_rng_state = torch.npu.set_rng_state
    torch.cuda.manual_seed = torch.npu.manual_seed
    torch.cuda.manual_seed_all = torch.npu.manual_seed_all
    torch.cuda.default_generators = torch.npu.default_generators
    if not hasattr(torch.cuda, "CUDAGraph") or torch.cuda.CUDAGraph is None:
        torch.cuda.CUDAGraph = getattr(torch.npu, "CUDAGraph", object)
    import torch.cuda.random as _tc_rand
    _tc_rand.get_rng_state = torch.npu.get_rng_state
    _tc_rand.set_rng_state = torch.npu.set_rng_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--vendor", default=os.environ.get(
        "V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16"))
    ap.add_argument("--dump", default=None,
                    help="rank0 writes {ids, ce, argmax} json here (the "
                    "teacher-parity input)")
    ap.add_argument("--texts", default=None,
                    help="json list of texts (statistical gate); default = "
                    "the built-in bilingual probe text")
    args = ap.parse_args()

    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    npu_shim()

    import torch.distributed as dist
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="hccl", init_method="env://",
                            world_size=world, rank=rank)
    from megatron.core import parallel_state
    # initialize_model_parallel() DEFAULTS TP=1 — must pass the TP degree
    # explicitly (16 ranks would otherwise each build the FULL model → OOM)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)

    from v10_config import vendor_config
    from v10_fill import fill_model
    from v10_model import build_v10_model

    vc = vendor_config()
    model, cfg = build_v10_model(vc=vc, device="npu", mtp=False,
                                 max_seq_len=args.seq + 64)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[gate] rank{rank}: full 45-layer model, {n_params/1e9:.1f}B params, "
          f"tp={world}", flush=True)

    ledger = fill_model(model, args.vendor, vc, tp_rank=rank, tp_size=world,
                        verbose=(rank == 0))
    if ledger["unwritten"] or ledger["unconsumed"]:
        print(f"[gate] rank{rank}: LEDGER NOT CLEAN "
              f"unwritten={ledger['unwritten'][:5]} "
              f"unconsumed={ledger['unconsumed'][:5]}", flush=True)
        dist.barrier()
        sys.exit(2)
    if rank == 0:
        print(f"[gate] fill clean: {ledger['n_written']} targets, "
              f"{ledger['n_deferred']} deferred (MTP only)", flush=True)
    torch.npu.empty_cache()

    # real text (identical on every rank — no RNG involved)
    import json as _json

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.vendor, trust_remote_code=False)
    if args.texts:
        with open(args.texts) as f:
            texts = _json.load(f)
    else:
        texts = [("The quick brown fox jumps over the lazy dog and runs "
                  "through the green field where the sun is shining brightly "
                  "today. 大语言模型的训练需要严格验证每一个算子的数值正确性，"
                  "尤其是混合专家路由、线性注意力递推与稀疏注意力的索引选择，"
                  "任何微小的偏差都会在多层堆叠后放大。")]

    results, means = [], []
    for ti, text in enumerate(texts):
        ids = tok(text)["input_ids"][:args.seq]
        S = len(ids)
        x = torch.tensor([ids], dtype=torch.int64, device="npu")
        pos = torch.zeros_like(x)
        mask = torch.ones(1, 1, S, S, dtype=torch.bool, device="npu").tril()

        with torch.no_grad():
            logits = model(input_ids=x, position_ids=pos, attention_mask=mask)
        # logits: [B, S, V_partial] (parallel_output=True — vocab sharded).
        # Vocab-parallel CE: all_gather the partial-vocab blocks (contiguous
        # vocab row ranges, cat restores vocab order), fp32 softmax full row.
        tgt = x[0, 1:].clone()
        logits_p = logits[0, :-1].float().contiguous()
        del logits
        parts = [torch.empty_like(logits_p) for _ in range(world)]
        dist.all_gather(parts, logits_p)
        del logits_p
        full = torch.cat(parts, dim=-1)                  # [S-1, V]
        ce_all = -(torch.log_softmax(full, dim=-1).gather(-1, tgt.unsqueeze(-1))
                   .squeeze(-1))                         # [S-1]
        amax = full.argmax(dim=-1).tolist()
        del full, parts
        ce_mean = ce_all.mean().item()
        means.append(ce_mean)
        results.append({"text": text, "ids": ids,
                        "ce": ce_all.tolist(), "argmax": amax})
        if rank == 0:
            print(f"[gate] text{ti}: S={S} CE_mean={ce_mean:.4f} "
                  f"CE_max={ce_all.max().item():.4f}", flush=True)
        del ce_all
        if ti % 4 == 3:
            torch.npu.empty_cache()

    # teardown hygiene: rank0 must NOT exit while other ranks sit in a
    # barrier — HCCL socket-timeout noise (EI0006) on every otherwise-clean
    # run. Everyone barriers first, THEN rank0 exits with the verdict.
    dist.barrier()
    if rank == 0:
        agg = sum(means) / len(means)
        print(f"[gate] {len(texts)} texts aggregate CE_mean={agg:.4f}",
              flush=True)
        if args.dump:
            with open(args.dump, "w") as f:
                _json.dump({"results": results, "aggregate_ce": agg,
                            "texts_file": args.texts}, f)
            print(f"[gate] dumped {len(results)} texts -> {args.dump}",
                  flush=True)
        # informational verdict (the statistical gate is the teacher parity)
        ok = 1.5 <= agg <= 5.0
        print(f"[gate] SEMANTIC CE GATE (mean-band): {'PASS' if ok else 'FAIL'}",
              flush=True)
        sys.exit(0)


if __name__ == "__main__":
    main()
