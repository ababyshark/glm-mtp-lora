#!/usr/bin/env python
"""V10 NPU 1-card smoke: truncated real-dims build + REAL Flash-BF16 fill +
real-text forward. Validates on Ascend: meta routing -> to_empty, dtype
upcasts from the real checkpoint (A_log/dt_bias fp32, conv bf16->fp32,
hc fn bf16->fp32, expert_bias fp32), KDA depthwise conv1d, mHC sinkhorn,
DSA topk, MoE, NPU mask discipline.

NOT the semantic CE gate (truncated model CE is not vendor-magnitude — the
real gate is the Phase-2 full TP16 build). This smoke asserts: finite CE on
real text, weight shapes == real dims, fill coverage for the built layers.

Usage (.38): torchrun --nproc_per_node=1 npu_smoke.py [--layers 2]
"""
import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from v10_config import vendor_config  # noqa: E402
from v10_fill import fill_model  # noqa: E402
from v10_model import build_v10_model  # noqa: E402

VENDOR_DIR = os.environ.get("V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16")


def real_token_ids(n=64):
    """A short real-text token sequence via the Flash tokenizer (fast rust
    tokenizer; no model code needed)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(VENDOR_DIR, trust_remote_code=False)
    text = ("The quick brown fox jumps over the lazy dog and runs through the "
            "field. 昇腾训练框架需要验证线性注意力与稀疏注意力的数值正确性，"
            "同时覆盖混合专家路由与超连接的门控语义。")
    ids = tok(text, return_tensors=None)["input_ids"]
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=4)
    args = ap.parse_args()

    vc = vendor_config()
    # 4 layers = kda-dense ×3 (layers 0-2) + dsa-moe (layer 3) — exercises all
    # four module families at REAL dims in one truncation
    model, cfg = build_v10_model(vc=vc, num_layers=args.layers, device="npu",
                                 mtp=False, max_seq_len=1024)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] built {args.layers}-layer truncation: "
          f"{n_params/1e6:.1f}M params on {torch.npu.current_device()}")

    ledger = fill_model(model, VENDOR_DIR, vc, verbose=True)
    assert not ledger["unwritten"], f"unwritten: {ledger['unwritten'][:10]}"
    assert not ledger["unconsumed"], f"unconsumed: {ledger['unconsumed'][:10]}"
    print(f"[smoke] fill ledger clean: {ledger['n_written']} targets, "
          f"{ledger['n_deferred']} deferred (rest-of-model)")

    # real-dims spot checks
    P = dict(model.named_parameters())
    li = 0  # KDA layer 0
    assert tuple(P[f"decoder.layers.{li}.self_attention.linear_q_proj.weight"].shape) \
        == (vc["linear_attn_config"]["num_heads"] *
            vc["linear_attn_config"]["head_dim"], vc["hidden_size"])
    assert P[f"decoder.layers.{li}.self_attention.conv1d_weight"].dtype == torch.float32
    assert P[f"decoder.layers.{li}.self_attention.A_log"].dtype == torch.float32
    assert P[f"decoder.layers.{li}.attn_hc.fn"].dtype == torch.float32
    if args.layers > 3:
        dsa = f"decoder.layers.3.self_attention."
        assert tuple(P[dsa + "kv_b_proj.weight"].shape) == \
            (cfg.num_attention_heads * (cfg.qk_head_dim + cfg.v_head_dim),
             cfg.kv_lora_rank)
        assert P[dsa + "indexer.index_kpool_compress_ape"].shape[0] == \
            cfg.dsa_indexer_kpool
        assert tuple(P["decoder.layers.3.mlp.experts.local_experts.0.linear_fc1.weight"]
                     .shape) == (2 * vc["moe_intermediate_size"], vc["hidden_size"])

    ids = real_token_ids()
    B = 1
    x = torch.tensor([ids], dtype=torch.int64, device="npu")     # [B,S] batch-first
    pos = torch.zeros_like(x)
    S = x.shape[1]
    mask = torch.ones(1, 1, S, S, dtype=torch.bool, device="npu").tril()
    with torch.no_grad():
        logits = model(input_ids=x, position_ids=pos, attention_mask=mask)
    assert logits.shape[:2] == (B, S), f"batch-first logits, got {tuple(logits.shape)}"
    assert torch.isfinite(logits.float()).all(), "non-finite logits"

    # next-token CE on the real sequence (teacher forcing, shift)
    logp = torch.log_softmax(logits[0, :-1].float(), dim=-1)
    tgt = x[0, 1:]
    ce = -(logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).mean()
    print(f"[smoke] truncation={args.layers} layers, S={S}: "
          f"CE={ce.item():.4f} (truncated-weights model — NOT the semantic gate)")

    # backward smoke (grad-enabled forward)
    logits_g = model(input_ids=x, position_ids=pos, attention_mask=mask)
    logits_g.float().sum().backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"[smoke] backward OK: {n_grad} params with grads")

    # determinism
    with torch.no_grad():
        l2 = model(input_ids=x, position_ids=pos, attention_mask=mask)
    assert torch.equal(logits, l2), "npu forward must be deterministic"
    print("[smoke] determinism OK")
    print("[v10-npu-smoke] PASS")


if __name__ == "__main__":
    main()
