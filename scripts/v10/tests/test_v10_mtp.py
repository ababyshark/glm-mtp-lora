#!/usr/bin/env python
"""V10 MTP layer tests (Phase 2b) — build/contract/order pins, tiny CPU.

Run on .12: python3 -m pytest scripts/v10/tests/test_v10_mtp.py -q -p no:warnings
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10 = os.path.dirname(HERE)
for p in (V10, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_v10_build import TINY_VC, cpu_env, _seed_params  # noqa: E402,F401


def _mcore_mtp_uses_mtp_model_layer() -> bool:
    """Version probe: the V10 MTP spec targets mcore 0.18.2, whose
    MultiTokenPredictionLayerSubmodules takes mtp_model_layer. Older mcores
    (e.g. 0.12.x) name it transformer_layer — skip cleanly there (run on the
    .38 venv_v7)."""
    try:
        import inspect

        from megatron.core.transformer.multi_token_prediction import (
            MultiTokenPredictionLayerSubmodules)
        return "mtp_model_layer" in inspect.signature(
            MultiTokenPredictionLayerSubmodules.__init__).parameters
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _mcore_mtp_uses_mtp_model_layer(),
    reason="mcore lacks MultiTokenPredictionLayerSubmodules(mtp_model_layer=...) "
           "(0.18.2 API); run on the .38 venv_v7")


def _tiny_mtp_model(cpu_env):
    from v10_model import build_v10_model
    model, cfg = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=True)
    _seed_params(model, seed=404)
    return model, cfg


def test_mtp_block_param_structure(cpu_env):
    """The MTP subtree keys must match the checkpoint layout (mapping §5):
    enorm/hnorm/eh_proj/final_layernorm(=shared_head.norm) + inner
    mtp_model_layer with input_layernorm/post_attention_layernorm + DSA + MoE,
    and NO hc_* anywhere."""
    model, cfg = _tiny_mtp_model(cpu_env)
    names = [n for n, _ in model.named_parameters() if n.startswith("mtp.")]
    joined = "\n".join(names)
    for must in ("mtp.layers.0.enorm.weight", "mtp.layers.0.hnorm.weight",
                 "mtp.layers.0.eh_proj.weight",
                 "mtp.layers.0.final_layernorm.weight",
                 "mtp.layers.0.mtp_model_layer.input_layernorm.weight",
                 "mtp.layers.0.mtp_model_layer.pre_mlp_layernorm.weight",
                 "mtp.layers.0.mtp_model_layer.self_attention.q_a_proj.weight",
                 "mtp.layers.0.mtp_model_layer.self_attention.indexer.",
                 "mtp.layers.0.mtp_model_layer.mlp.router.weight",
                 "mtp.layers.0.mtp_model_layer.mlp.experts.local_experts.0."):
        assert must in joined, f"missing MTP param: {must}"
    assert "hc_" not in joined, "MTP layer must have NO mHC (checkpoint: no hc keys)"
    # eh_proj: combined cat input -> [h, 2h]
    eh = dict(model.named_parameters())["mtp.layers.0.eh_proj.weight"]
    assert eh.shape == (TINY_VC["hidden_size"], 2 * TINY_VC["hidden_size"])


def test_mtp_forward_loss_finite(cpu_env):
    """GPTModel with mtp_block_spec + labels runs the MTP loss path."""
    from v10_model import build_v10_model

    model, cfg = _tiny_mtp_model(cpu_env)
    B, S = 2, 16
    V = TINY_VC["vocab_size"]
    ids = torch.randint(1, V, (B, S))
    pos = torch.zeros(B, S, dtype=torch.int64)
    labels = torch.roll(ids, -1, dims=1)
    mask = torch.ones(B, 1, S, S, dtype=torch.bool).tril()
    out = model(input_ids=ids, position_ids=pos, labels=labels,
                attention_mask=mask)
    loss = out[0] if isinstance(out, tuple) else out
    # calculate_per_token_loss=False -> per-token loss [B, S] (V8 smoke: mean)
    val = loss.float().mean()
    assert torch.isfinite(val), "MTP loss non-finite"
    val.backward()
    # MTP params got grads
    mtp_grads = [n for n, p in model.named_parameters()
                 if n.startswith("mtp.") and p.grad is not None]
    assert len(mtp_grads) > 10, f"MTP grads missing: {len(mtp_grads)}"


def test_mtp_kv_b_full_dims_double_split_regression(cpu_env):
    """Phase 2a probe: mcore Column/Row split by parallel_state TP themselves
    — custom modules must pass FULL dims. At TP1 both agree; the pin is the
    module attribute contract (full dims into the linear factory)."""
    from v10_model import build_v10_model

    model, cfg = _tiny_mtp_model(cpu_env)
    inner = dict(model.named_modules())[
        "mtp.layers.0.mtp_model_layer.self_attention"]
    assert inner.num_heads * inner.qk_nope == \
        inner.q_b_proj.weight.shape[0] * cfg.tensor_model_parallel_size
    assert inner.num_heads * (inner.qk_nope + inner.v_head_dim) == \
        inner.kv_b_proj.weight.shape[0] * cfg.tensor_model_parallel_size


def test_swap_eh_halves():
    """mcore cats [e | h]; the GLM checkpoint eh_proj is [h | e] — the swap
    restores mcore's input order. Pure function pin (fill integration lands
    with the MTP fill rules)."""
    from v10_mtp_fill import swap_eh_halves
    h, e = torch.arange(8).float().reshape(1, 8), torch.arange(8, 16).float().reshape(1, 8)
    w = torch.cat([h, e], dim=1)              # GLM order [h | e] -> [1, 16]
    swapped = swap_eh_halves(w)               # -> [e | h]
    assert torch.equal(swapped[:, :8], e)
    assert torch.equal(swapped[:, 8:], h)
    # involution
    assert torch.equal(swap_eh_halves(swapped), w)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
