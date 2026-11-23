#!/usr/bin/env python
"""V10 unit math tests — golden-reference comparisons (no mcore build).

Run on .38: /root/venv_v7/bin/python -m pytest scripts/v10/tests/test_v10_units.py -q
(v10_modules/v10_kda import megatron at module level; the venv has it.)

Goldens = tests/hf_reference.py (vendored line-by-line HF port from the
parity-validated reference project, tiny/medium parity rel<=1.7e-06).
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

from hf_reference import (hf_hyper_connection_gates, hf_recurrent_kda,  # noqa: E402
                          hf_rmsnorm)
from v10_kda import kda_causal_conv1d, kda_chunk_forward  # noqa: E402
from v10_modules import (GlmHyperConnectionModule, GlmRMSNorm,  # noqa: E402
                         RMSNormGated)

torch.manual_seed(7)


# ---------------------------------------------------------------------------
# KDA
# ---------------------------------------------------------------------------
def test_kda_chunk_matches_recurrent_golden():
    """chunk (Householder) vs naive recurrent delta rule — fp32 tol 1e-4.
    S=70 (non-multiple of 64) exercises the padding path."""
    B, S, H, Dk, Dv = 2, 70, 2, 8, 8
    q = torch.randn(B, S, H, Dk)
    k = torch.randn(B, S, H, Dk)
    v = torch.randn(B, S, H, Dv)
    g = -torch.rand(B, S, H, Dk) * 5.0        # ∈ (-5, 0)
    beta = torch.rand(B, S, H)

    out_chunk = kda_chunk_forward(q, k, v, g, beta, chunk_size=64)
    out_gold, _ = hf_recurrent_kda(q, k, v, g, beta)
    rel = (out_chunk.float() - out_gold.float()).abs().max() / \
        out_gold.float().abs().max().clamp_min(1e-6)
    assert rel < 1e-4, f"chunk vs recurrent rel={rel:.3e}"


def test_kda_chunk_exact_multiple_of_chunk():
    B, S, H, D = 1, 128, 1, 4
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    g = -torch.rand(B, S, H, D) * 5.0
    beta = torch.rand(B, S, H)
    out_chunk = kda_chunk_forward(q, k, v, g, beta, chunk_size=64)
    out_gold, _ = hf_recurrent_kda(q, k, v, g, beta)
    rel = (out_chunk.float() - out_gold.float()).abs().max() / \
        out_gold.float().abs().max().clamp_min(1e-6)
    assert rel < 1e-4


def test_kda_causal_conv1d_causality():
    """Conv output at t must depend only on x[<=t]: perturbing the future
    must leave the past untouched."""
    B, C, S, K = 1, 6, 12, 4
    x = torch.randn(B, C, S)
    w = torch.randn(C, 1, K)
    out1 = kda_causal_conv1d(x, w, S, K, activation=None)
    x2 = x.clone()
    x2[..., 8:] += 10.0
    out2 = kda_causal_conv1d(x2, w, S, K, activation=None)
    assert torch.equal(out1[..., :8], out2[..., :8]), "causality violated"
    assert not torch.equal(out1[..., 8:], out2[..., 8:])


def test_kda_conv_left_pad_not_circular():
    """Left zero padding (strict causal), not circular wrap: t=0 output must
    see only x[0] and zeros."""
    B, C, S, K = 1, 1, 4, 4
    x = torch.zeros(B, C, S)
    x[..., 3] = 5.0                      # spike at the END
    w = torch.ones(C, 1, K)
    out = kda_causal_conv1d(x, w, S, K, activation=None)
    assert out[..., 0].abs().max() == 0, "t=0 must not see future tokens"


# ---------------------------------------------------------------------------
# mHC
# ---------------------------------------------------------------------------
def _make_hc(n=4, h=8, seed=3):
    torch.manual_seed(seed)
    hc = GlmHyperConnectionModule(n_streams=n, hidden=h)
    with torch.no_grad():
        hc.fn.normal_(0, 0.02)
        hc.base.normal_(0, 0.1)
    return hc


def test_hc_gates_match_golden():
    hc = _make_hc()
    b, s, n, h = 2, 5, hc.n, hc.hidden
    streams = torch.randn(b, s, n, h)
    pre, post, comb = hc.compute_gates(streams)
    pre_g, post_g, comb_g = hf_hyper_connection_gates(
        streams, hc.fn, hc.base, hc.scale, n, hc.eps, hc.iters)
    assert torch.allclose(pre, pre_g, atol=1e-6), "pre mismatch"
    assert torch.allclose(post, post_g, atol=1e-6), "post mismatch"
    assert torch.allclose(comb, comb_g, atol=1e-6), "comb mismatch"


def test_hc_sinkhorn_double_stochastic():
    hc = _make_hc()
    streams = torch.randn(1, 3, hc.n, hc.hidden)
    _, _, comb = hc.compute_gates(streams)
    assert torch.allclose(comb.sum(dim=-1), torch.ones_like(comb.sum(dim=-1)),
                          atol=1e-3), f"row sum off: {comb.sum(-1)[0,0]}"
    assert torch.allclose(comb.sum(dim=-2), torch.ones_like(comb.sum(dim=-2)),
                          atol=1e-3), f"col sum off: {comb.sum(-2)[0,0]}"


def test_hc_gate_ranges():
    hc = _make_hc()
    streams = torch.randn(1, 3, hc.n, hc.hidden)
    pre, post, _ = hc.compute_gates(streams)
    assert (pre > 0).all() and (pre < 1 + hc.eps + 1e-6).all()
    assert (post > 0).all() and (post < 2 + 1e-6).all()


def test_hc_expand_contract():
    hc = GlmHyperConnectionModule(n_streams=4, hidden=6)
    x = torch.randn(2, 3, 6)
    ex = GlmHyperConnectionModule.expand_streams(x, 4)
    assert ex.shape == (2, 3, 4, 6 * 4) or ex.shape == (2, 3, 24)
    streams = torch.randn(2, 3, 4, 6)
    c = GlmHyperConnectionModule.mean_contract(streams)
    assert c.shape == (2, 3, 6)
    assert torch.allclose(c, streams.mean(dim=2))


# ---------------------------------------------------------------------------
# norms
# ---------------------------------------------------------------------------
def test_glm_rmsnorm_matches_golden():
    torch.manual_seed(1)
    norm = GlmRMSNorm(hidden_size=16, eps=1e-5)
    with torch.no_grad():
        norm.weight.normal_(1.0, 0.1)
    x = torch.randn(4, 5, 16, dtype=torch.bfloat16)
    out = norm(x)
    gold = hf_rmsnorm(x, norm.weight, 1e-5)
    assert torch.allclose(out.float(), gold.float(), atol=1e-3)


def test_rmsnorm_gated_manual():
    norm = RMSNormGated(8, eps=1e-5)
    x = torch.randn(2, 3, 8)
    gate = torch.randn(2, 3, 8)
    out = norm(x, gate)
    xf = x.float()
    manual = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-5)) \
        * norm.weight.float() * torch.sigmoid(gate.float())
    assert torch.allclose(out.float(), manual, atol=1e-5)


# ---------------------------------------------------------------------------
# fill composite per-rank interleave (mcore 0.18.2 glu: chunk on TP-LOCAL out)
# ---------------------------------------------------------------------------
def test_composite_fuse_fc1_per_rank_interleave():
    """mcore 0.18.2 glu() chunks the TP-LOCAL output (mlp.py torch.chunk(x, 2,
    dim=-1)) — rank r's gated fc1 must be [gate[r*f/tp:(r+1)*f/tp]; up[...]]."""
    from v10_fill import composite_fuse
    torch.manual_seed(5)
    f, h = 16, 8
    gate, up = torch.randn(f, h), torch.randn(f, h)
    tp = 2
    for r in range(tp):
        fused = composite_fuse([gate, up], "column", r, tp)
        half = f // tp
        expect = torch.cat([gate[r * half:(r + 1) * half],
                            up[r * half:(r + 1) * half]], dim=0)
        assert torch.equal(fused, expect), f"fc1 rank{r} must be shard-then-cat"
        assert fused.shape == (f * 2 // tp, h)
    assert torch.equal(composite_fuse([gate, up], "column", 0, 1),
                       torch.cat([gate, up], dim=0)), "TP=1 must stay plain cat"


def test_composite_fuse_conv1d_per_rank_interleave():
    """The fused conv consumes cat([q,k,v]) of three INDEPENDENT ColumnParallel
    outputs — rank r's conv rows must be q/k/v slices of the SAME rank range."""
    from v10_fill import composite_fuse
    torch.manual_seed(6)
    c, k = 8, 4
    qc, kc, vc_ = torch.randn(c, 1, k), torch.randn(c, 1, k), torch.randn(c, 1, k)
    tp = 4
    for r in range(tp):
        fused = composite_fuse([qc, kc, vc_], "column", r, tp)
        per = c // tp
        expect = torch.cat([qc[r * per:(r + 1) * per],
                            kc[r * per:(r + 1) * per],
                            vc_[r * per:(r + 1) * per]], dim=0)
        assert torch.equal(fused, expect), f"conv rank{r} must be shard-then-cat"
        assert fused.shape == (c * 3 // tp, 1, k)


def test_composite_fuse_cat_then_shard_is_wrong():
    """The regression this guards: cat-then-shard hands rank 0 a PURE-gate
    (or pure-q) block — must differ from the correct per-rank interleave."""
    from v10_fill import composite_fuse
    gate, up = torch.ones(8, 4), -torch.ones(8, 4)
    fused = composite_fuse([gate, up], "column", 0, 2)
    assert torch.equal(fused, torch.cat([gate[:4], up[:4]], dim=0))
    assert not torch.equal(fused, torch.cat([gate, up], dim=0)[:8]), \
        "cat-then-shard regression detected"


def test_sliced_checkpoint_equals_full_then_shard_slice(tmp_path):
    """SlicedCheckpoint.slice_tensor must be BIT-IDENTICAL to
    shard_slice(full_tensor) for every mode — the byte-range IO path is
    behavior-transparent (fill math untouched)."""
    import json as _json

    from safetensors.torch import save_file

    from v10_fill import SlicedCheckpoint, shard_slice
    torch.manual_seed(12)
    tensors = {"a_col": torch.randn(16, 8).to(torch.bfloat16),
               "b_row": torch.randn(8, 12),
               "c_rep": torch.randn(6),
               "d_conv": torch.randn(12, 1, 3).to(torch.bfloat16),
               "e_1d": torch.randn(8, dtype=torch.float32)}
    save_file(tensors, str(tmp_path / "shard.safetensors"))
    wm = {k: "shard.safetensors" for k in tensors}
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        _json.dump({"weight_map": wm}, f)
    sc = SlicedCheckpoint(str(tmp_path), wm)
    for key, mode, tp in (("a_col", "column", 4), ("b_row", "row", 2),
                          ("c_rep", "replicated", 4), ("d_conv", "column", 3),
                          ("e_1d", "column", 4)):
        full = tensors[key]
        for r in range(tp):
            got = sc.slice_tensor(key, mode, r, tp)
            want = shard_slice(full, mode, r, tp)
            assert torch.equal(got, want), f"{key} {mode} rank{r}/{tp}"
        assert torch.equal(sc.full(key), full)
        assert torch.equal(sc.load(key, mode, tp - 1, tp),
                           shard_slice(full, mode, tp - 1, tp))
    # TP=1 dispatch: everything is the full tensor
    for key, mode in (("a_col", "column"), ("b_row", "row"), ("c_rep", "replicated")):
        assert torch.equal(sc.load(key, mode, 0, 1), tensors[key])


# ---------------------------------------------------------------------------
# causal_ends_from_mask (right-aligned-causal contract)
# ---------------------------------------------------------------------------
def test_causal_ends_from_mask_tril():
    from v10_modules import causal_ends_from_mask
    S, B = 10, 3
    tril = torch.ones(B, 1, S, S, dtype=torch.bool).tril()
    ends = causal_ends_from_mask(tril, S)
    assert torch.equal(ends, torch.arange(1, S + 1).unsqueeze(0).expand(B, -1))


def test_causal_ends_from_mask_right_pad_prefix_ok():
    """Right-aligned padding-aware masks (pad KEYS excluded for every query)
    are contiguous prefixes — accepted; pad rows get the content length."""
    from v10_modules import causal_ends_from_mask
    S, B, L = 10, 2, 6
    qpos = torch.arange(S).view(1, 1, S, 1)
    kpos = torch.arange(S).view(1, 1, 1, S)
    m = ((kpos < L) & (kpos <= qpos)).expand(B, 1, S, S)   # content [0,L)
    ends = causal_ends_from_mask(m, S)
    assert torch.equal(ends[:, :L], torch.arange(1, L + 1).unsqueeze(0).expand(B, -1))
    assert (ends[:, L:] == L).all()


def test_causal_ends_from_mask_left_pad_raises():
    from v10_modules import causal_ends_from_mask
    S, L = 10, 4
    start = S - L                                  # pads at [0, start)
    qpos = torch.arange(S).view(1, 1, S, 1)
    kpos = torch.arange(S).view(1, 1, 1, S)
    m = (qpos >= start) & (kpos >= start) & (kpos <= qpos)
    with pytest.raises(ValueError):
        causal_ends_from_mask(m, S)


def test_causal_ends_from_mask_rejects_bad_input():
    from v10_modules import causal_ends_from_mask
    with pytest.raises(ValueError):
        causal_ends_from_mask(torch.ones(1, 1, 4, 4), 4)      # float mask
    with pytest.raises(ValueError):
        causal_ends_from_mask(torch.ones(2, 4, 4, dtype=torch.bool), 5)  # shape


# ---------------------------------------------------------------------------
# fused topk with per-batch causal_ends
# ---------------------------------------------------------------------------
def _topk_inputs(sq=16, B=2, heads=2, dim=4, pool=2, topk=8, seed=9):
    torch.manual_seed(seed)
    return (torch.randn(sq, B, heads, dim), torch.randn(sq, B, dim),
            torch.rand(sq, B, heads), torch.randn(sq, B, dim),
            torch.zeros(pool, dim))


def test_fused_topk_1d_vs_batch_ends_equivalence():
    from v10_dsa import fused_qk_topk_kpool_causal
    q, k, w, gate, ape = _topk_inputs(B=1)
    ends1d = torch.arange(1, 17)
    a = fused_qk_topk_kpool_causal(q, k, w, 8, 2, gate, ape, ends1d)
    b = fused_qk_topk_kpool_causal(q, k, w, 8, 2, gate, ape, ends1d.unsqueeze(0))
    assert torch.equal(a, b), "[B,sq] ends must equal the shared [sq] form"


def test_fused_topk_per_batch_bound_respected():
    """Batch item 1 truncated at L (right pad): every selection of its pad
    rows must stay < L, and item 0's rows must be unaffected."""
    from v10_dsa import fused_qk_topk_kpool_causal
    q, k, w, gate, ape = _topk_inputs()
    sq, L = 16, 9
    ends1d = torch.arange(1, sq + 1)
    row1 = torch.where(torch.arange(sq) < L, ends1d, torch.full((sq,), L))
    ends2 = torch.stack([ends1d, row1])
    out = fused_qk_topk_kpool_causal(q, k, w, 8, 2, gate, ape, ends2)
    assert out.shape[0] == 2
    sel = out[1, L:][out[1, L:] >= 0]
    assert (sel.long() < L).all(), "pad-row topk crossed the valid bound"
    full = fused_qk_topk_kpool_causal(q[:, :1], k[:, :1], w[:, :1], 8, 2,
                                      gate[:, :1], ape, ends1d)
    assert torch.equal(out[0], full[0]), "item 0 rows must be mask-independent"


def test_fused_topk_single_row_ends_broadcast():
    """A [1, sq] ends tensor (mask [1,1,S,S] convention) must broadcast over a
    wider hidden batch, bit-exact vs the explicitly expanded form."""
    from v10_dsa import fused_qk_topk_kpool_causal
    q, k, w, gate, ape = _topk_inputs()
    ends1 = torch.arange(1, 17).unsqueeze(0)              # [1, sq]
    out = fused_qk_topk_kpool_causal(q, k, w, 8, 2, gate, ape, ends1)
    ref = fused_qk_topk_kpool_causal(q, k, w, 8, 2, gate, ape,
                                     ends1.expand(2, -1).contiguous())
    assert torch.equal(out, ref)
    with pytest.raises(ValueError):
        fused_qk_topk_kpool_causal(q, k, w, 8, 2, gate, ape,
                                   torch.arange(1, 17).unsqueeze(0).repeat(3, 1))


# ---------------------------------------------------------------------------
# config mapping (V8 iron rules §2: SiLU / clamp / noaux_tc / NoPE / eps)
# ---------------------------------------------------------------------------
def test_config_layer_kind_mapping():
    from v10_config import attn_kind_per_layer, ffn_kind_per_layer
    vc = dict(layer_types=["linear_attention", "deepseek_sparse_attention",
                           "full_attention", "linear_attention"],
              num_hidden_layers=4,
              mlp_layer_types=["dense", "sparse", "sparse", "dense"],
              first_k_dense_replace=1)
    assert attn_kind_per_layer(vc) == ["kda", "dsa", "dsa", "kda"]
    assert ffn_kind_per_layer(vc) == ["dense", "moe", "moe", "dense"]
    with pytest.raises(ValueError):
        attn_kind_per_layer(dict(vc, layer_types=["gdn"], num_hidden_layers=1))


def test_config_moe_and_clamp_kwargs():
    from test_v10_build import TINY_VC
    from v10_config import v10_config_kwargs
    kw = v10_config_kwargs(dict(TINY_VC))
    assert kw["activation_func"] is torch.nn.functional.silu          # V8 rule 3
    assert kw["activation_func_clamp_value"] == TINY_VC["swiglu_limit"]
    assert kw["moe_router_score_function"] == "sigmoid"
    assert kw["moe_router_load_balancing_type"] == "none"
    assert kw["moe_router_enable_expert_bias"] is True
    assert kw["moe_router_bias_update_rate"] == 0.0
    assert kw["moe_router_topk_scaling_factor"] == TINY_VC["routed_scaling_factor"]
    assert kw["moe_router_dtype"] == "fp32"
    assert kw["qk_pos_emb_head_dim"] == 0                             # NoPE
    assert kw["layernorm_epsilon"] == TINY_VC["rms_norm_eps"]
    assert kw["moe_layer_freq"] == [0, 1]


def test_config_glm_attrs_installed():
    from test_v10_build import TINY_VC
    from v10_config import install_glm_attrs, v10_config_kwargs
    from v10_model import v10_transformer_config
    cfg = v10_transformer_config(dict(TINY_VC))
    assert cfg.glm_linear_num_heads == 2
    assert cfg.glm_linear_head_dim == 8
    assert cfg.glm_linear_conv_kernel_dim == 4
    assert cfg.glm_gate_lower_bound == -5.0
    assert cfg.glm_mhc_streams == TINY_VC["hc_mult"]
    assert cfg.glm_hc_eps == TINY_VC["hc_eps"]
    assert cfg.glm_mhc_sinkhorn_iters == TINY_VC["hc_sinkhorn_iters"]
    assert cfg.dsa_indexer_topk == 8
    assert cfg.dsa_indexer_n_heads == 2
    assert cfg.dsa_indexer_head_dim == 4
    assert cfg.dsa_indexer_kpool == 2
    assert cfg.dsa_indexer_always_select_tail is True


# ---------------------------------------------------------------------------
# api_ce_gate.extract_prompt_ce (NameError regression + both plp forms)
# ---------------------------------------------------------------------------
class _LP:
    def __init__(self, lp):
        self.logprob = lp


def test_extract_prompt_ce_http_json_form():
    import api_ce_gate
    ids = [7, 11, 13, 17]
    plp = [None,
           {str(11): {"logprob": -1.0}},
           {str(13): {"logprob": -2.0}},
           {str(17): {"logprob": -3.5}}]
    assert abs(api_ce_gate.extract_prompt_ce(plp, ids) - 6.5 / 3) < 1e-9


def test_extract_prompt_ce_offline_object_form():
    import api_ce_gate
    ids = [7, 11, 13, 17]
    plp = [None, {11: _LP(-1.0)}, {13: _LP(-2.0)}, {17: _LP(-3.5)}]
    assert abs(api_ce_gate.extract_prompt_ce(plp, ids) - 6.5 / 3) < 1e-9


def test_extract_prompt_ce_skips_missing_positions():
    import api_ce_gate
    ids = [7, 11, 13, 17]
    plp = [None, None, {str(13): {"logprob": -2.0}}, {}]
    assert api_ce_gate.extract_prompt_ce(plp, ids) == 2.0


def test_extract_prompt_ce_empty_raises():
    """All-None positions must raise, not silently read CE=0.0 (= fake PASS)."""
    import api_ce_gate
    with pytest.raises(ValueError):
        api_ce_gate.extract_prompt_ce([None, None], [7, 11])
    with pytest.raises(ValueError):
        api_ce_gate.extract_prompt_ce([None, {"999": {"logprob": -1.0}}], [7, 11])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def test_kda_chunk_backward_finite():
    """THE nan root-cause regression (anomaly-traced 2026-11-20): the decay
    mask's triu exponent exp(+) = inf — forward masked (finite loss) but the
    mul backward computed 0*inf = NaN into every LoRA grad. The clamp(max=0)
    fix must keep ALL grads finite."""
    from v10_kda import kda_chunk_forward
    B, S, H, D = 1, 70, 2, 8
    torch.manual_seed(7)
    q = torch.randn(B, S, H, D, requires_grad=True)
    k = torch.randn(B, S, H, D, requires_grad=True)
    v = torch.randn(B, S, H, D, requires_grad=True)
    g = (-torch.rand(B, S, H, D) * 5.0).requires_grad_(True)
    beta = torch.rand(B, S, H, requires_grad=True)
    out = kda_chunk_forward(q, k, v, g, beta, chunk_size=64)
    assert torch.isfinite(out.float()).all()
    out.float().pow(2).sum().backward()
    for name, t in (("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)):
        assert t.grad is not None, f"no grad for {name}"
        assert torch.isfinite(t.grad).all(), f"NON-FINITE grad for {name}"
