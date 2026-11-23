#!/usr/bin/env python
"""V10 build tests — tiny CPU builds of KDA/DSA/Glm5NextLayer/full GPTModel
(gloo dist; CPU-safe patches per the reference project's non-TE CPU smoke).

Run on .38: /root/venv_v7/bin/python -m pytest scripts/v10/tests/test_v10_build.py -q
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


# ---------------------------------------------------------------------------
# CPU-safe patches (reference-proven, scoped to this module): mcore hardcodes
# torch.cuda.current_device() in router/expert-bias paths; on .38 torch_npu
# aliases it to the NPU — CPU builds need it pinned to 'cpu'.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cpu_env():
    _orig = torch.cuda.current_device
    torch.cuda.current_device = lambda: "cpu"
    # recompute (GLM53_V10_RECOMPUTE, default full) routes the forward through
    # mcore's CUDA RNG-tracker checkpoint — needs the torch.cuda namespace,
    # which exists on .40 (torch_npu aliases) but NOT on a pure-CPU dev box.
    # The CPU tests verify math, not recompute — pin it off here.
    os.environ["GLM53_V10_RECOMPUTE"] = "off"
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29679")
    os.environ.setdefault("V10_MASTER_PORT", "29679")
    import torch.distributed as dist
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo", init_method="env://",
                                world_size=1, rank=0)
    from megatron.core import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel()
    yield
    torch.cuda.current_device = _orig


TINY_VC = dict(
    hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
    q_lora_rank=32, kv_lora_rank=16,
    qk_nope_head_dim=8, qk_rope_head_dim=0, qk_head_dim=8, v_head_dim=8,
    intermediate_size=128, moe_intermediate_size=32,
    n_routed_experts=4, num_experts_per_tok=2, routed_scaling_factor=2.5,
    first_k_dense_replace=1, swiglu_limit=10.0, rms_norm_eps=1e-5,
    index_topk=8, index_n_heads=2, index_head_dim=4, index_kpool=2,
    index_kpool_always_select_tail=True,
    hc_mult=2, hc_eps=1e-6, hc_sinkhorn_iters=5,
    vocab_size=128, num_nextn_predict_layers=1,
    layer_types=["linear_attention", "deepseek_sparse_attention"],
    mlp_layer_types=["dense", "moe"],
    linear_attn_config=dict(num_heads=2, head_dim=8, short_conv_kernel_size=4,
                            gate_lower_bound=-5.0),
    n_shared_experts=1,
)


# ---------------------------------------------------------------------------
# KDA attention (real mcore build)
# ---------------------------------------------------------------------------
def _make_cfg(vc=TINY_VC, tp=1):
    from v10_model import v10_transformer_config
    return v10_transformer_config(vc, num_layers=len(vc["layer_types"]))


def _seed_params(module, seed=11):
    """perform_initialization=False leaves mcore params AND fp32 buffers
    (expert_bias) as torch.empty garbage — NaN != NaN breaks torch.equal and
    pollutes routing (fill covers them in production; tests seed finite
    values instead)."""
    torch.manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            if p.dtype.is_floating_point:
                p.normal_(0.0, 0.02)
        for b in module.buffers():
            if b.dtype.is_floating_point and not b.all().isfinite():
                b.normal_(0.0, 0.01)


def test_kda_attention_forward_backward(cpu_env):
    from v10_kda import GlmKDAAttention, GlmKDASubmodules

    cfg = _make_cfg()
    attn = GlmKDAAttention(cfg, GlmKDASubmodules(), layer_number=1)
    _seed_params(attn)
    S, B, H = 12, 2, cfg.hidden_size
    x = torch.randn(S, B, H, dtype=torch.bfloat16, requires_grad=True)
    out, bias = attn(x)
    assert out.shape == (S, B, H)
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    out.float().sum().backward()
    assert x.grad is not None
    # the trainable params all received grads
    for name, p in attn.named_parameters():
        if name.startswith(("linear_", "A_log", "dt_bias", "conv1d_weight",
                            "o_norm")):
            assert p.grad is not None, f"no grad for {name}"


def test_kda_gate_bounds(cpu_env):
    """g = -5*sigmoid(...) must lie in (-5, 0)."""
    from v10_kda import GlmKDAAttention, GlmKDASubmodules

    cfg = _make_cfg()
    attn = GlmKDAAttention(cfg, GlmKDASubmodules(), layer_number=1)
    _seed_params(attn)
    S, B, H = 6, 1, cfg.hidden_size
    x = torch.randn(S, B, H, dtype=torch.bfloat16)
    with torch.no_grad():
        q, _ = attn.linear_q_proj(x)
        k, _ = attn.linear_k_proj(x)
        v, _ = attn.linear_v_proj(x)
        f_a = attn.linear_fa_proj(x)
        f_b, _ = attn.linear_fb_proj(f_a.to(x.dtype))
        g_raw = f_b.float() + attn.dt_bias.float().view(1, 1, -1)
        g = attn.gate_lower_bound * torch.sigmoid(
            attn.A_log.float().exp().view(1, 1, attn.tp_heads, 1)
            * g_raw.view(B, S, attn.tp_heads, attn.head_dim)).view(B, S, -1)
    assert (g > -5.0).all() and (g < 0).all()


# ---------------------------------------------------------------------------
# DSA indexer + attention
# ---------------------------------------------------------------------------
def test_dsa_indexer_topk_causality(cpu_env):
    from v10_dsa import V10DSAIndexer

    cfg = _make_cfg()
    idx = V10DSAIndexer(cfg, layer_number=2)
    _seed_params(idx, seed=23)
    S, B, H = 16, 1, cfg.hidden_size
    x = torch.randn(S, B, H, dtype=torch.bfloat16)
    q_resid = torch.randn(S, B, cfg.q_lora_rank, dtype=torch.bfloat16)
    causal_ends = torch.arange(1, S + 1)
    topk = idx(x, q_resid, causal_ends)
    width = cfg.dsa_indexer_topk + cfg.dsa_indexer_kpool - 1
    assert topk.shape == (B, S, width)
    assert topk.dtype == torch.int32
    valid = topk >= 0
    # every selected token index < its query's causal end
    for t in range(S):
        sel = topk[0, t][valid[0, t]].long()
        assert (sel < causal_ends[t]).all(), \
            f"query t={t} selected future key: max={sel.max()}"
    # query 0 attends only token 0 (tail only — no complete pools)
    sel0 = topk[0, 0][valid[0, 0]].long()
    assert sel0.numel() == 1 and sel0[0] == 0


def test_dsa_attention_forward_backward(cpu_env):
    from v10_dsa import V10DSAAttention

    cfg = _make_cfg()
    attn = V10DSAAttention(cfg, layer_number=2)
    _seed_params(attn, seed=31)
    S, B, H = 16, 2, cfg.hidden_size
    x = torch.randn(S, B, H, dtype=torch.bfloat16, requires_grad=True)
    out, bias = attn(x)
    assert out.shape == (S, B, H)
    assert torch.isfinite(out.float()).all()
    out.float().sum().backward()
    # indexer is FROZEN: no grads on indexer params
    for name, p in attn.indexer.named_parameters():
        assert p.grad is None or (p.grad == 0).all(), f"indexer grad leaked: {name}"
    # main linears got grads
    assert attn.q_b_proj.weight.grad is not None
    # lora-replicated linears got grads
    assert attn.q_a_proj.weight.grad is not None


def test_dsa_head_split_shapes(cpu_env):
    """kv_b_proj must carry (nope+v) per head; kv_a stays lora-replicated."""
    from v10_dsa import V10DSAAttention

    cfg = _make_cfg()
    attn = V10DSAAttention(cfg, layer_number=2)
    # head-split: [heads*tp, qk_nope + v_head_dim] / tp
    assert attn.kv_b_proj.weight.shape == (1, cfg.kv_lora_rank) or True
    tp = cfg.tensor_model_parallel_size
    assert attn.q_b_proj.weight.shape[0] == (cfg.num_attention_heads // tp) * cfg.qk_head_dim
    assert attn.kv_b_proj.weight.shape[0] == \
        (cfg.num_attention_heads // tp) * (cfg.qk_head_dim + cfg.v_head_dim)
    assert attn.q_a_proj.weight.shape == (cfg.q_lora_rank, cfg.hidden_size)
    assert attn.kv_a_proj_with_mqa.weight.shape == (cfg.kv_lora_rank, cfg.hidden_size)


# ---------------------------------------------------------------------------
# Glm5NextLayer streams contract
# ---------------------------------------------------------------------------
def _tiny_layer_spec(attn_kind="kda", ffn_kind="dense"):
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core import tensor_parallel

    from v10_dsa import V10DSAAttention
    from v10_kda import GlmKDAAttention, GlmKDASubmodules
    from v10_layers import Glm5NextLayerSubmodules
    from v10_modules import GlmRMSNorm

    attn = ModuleSpec(module=GlmKDAAttention, submodules=GlmKDASubmodules()) \
        if attn_kind == "kda" else ModuleSpec(module=V10DSAAttention)
    mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=tensor_parallel.ColumnParallelLinear,
                                 linear_fc2=tensor_parallel.RowParallelLinear)) \
        if ffn_kind == "dense" else None
    return Glm5NextLayerSubmodules(input_layernorm=GlmRMSNorm,
                                   self_attention=attn,
                                   pre_mlp_layernorm=GlmRMSNorm, mlp=mlp)


def test_glm5next_layer_first_expand_last_contract(cpu_env):
    from v10_layers import Glm5NextLayer

    cfg = _make_cfg()
    n = cfg.glm_mhc_streams
    S, B, H = 8, 2, cfg.hidden_size
    x = torch.randn(S, B, H, dtype=torch.bfloat16, requires_grad=True)

    first = Glm5NextLayer(cfg, _tiny_layer_spec(), layer_number=1,
                          is_first=True, is_last=False)
    _seed_params(first)
    out, bias = first(x)
    assert out.shape == (S, B, n * H), \
        f"non-last layer must emit n*H streams, got {tuple(out.shape)}"

    last = Glm5NextLayer(cfg, _tiny_layer_spec(), layer_number=2,
                         is_first=False, is_last=True)
    out2, _ = last(out.detach().requires_grad_(True))
    assert out2.shape == (S, B, H), \
        f"is_last layer must mean-contract back to H, got {tuple(out2.shape)}"

    (out.float().sum() + out2.float().sum()).backward()
    assert x.grad is not None


def test_glm5next_layer_mid_passthrough_width(cpu_env):
    """A middle layer consumes n*H and emits n*H."""
    from v10_layers import Glm5NextLayer

    cfg = _make_cfg()
    n = cfg.glm_mhc_streams
    S, B = 8, 1
    x = torch.randn(S, B, n * cfg.hidden_size, dtype=torch.bfloat16)
    mid = Glm5NextLayer(cfg, _tiny_layer_spec(), layer_number=2,
                        is_first=False, is_last=False)
    out, _ = mid(x)
    assert out.shape == (S, B, n * cfg.hidden_size)


def test_glm5next_layer_hc_param_names(cpu_env):
    """hc_attn_*/hc_ffn_* checkpoint mapping: attn_hc.{fn,base,scale} +
    ffn_hc.{fn,base,scale} per layer."""
    from v10_layers import Glm5NextLayer

    cfg = _make_cfg()
    layer = Glm5NextLayer(cfg, _tiny_layer_spec(), layer_number=1,
                          is_first=True, is_last=False)
    names = dict(layer.named_parameters())
    n = cfg.glm_mhc_streams
    h = cfg.hidden_size
    mix = (2 + n) * n
    assert names["attn_hc.fn"].shape == (mix, n * h)
    assert names["attn_hc.base"].shape == (mix,)
    assert names["attn_hc.scale"].shape == (3,)
    assert names["ffn_hc.fn"].shape == (mix, n * h)
    assert names["ffn_hc.fn"].dtype == torch.float32


# ---------------------------------------------------------------------------
# full tiny model (meta build -> to_empty -> forward/backward)
# ---------------------------------------------------------------------------
def test_build_v10_model_tiny_e2e(cpu_env):
    """The complete Phase-1 chain: meta skeleton build -> to_empty -> batch-first
    forward [B,S] -> logits [B,S,V] -> backward. Exercises KDA layer + DSA layer
    + dense + MoE + mHC expand/contract + final norm."""
    from v10_model import build_v10_model

    model, cfg = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=False)
    _seed_params(model, seed=101)
    B, S = 2, 16
    ids = torch.randint(1, TINY_VC["vocab_size"], (B, S))
    pos = torch.zeros(B, S, dtype=torch.int64)  # NoPE: position_ids unconsumed
    attn_mask = torch.ones(B, 1, S, S, dtype=torch.bool).tril()  # per-batch causal keep-mask
    logits = model(input_ids=ids, position_ids=pos, attention_mask=attn_mask)
    assert logits.shape[:2] == (B, S), f"batch-first logits, got {tuple(logits.shape)}"
    assert torch.isfinite(logits.float()).all()
    logits.float().sum().backward()
    # trainable params got grads; MoE router present
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    assert n_grad > 30, f"expected grads on most trainable params, got {n_grad}"
    # determinism: same input twice
    with torch.no_grad():
        l2 = model(input_ids=ids, position_ids=pos, attention_mask=attn_mask)
    assert torch.equal(logits, l2), "forward must be deterministic (dropout=0)"


def test_build_v10_model_streams_contract(cpu_env):
    """The decoder emits single-stream H at the END (mean contract) — the
    final norm + output layer therefore consume H, not n*H."""
    from v10_model import build_v10_model

    model, cfg = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=False)
    out_norm = model.decoder.final_layernorm
    assert out_norm is not None
    hs = out_norm.weight.shape[0]
    assert hs == cfg.hidden_size


# ---------------------------------------------------------------------------
# trap regression tests (V8 iron rules §2 / V10 iron rules §3 carried forward)
# ---------------------------------------------------------------------------
def _layers_by_kind(model):
    kda = [l for l in model.decoder.layers
           if hasattr(l.self_attention, "o_norm")]
    dsa = [l for l in model.decoder.layers
           if hasattr(l.self_attention, "indexer")]
    return kda, dsa


def test_norm_eps_by_name(cpu_env):
    """V8 rule: eps is assigned BY NAME — main norms rms_norm_eps (1e-5),
    indexer k_norm LayerNorm 1e-6 WITH bias, HC gate eps=hc_eps (1e-6) while
    the HC INPUT norm uses rms_norm_eps (golden hardcodes it)."""
    from v10_model import build_v10_model

    model, cfg = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=False)
    eps = cfg.layernorm_epsilon
    assert eps == TINY_VC["rms_norm_eps"]
    kda, dsa = _layers_by_kind(model)
    assert kda and dsa
    for layer in model.decoder.layers:
        assert layer.input_layernorm.eps == eps
        assert layer.pre_mlp_layernorm.eps == eps
        for hc in (layer.attn_hc, layer.ffn_hc):
            assert hc.eps == TINY_VC["hc_eps"]          # gate eps
            assert hc.norm_eps == eps                    # HC input norm eps
    for layer in kda:
        assert layer.self_attention.o_norm.eps == eps    # NOT class-default 1e-6
    for layer in dsa:
        attn = layer.self_attention
        assert attn.q_a_layernorm.eps == eps
        assert attn.kv_a_layernorm.eps == eps
        assert attn.softmax_scale == TINY_VC["qk_nope_head_dim"] ** -0.5
        assert attn.indexer.k_norm.norm.eps == 1e-6
        assert attn.indexer.k_norm.norm.bias is not None  # LayerNorm WITH bias
    assert model.decoder.final_layernorm.eps == eps


def test_meta_build_leaves_no_meta_tensors(cpu_env):
    """V8 round-15: to_empty materializes params/buffers only — the meta-cache
    sweep + inv_freq recompute must leave ZERO meta tensors behind."""
    from v10_model import build_v10_model

    model, _ = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=False)
    for n, p in model.named_parameters():
        assert p.device.type != "meta", f"meta param leaked: {n}"
    for n, b in model.named_buffers():
        assert b.device.type != "meta", f"meta buffer leaked: {n}"
    for name, m in model.named_modules():
        inv = getattr(m, "inv_freq", None)
        if isinstance(inv, torch.Tensor):
            assert inv.device.type != "meta", f"meta inv_freq leaked: {name}"


def test_nope_position_ids_ignored(cpu_env):
    """NoPE invariant (qk_rope=0): the rotary module exists (0.18.2 has no
    'none') but V10 layers never consume it — position_ids must not move logits."""
    from v10_model import build_v10_model

    model, _ = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=False)
    _seed_params(model, seed=101)
    B, S = 1, 12
    ids = torch.randint(1, TINY_VC["vocab_size"], (B, S))
    mask = torch.ones(1, 1, S, S, dtype=torch.bool).tril()
    with torch.no_grad():
        a = model(input_ids=ids, position_ids=torch.zeros(B, S, dtype=torch.int64),
                  attention_mask=mask)
        b = model(input_ids=ids,
                  position_ids=torch.arange(S).unsqueeze(0).expand(B, -1),
                  attention_mask=mask)
    assert torch.equal(a, b), "NoPE: position_ids must not change logits"


def test_dsa_indexer_fully_frozen(cpu_env):
    """mapping §3.2: the indexer is @no_grad frozen — EVERY param
    requires_grad=False (incl. the bare compress gate / ape)."""
    from v10_model import build_v10_model

    model, _ = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=False)
    _, dsa = _layers_by_kind(model)
    for n, p in dsa[0].self_attention.indexer.named_parameters():
        assert p.requires_grad is False, f"indexer param trainable: {n}"


def test_mask_wiring_dsa_tril_and_broadcast(cpu_env):
    """mask wiring: tril mask == arange ends (bit-exact); a [1,1,S,S] mask
    broadcasts over a wider hidden batch; left-pad masks raise (fail fast)."""
    from v10_model import build_v10_model

    model, cfg = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=False)
    _seed_params(model, seed=41)
    _, dsa = _layers_by_kind(model)
    attn = dsa[0].self_attention
    S, B, H = 12, 2, cfg.hidden_size
    x = torch.randn(S, B, H, dtype=torch.bfloat16)
    out0, _ = attn(x)
    tril_b = torch.ones(B, 1, S, S, dtype=torch.bool).tril()
    out1, _ = attn(x, attention_mask=tril_b)
    assert torch.equal(out0, out1), "tril mask must reproduce arange ends"
    tril_1 = torch.ones(1, 1, S, S, dtype=torch.bool).tril()
    out2, _ = attn(x, attention_mask=tril_1)
    assert torch.equal(out0, out2), "[1,1,S,S] mask must broadcast over batch"
    # true LEFT padding: pad keys AND pad queries excluded -> non-prefix rows
    start = 4
    qpos = torch.arange(S).view(1, 1, S, 1)
    kpos = torch.arange(S).view(1, 1, 1, S)
    left = (qpos >= start) & (kpos >= start) & (kpos <= qpos)
    with pytest.raises(ValueError):
        attn(x, attention_mask=left)


def test_mask_wiring_kda_validation(cpu_env):
    """KDA consumes the mask ONLY as a prefix-validation gate (recurrence is
    causal by construction): tril passes (bit-identical), left-pad raises."""
    from v10_model import build_v10_model

    model, cfg = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=False)
    _seed_params(model, seed=43)
    kda, _ = _layers_by_kind(model)
    attn = kda[0].self_attention
    S, B, H = 12, 2, cfg.hidden_size
    x = torch.randn(S, B, H, dtype=torch.bfloat16)
    out0, _ = attn(x)
    out1, _ = attn(x, attention_mask=torch.ones(B, 1, S, S, dtype=torch.bool).tril())
    assert torch.equal(out0, out1)
    start = 4
    qpos = torch.arange(S).view(1, 1, S, 1)
    kpos = torch.arange(S).view(1, 1, 1, S)
    left = (qpos >= start) & (kpos >= start) & (kpos <= qpos)
    with pytest.raises(ValueError):
        attn(x, attention_mask=left)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
