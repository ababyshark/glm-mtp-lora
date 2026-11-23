"""V10 config — GLM-5.3-Flash-BF16 text_config -> mcore 0.18.2
MLATransformerConfig kwargs + hybrid layer pattern helpers.

Single source of truth for dims/flags; semantic source is
AIfromScratch/glm53flash-mtp/docs/ARCHITECTURE_MAPPING.md (+ glm_config.py).

V10 router = vendor noaux_tc (V8-proven kwargs): score_function sigmoid,
load_balancing none (frozen base, no aux loss), static expert bias
(bias_update_rate 0.0), topk scaling 2.5, fp32 router. e_score_correction_bias
lands in router.expert_bias buffers via fill.
"""
import json
import os

import torch

VENDOR_CFG = os.environ.get(
    "V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16") + "/config.json"


def vendor_config(path=VENDOR_CFG):
    cfg = json.load(open(path))
    return cfg.get("text_config", cfg)


def v10_config_kwargs(vc, num_layers=None):
    """MLATransformerConfig kwargs — Flash-corrected."""
    nl = num_layers if num_layers is not None else vc["num_hidden_layers"]
    kd = min(vc["first_k_dense_replace"], max(nl - 1, 0)) if nl < vc["num_hidden_layers"] \
        else vc["first_k_dense_replace"]
    kwargs = dict(
        hidden_size=vc["hidden_size"],
        num_layers=nl,
        num_attention_heads=vc["num_attention_heads"],
        # DSA dims (custom V10DSAAttention reads these off the config)
        multi_latent_attention=True,
        q_lora_rank=vc["q_lora_rank"],
        kv_lora_rank=vc["kv_lora_rank"],
        qk_head_dim=vc["qk_nope_head_dim"],      # nope only (qk_rope_head_dim=0)
        qk_pos_emb_head_dim=vc["qk_rope_head_dim"],
        v_head_dim=vc["v_head_dim"],
        # DSA indexer
        dsa_indexer_topk=vc["index_topk"],
        dsa_indexer_n_heads=vc["index_n_heads"],
        dsa_indexer_head_dim=vc["index_head_dim"],
        dsa_indexer_kpool=vc["index_kpool"],
        # MoE (vendor noaux_tc, V8-proven kwargs)
        num_moe_experts=vc["n_routed_experts"],
        moe_router_topk=vc["num_experts_per_tok"],
        moe_ffn_hidden_size=vc["moe_intermediate_size"],
        moe_router_load_balancing_type="none",
        moe_router_score_function="sigmoid",
        moe_router_pre_softmax=False,
        moe_router_enable_expert_bias=True,
        moe_router_bias_update_rate=0.0,
        moe_router_topk_scaling_factor=vc["routed_scaling_factor"],
        moe_router_dtype="fp32",
        moe_shared_expert_intermediate_size=vc["moe_intermediate_size"],
        moe_layer_freq=[0] * kd + [1] * (nl - kd),
        moe_grouped_gemm=False,
        # dense MLP
        ffn_hidden_size=vc["intermediate_size"],
        gated_linear_unit=True,
        add_bias_linear=False,
        # swiglu_limit=10: mcore 0.18.2 native (mlp.py:221-223 — gate.clamp(max),
        # up.clamp(±), exactly GLM MG:137-142)
        activation_func=torch.nn.functional.silu,
        activation_func_clamp_value=vc.get("swiglu_limit"),
        layernorm_epsilon=vc.get("rms_norm_eps", 1e-5),
        normalization="RMSNorm",
        # NoPE everywhere: no rotary module at all (Flash has rope=0)
        qk_layernorm=False,
        bf16=True,
        params_dtype=torch.bfloat16,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        perform_initialization=False,
        calculate_per_token_loss=False,
    )
    # activation recompute (gradient checkpointing) — LoRA training backprops
    # through all 45 layers; full recompute drops the activation residency so
    # B=4 fits the 61.28G cards (B=4 full-graph OOMed at 59.5G, 2026-11-20).
    # NOTE: this MUST run before the return — the first insertion landed after
    # `return` and was silently dead (found by the .40 OOM replay).
    import os as _os
    rc = _os.environ.get("GLM53_V10_RECOMPUTE", "full")
    if rc and rc != "off":
        kwargs["recompute_granularity"] = rc          # 'full' | 'selective'
        if rc == "full":
            kwargs["recompute_method"] = "uniform"
            kwargs["recompute_num_layers"] = 1
    return kwargs


def attn_kind_per_layer(vc):
    """KDA/DSA per layer — from layer_types ('linear_attention' = KDA)."""
    out = []
    for t in vc["layer_types"]:
        if t == "linear_attention":
            out.append("kda")
        elif t in ("deepseek_sparse_attention", "full_attention"):
            out.append("dsa")
        else:
            raise ValueError(f"unknown layer_types entry {t!r}")
    assert len(out) == vc["num_hidden_layers"]
    return out


def ffn_kind_per_layer(vc):
    k = vc["first_k_dense_replace"]
    n = vc["num_hidden_layers"]
    if vc.get("mlp_layer_types"):
        return ["dense" if t == "dense" else "moe" for t in vc["mlp_layer_types"]]
    return ["dense"] * k + ["moe"] * (n - k)


def install_glm_attrs(cfg, vc):
    """glm_* custom attrs the ported modules read (reference glm_config.py
    naming, values == Flash real values)."""
    lac = vc.get("linear_attn_config") or {}
    cfg.glm_linear_num_heads = lac.get("num_heads", 64)
    cfg.glm_linear_head_dim = lac.get("head_dim", 128)
    cfg.glm_linear_conv_kernel_dim = lac.get("short_conv_kernel_size", 4)
    cfg.glm_gate_lower_bound = lac.get("gate_lower_bound", -5.0)
    cfg.glm_linear_activation = "silu"
    cfg.glm_mhc_streams = vc.get("hc_mult", 4)
    cfg.glm_hc_eps = vc.get("hc_eps", 1e-6)
    cfg.glm_mhc_sinkhorn_iters = vc.get("hc_sinkhorn_iters", 20)
    cfg.dsa_indexer_always_select_tail = vc.get("index_kpool_always_select_tail", True)
    return cfg
