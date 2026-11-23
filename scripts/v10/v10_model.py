"""V10 model builder — GLM-5.3-Flash (Glm5Next) on mcore 0.18.2 + GPTModel.

Structure: GPTModel(embedding + TransformerBlock(45x Glm5NextLayer) + final
norm + output layer). NoPE (qk_rope=0): the GPTModel rope module is created
(position_embedding_type has no 'none' in 0.18.2) but NEVER called — the
custom layers ignore rotary_pos_emb kwargs.

Meta-build discipline (V8-proven): build under ambient meta ctx + route
torch.cuda.current_device to meta (mcore linear init device beats ambient
ctx), then to_empty(materialize) + meta-cache sweep + inv_freq recompute.
"""
import dataclasses as dc
import json
import os

import torch

from v10_config import (attn_kind_per_layer, ffn_kind_per_layer, install_glm_attrs,
                        v10_config_kwargs, vendor_config)


def _resolved_tp():
    """TP degree the build must use — parallel_state when already initialized
    (the launch script owns it), else WORLD_SIZE. The CUSTOM modules read
    config.tensor_model_parallel_size while mcore linears query
    parallel_state directly — both must agree (mixed-build probe)."""
    try:
        from megatron.core import parallel_state
        if parallel_state.model_parallel_is_initialized():
            return parallel_state.get_tensor_model_parallel_world_size()
    except Exception:
        pass
    return int(os.environ.get("WORLD_SIZE", "1"))


def v10_transformer_config(vc, num_layers=None, num_experts=None, tp_size=None,
                           mtp=False):
    from megatron.core.transformer.transformer_config import MLATransformerConfig
    kwargs = v10_config_kwargs(vc, num_layers)
    kwargs["tensor_model_parallel_size"] = tp_size or _resolved_tp()
    if mtp:
        kwargs["mtp_num_layers"] = vc.get("num_nextn_predict_layers", 1)
    elif "mtp_num_layers" not in kwargs:
        kwargs["mtp_num_layers"] = None  # mtp=False ⇔ config.mtp_num_layers=None (V8 rule)
    if num_experts is not None:
        kwargs["num_moe_experts"] = num_experts
    fields = {f.name for f in dc.fields(MLATransformerConfig)}
    dropped = [k for k in kwargs if k not in fields]
    for k in dropped:
        kwargs.pop(k)
    cfg = MLATransformerConfig(**kwargs)
    install_glm_attrs(cfg, vc)
    # DSA indexer attrs (0.18.2 MLATransformerConfig has no dsa_* fields)
    cfg.dsa_indexer_topk = vc["index_topk"]
    cfg.dsa_indexer_n_heads = vc["index_n_heads"]
    cfg.dsa_indexer_head_dim = vc["index_head_dim"]
    cfg.dsa_indexer_kpool = vc["index_kpool"]
    cfg.dsa_indexer_always_select_tail = vc.get("index_kpool_always_select_tail", True)
    return cfg


def build_layer_specs(cfg, vc, num_layers=None):
    """45 (or truncated) Glm5NextLayer specs: KDA/DSA attention x dense/MoE mlp."""
    from functools import partial

    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.spec_utils import ModuleSpec
    from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
    from megatron.core import tensor_parallel

    from v10_dsa import V10DSAAttention
    from v10_kda import GlmKDAAttention, GlmKDASubmodules
    from v10_layers import Glm5NextLayer, Glm5NextLayerSubmodules
    from v10_modules import GlmRMSNorm

    attns = attn_kind_per_layer(vc)
    ffns = ffn_kind_per_layer(vc)
    if num_layers is not None:
        attns, ffns = attns[:num_layers], ffns[:num_layers]

    kda_attn = ModuleSpec(module=GlmKDAAttention, submodules=GlmKDASubmodules())
    dsa_attn = ModuleSpec(module=V10DSAAttention)
    dense_mlp = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(
            linear_fc1=tensor_parallel.ColumnParallelLinear,
            linear_fc2=tensor_parallel.RowParallelLinear))
    moe_mlp = get_moe_module_spec(use_te=False, num_experts=cfg.num_moe_experts,
                                  moe_grouped_gemm=False)
    from v10_grouped_experts import V10GroupedExperts, grouped_enabled
    if grouped_enabled():
        # pad+bmm experts (launch-bound fix, ledger #17): drop-in for
        # SequentialMLP — same build call (num_local_experts, config,
        # pg_collection=, name=) and same forward signature.
        # get_moe_module_spec returns partial(MoELayer, submodules=MoESubmodules(...))
        moe_mlp.keywords["submodules"].experts = V10GroupedExperts

    specs = []
    n = len(attns)
    for i, (a, f) in enumerate(zip(attns, ffns)):
        specs.append(ModuleSpec(
            module=Glm5NextLayer,
            submodules=Glm5NextLayerSubmodules(
                input_layernorm=GlmRMSNorm,
                self_attention=kda_attn if a == "kda" else dsa_attn,
                pre_mlp_layernorm=GlmRMSNorm,
                mlp=dense_mlp if f == "dense" else moe_mlp),
            params={"is_first": i == 0, "is_last": i == n - 1}))
    return TransformerBlockSubmodules(layer_specs=specs, layer_norm=GlmRMSNorm)


def _init_dist(device, tp_size=None):
    import torch.distributed as dist
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", os.environ.get("V10_MASTER_PORT", "29677"))
        backend = "hccl" if device != "cpu" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://",
                                world_size=int(os.environ.get("WORLD_SIZE", "1")),
                                rank=int(os.environ.get("RANK", "0")))
    from megatron.core import parallel_state
    if not parallel_state.model_parallel_is_initialized():
        # default TP = full world (V10 is a TP-only deployment; DP/PP/EP added
        # in later phases). initialize_model_parallel() DEFAULTS TP=1 — a
        # 16-rank launch would build the FULL model per rank (OOM probe).
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=tp_size or int(
                os.environ.get("WORLD_SIZE", "1")))


def _gpt_model(cfg, block_spec, vocab_size, max_seq_len, mtp_block_spec=None):
    from megatron.core.models.gpt.gpt_model import GPTModel
    return GPTModel(
        config=cfg,
        transformer_layer_spec=block_spec,
        vocab_size=vocab_size,
        max_sequence_length=max_seq_len,
        pre_process=True, post_process=True,
        parallel_output=True,
        share_embeddings_and_output_weights=False,
        mtp_block_spec=mtp_block_spec,
        # 0.18.2 has no 'none'; the rotary module is created but our layers
        # never consume it (Flash = NoPE everywhere)
        position_embedding_type="rope")


def build_v10_model(vc=None, num_layers=None, device="meta", mtp=False,
                    num_experts=None, vocab_size=None, max_seq_len=2048,
                    tp_size=None):
    """Build the V10 Glm5Next model. device: 'meta' (skeleton) | 'cpu' | 'npu'.
    tp_size: TP degree for the build (default = WORLD_SIZE). mtp is Phase-2
    (False in Phase 1 — the semantic gate is main-head CE)."""
    vc = vc or vendor_config()
    if device != "meta":
        _init_dist(device, tp_size)

    cfg = v10_transformer_config(vc, num_layers, num_experts,
                                 tp_size=tp_size or _resolved_tp(), mtp=mtp)
    block_spec = build_layer_specs(cfg, vc, num_layers)
    vocab = vocab_size or vc["vocab_size"]

    mtp_block_spec = None
    if mtp:
        from v10_mtp import build_mtp_block_spec
        mtp_block_spec = build_mtp_block_spec(
            num_experts=cfg.num_moe_experts)

    if device == "meta":
        with torch.device("meta"):
            model = _gpt_model(cfg, block_spec, vocab, max_seq_len, mtp_block_spec)
        return model, cfg

    # meta-build routing: mcore linear init allocates on
    # torch.cuda.current_device() EXPLICITLY (torch_npu aliases cuda) — route
    # to meta, then materialize (V8-proven discipline)
    _orig = torch.cuda.current_device
    torch.cuda.current_device = lambda: "meta"
    try:
        with torch.device("meta"):
            model = _gpt_model(cfg, block_spec, vocab, max_seq_len, mtp_block_spec)
    finally:
        torch.cuda.current_device = _orig

    dev = torch.device(device) if device == "cpu" else \
        torch.device("npu", torch.npu.current_device())
    model.to_empty(device=dev)
    cleared = _meta_cache_sweep(model)
    n_fixed = _fix_meta_inv_freq(model, dev)
    print(f"[v10-model] meta cache sweep: {len(cleared)} cleared; "
          f"inv_freq recomputed: {n_fixed} (unused NoPE module)", flush=True)
    return model, cfg


def _meta_cache_sweep(model):
    """to_empty materializes params/buffers ONLY — plain-attribute caches
    pre-warmed on meta must be cleared (V8 audit round-15)."""
    cleared = []

    def _has_meta(v):
        if isinstance(v, torch.Tensor):
            return v.device.type == "meta"
        if isinstance(v, (tuple, list)):
            return any(isinstance(t, torch.Tensor) and t.device.type == "meta"
                       for t in v)
        return False

    for mod_name, m in model.named_modules():
        d = getattr(m, "__dict__", {})
        for attr in list(d.keys()):
            if attr in ("_parameters", "_buffers", "_modules",
                        "_non_persistent_set"):
                continue
            val = d[attr]
            if isinstance(val, dict):
                for k in list(val.keys()):
                    if _has_meta(val[k]):
                        val.pop(k)
                        cleared.append(f"{mod_name or 'ROOT'}.{attr}[{k!r}]")
            elif _has_meta(val) and not isinstance(val, torch.nn.Parameter):
                if attr == "inv_freq":
                    continue
                cleared.append(f"REPORT-ONLY {mod_name or 'ROOT'}.{attr}")
    return cleared


def _fix_meta_inv_freq(model, dev):
    """RotaryEmbedding.inv_freq plain attribute created on meta (to_empty
    blind). Rope is UNUSED by V10 layers — recompute only to keep the module
    instantiable (V8 closed formula, defaults)."""
    base = float(getattr(model.config, "rotary_base", 10000.0))
    fixed = 0
    for m in model.modules():
        inv = getattr(m, "inv_freq", None)
        if isinstance(inv, torch.Tensor) and inv.device.type == "meta":
            dim = int(inv.shape[-1]) * 2
            m.inv_freq = 1.0 / (base ** (
                torch.arange(0, dim, 2, dtype=torch.float32, device=dev) / dim))
            fixed += 1
    return fixed
