"""V10 MTP layer (layers.45) — Glm5Next draft layer on mcore 0.18.2's
MultiTokenPrediction machinery. NON-QUANTIZED clean path (unlike the V8
W4A8 contract, Flash MTP has NO own-embed / NO rot / NO own draft head):

  h = hnorm(hidden);  e = enorm(Emb(t+1))          # main embedding (mcore passes it)
  x = eh_proj(cat([h, e]))                          # combined [4096, 8192]
  h_mtp = MTPInnerLayer(x)                          # DSA attn + MoE, STANDARD residual
  logits = lm_head(final_layernorm(h_mtp))          # shared head = MAIN output_layer

mcore 0.18.2 wiring (probed 2026-11-19):
- MultiTokenPredictionLayer builds enorm/hnorm/eh_proj(=ColumnParallelLinear(2h,h))/final_layernorm
  from its submodules spec; the inner layer from mtp_model_layer spec.
- _concat_embeddings cats (enorm(emb), hnorm(hidden)) = **[e | h]** order —
  and the GLM checkpoint eh_proj is ALREADY [e | h] (diag v2 cell D,
  2026-11-20: raw order draft CE 1.37 vs swapped 12.72 — the earlier [h|e]
  推断 was WRONG; fill swaps OFF by default, GLM53_V10_EH_SWAP=1 to repro).
- final_layernorm == checkpoint shared_head.norm.
- The inner self_attention spec MUST carry params={"attn_mask_type": causal}
  (MultiTokenPredictionLayer SUPPORTED_ATTN_MASK assert).
- Draft head = main output_layer (mcore process_mtp_loss default) ✓ Flash shared-head.
"""
from megatron.core import tensor_parallel
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionBlock,
    MultiTokenPredictionBlockSubmodules,
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_layer import TransformerLayer, TransformerLayerSubmodules
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.utils import make_viewless_tensor

from v10_modules import GlmRMSNorm


def _patch_mtp_embedding_no_roll():
    """GLM/DSv3 depth-1 convention (diag E4, 2026-11-20): the draft input
    embedding is e(t) — the CURRENT token's embedding — paired with h(t) to
    predict t+1. mcore 0.18.2 _get_embeddings rolls input_ids by -1 (e(t+1)),
    which puts the vendor-trained MTP off-distribution (draft CE 12.72 with
    e(t+1) vs 11.73 with e(t); a trained draft must be ~2-6). This patch
    removes the INPUT roll only — the LABEL roll in process_mtp_loss (-1,
    predict t+1) is correct and stays. Env: GLM53_V10_MTP_E_T=0 disables."""
    import megatron.core.transformer.multi_token_prediction as _mtp_mod

    orig = _mtp_mod.MultiTokenPredictionLayer._get_embeddings
    if getattr(orig, "_v10_no_roll", False):
        return

    def no_roll_embeddings(self, input_ids, position_ids, embedding,
                           hidden_states, packed_seq_params=None,
                           padding_mask=None):
        decoder_input = embedding(input_ids=input_ids, position_ids=position_ids)
        decoder_input = decoder_input[0] if isinstance(decoder_input, tuple) \
            else decoder_input
        hidden_states = make_viewless_tensor(inp=hidden_states,
                                             requires_grad=True, keep_graph=True)
        return input_ids, position_ids, padding_mask, decoder_input, hidden_states

    no_roll_embeddings._v10_no_roll = True
    _mtp_mod.MultiTokenPredictionLayer._get_embeddings = no_roll_embeddings


def _patch_mtp_checkpoint_padding_mask():
    """mcore 0.18.2 (571370c82) bug: MultiTokenPredictionLayer.forward passes
    padding_mask into _checkpointed_forward, whose signature lacks it — fires
    the moment recompute_granularity=full + MTP (V10 T4, 2026-11-20). Flash
    has no padding_mask semantics (always None in V10): dropping the kwarg is
    exactly equivalent."""
    import megatron.core.transformer.multi_token_prediction as _m
    cls = _m.MultiTokenPredictionLayer
    orig = cls._checkpointed_forward
    if getattr(orig, "_v10_padding_mask_fix", False):
        return

    def fixed(self, *args, **kwargs):
        kwargs.pop("padding_mask", None)
        return orig(self, *args, **kwargs)

    fixed._v10_padding_mask_fix = True
    cls._checkpointed_forward = fixed


def _mtp_inner_spec(moe):
    """Standard-residual inner layer: DSA attn + MoE (checkpoint layers.45 has
    input_layernorm/post_attention_layernorm, NO hc_* keys)."""
    from v10_dsa import V10DSAAttention

    attn = ModuleSpec(module=V10DSAAttention,
                      params={"attn_mask_type": AttnMaskType.causal})
    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=GlmRMSNorm,
            self_attention=attn,
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=GlmRMSNorm,
            mlp=moe,
            mlp_bda=get_bias_dropout_add,
        ),
    )


def build_mtp_block_spec(num_experts=288):
    """MTP block spec with the Flash draft layer (MoE experts = main stack's)."""
    import os

    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec

    if os.environ.get("GLM53_V10_MTP_E_T", "1") == "1":
        _patch_mtp_embedding_no_roll()
    _patch_mtp_checkpoint_padding_mask()
    moe = get_moe_module_spec(use_te=False, num_experts=num_experts,
                              moe_grouped_gemm=False)
    from v10_grouped_experts import V10GroupedExperts, grouped_enabled
    if grouped_enabled():
        # the MTP draft layer is MoE too — same pad+bmm swap (mtp.experts
        # is full-param trainable per the freeze contract)
        moe.keywords["submodules"].experts = V10GroupedExperts
    inner = _mtp_inner_spec(moe)

    mtp_layer_spec = ModuleSpec(
        module=MultiTokenPredictionLayer,
        submodules=MultiTokenPredictionLayerSubmodules(
            enorm=GlmRMSNorm,
            hnorm=GlmRMSNorm,
            eh_proj=tensor_parallel.ColumnParallelLinear,
            mtp_model_layer=inner,
            layer_norm=GlmRMSNorm,          # == checkpoint shared_head.norm
        ),
    )
    return MultiTokenPredictionBlockSubmodules(layer_specs=[mtp_layer_spec])
