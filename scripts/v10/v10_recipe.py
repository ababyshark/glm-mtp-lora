"""V10 Phase 4: the V5.4a judgment recipe on the Flash (Glm5Next) MTP.

Geometry DIFFERS from the W4A8 port (the V8 line's v8_recipe.py, source repo
only — NOT merged here):
- Flash MTP inner layer is a STANDARD SEQUENCE self-attention layer (DSv3
  structure): fused = eh_proj(cat) -> causal self-attn over S -> mlp. There is
  NO depth-axis attention, NO hd_stack chaining (that was the W4A8 vendor's
  multi-depth quirk; Flash is depth=1, serving replays the layer).
- NO rot, NO own embed (main embedding), NO own draft head — shared head =
  MAIN output_layer (checkpoint shared_head.head == lm_head family).
- mcore concat order [e | h] matches the fill's swapped eh_proj (v10_mtp_fill).

Losses (branch-1 V5.4a semantics, env knobs identical):
- CE: mean over supervised rows (chunked_supervised_ce port — shared with V8)
- margin: mean(relu(m − (logit_tgt − max_other))) × GLM53_ACC_MARGIN_WEIGHT
- KL(verifier||draft) at matched rows × GLM53_MTP_DISTILL_WEIGHT (2.33);
  verifier = main lm_head on h0, detached, row-chunked, vocab-gathered
- agree-CE chasing the verifier argmax × GLM53_AGREE_CE_WEIGHT (0.5)
- final: mean_rows(CE + margin_term) + dw·mean_kl + aw·mean_agree
"""
import os

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


def _tp():
    try:
        from megatron.core import parallel_state
        return (parallel_state.get_tensor_model_parallel_world_size(),
                parallel_state.get_tensor_model_parallel_rank())
    except Exception:
        return 1, 0


def _head_out(head, rows):
    o = head(rows)
    return o[0] if isinstance(o, tuple) else o


def _vocab_gather(sharded):
    """[rows, V/tp] -> [rows, V] — DIFFERENTIABLE over TP.

    BUG FIX (2026-11-20, error-ledger #18): the previous dist.all_gather is
    NOT differentiable — at TP16 it SEVERED the gradient of the recipe's CE
    term on the grad-carrying draft rows (V8's TP16 training had the same
    silent severing in its W4A8 port). Uses mcore's autograd-native gather; under
    no_grad it degrades to a plain gather (the verifier call path)."""
    tp, _ = _tp()
    if tp == 1:
        return sharded
    from megatron.core import tensor_parallel
    return tensor_parallel.gather_from_tensor_model_parallel_region(
        sharded.contiguous())


# chunked_ce_margin is the shared V5.4a asset — the V8-line verbatim copy
# (geometry-independent: head + normed rows + targets), vendored as
# v10_ce_margin.py for self-containment.
from v10_ce_margin import chunked_ce_margin  # noqa: E402,F401


def draft_stream(model, input_ids, base_hidden):
    """One depth-1 draft pass over the whole fused sequence.

    input_ids [B, S] batch-first (B=1); base_hidden [S, B, H] (decoder output,
    DETACHED by the caller — the recipe never grads the base).
    Returns normed draft hidden [S, B, H] (final_layernorm'd, grad-carrying).
    """
    mtp = model.mtp
    layer = mtp.layers[0]
    inner = layer.mtp_model_layer if hasattr(layer, "mtp_model_layer") else layer
    dev = input_ids.device
    B, S = input_ids.shape
    h0 = base_hidden                                   # [S, B, H] detached
    # e = Emb(t+1): roll LEFT (mcore MTP native path rolls -1 identically)
    rolled = torch.roll(input_ids, -1, dims=-1)
    emb = model.embedding(input_ids=rolled, position_ids=None)  # [S, B, H]
    if emb.dim() == 4:  # hybrid VocabParallelEmbedding variance guard
        emb = emb.reshape(emb.shape[0], emb.shape[1], -1)
    fused = layer.eh_proj(torch.cat([layer.enorm(emb), layer.hnorm(h0)], dim=-1))
    fused = fused[0] if isinstance(fused, tuple) else fused
    tp_size, _ = _tp()
    if tp_size > 1:
        # eh_proj is built gather_output=False in mcore MTP; the recipe owns
        # the gather (same as the V8 slim-eh_proj handling)
        from megatron.core import tensor_parallel
        fused = tensor_parallel.gather_from_tensor_model_parallel_region(fused)
    # standard causal self-attention over the fused sequence + residual + mlp.
    # Mask contract (v10_modules.causal_ends_from_mask): bool KEEP-mask
    # [B,1,S,S], True=keep, rows must be causal prefixes (tril), NOT the
    # triu True=masked form.
    causal_keep = torch.ones(S, S, dtype=torch.bool, device=dev).tril() \
        .unsqueeze(0).unsqueeze(0)                     # [1,1,S,S] True=keep
    out = inner(fused, attention_mask=causal_keep)
    out = out[0] if isinstance(out, tuple) else out
    return layer.final_layernorm(out)                  # [S, B, H]


def serving_chain_recipe_loss(model, input_ids, labels, base_hidden):
    """The V5.4a recipe on Flash depth-1. input_ids/labels [B, S] (B=1 enforced,
    branch-1 parity). base_hidden [S, B, H] decoder output (detached inside).

    Loss decomp via model._last_kl / model._last_agree (GLM53_LOSS_DECOMP=1).
    """
    dev = input_ids.device
    ids = input_ids.to(dev)
    labels = labels.to(dev)
    B, S = ids.shape
    assert B == 1, "serving_chain KL/agree-CE require per-device B=1"
    tp_size, rank = _tp()
    h0 = base_hidden.detach()

    # env knobs (branch-1 identical; depth fixed at 1 for Flash)
    dw = float(os.environ.get("GLM53_MTP_DISTILL_WEIGHT", "0") or 0)
    aw = float(os.environ.get("GLM53_AGREE_CE_WEIGHT", "0") or 0)
    if aw > 0 and dw <= 0:
        raise RuntimeError("GLM53_AGREE_CE_WEIGHT requires the verifier "
                           "stream (GLM53_MTP_DISTILL_WEIGHT>0) — branch-1 "
                           ":1765 invariant")
    acc_w = float(os.environ.get("GLM53_ACC_MARGIN_WEIGHT", "0") or 0)
    acc_margin = float(os.environ.get("GLM53_ACC_MARGIN", "0.5") or 0)
    chunk = int(os.environ.get("GLM53_HEAD_CE_CHUNK", "512") or 512)

    # verifier stream (once, detached): main lm_head on h0, row-chunked
    vlp = vargmax = None
    if dw > 0:
        rows_v = h0[:, 0].detach()                     # [S, H]
        _vrows = int(os.environ.get("GLM53_VERIFIER_ROWS", "128"))
        vlp_rows, varg_rows = [], []
        for r0 in range(0, rows_v.shape[0], _vrows):
            vlg = _head_out(model.output_layer, rows_v[r0:r0 + _vrows])
            vlg = _vocab_gather(vlg) if tp_size > 1 else vlg
            vlg = vlg.float()
            varg_rows.append(vlg.argmax(dim=-1).detach())
            vlp_rows.append(torch.log_softmax(vlg, dim=-1).to(torch.bfloat16))
            del vlg
        vargmax = torch.cat(varg_rows, dim=0).detach()
        vlp = torch.cat(vlp_rows, dim=0).detach()

    normed = draft_stream(model, ids, h0)              # [S, B, H] grad-carrying
    rows = normed[:, 0]                                # [S, H]
    # draft predicts token t+1 from fused position t -> rows[:-1], labels[1:]
    mtp_labels = labels[0][1:]
    head = model.output_layer                          # shared head
    # chunked_ce_margin returns (ce, margin) iff want_margin else a scalar —
    # unpacking a 0-d tensor raises (probe); branch on the flag
    res = chunked_ce_margin(head, rows[:-1], mtp_labels,
                            chunk=max(1, chunk), want_margin=acc_w > 0,
                            margin=acc_margin, tp_gather=tp_size > 1)
    if acc_w > 0:
        ce, mg = res
    else:
        ce, mg = res, None
    cnt = (mtp_labels != IGNORE_INDEX).sum()
    ce_sum = ce * torch.clamp(cnt, min=1).float()
    out = ce_sum / torch.clamp(cnt, min=1).float()
    if acc_w > 0:
        out = out + acc_w * (mg / torch.clamp(cnt, min=1).float())

    _kl_s = torch.zeros((), device=dev, dtype=torch.float32)
    _kl_n = 0
    _ag_s = torch.zeros((), device=dev, dtype=torch.float32)
    _ag_n = 0
    if vlp is not None:
        dr = rows[:-1]                                 # draft rows [S-1, H]
        vr = vlp[:dr.shape[0]]                         # verifier rows (t+1 aligned)
        dlg = _head_out(head, dr)
        dlg = _vocab_gather(dlg) if tp_size > 1 else dlg
        dlg = dlg.float()
        dlp = torch.log_softmax(dlg, dim=-1)
        pv = vr.float().exp()
        kl = (pv * (vr.float() - dlp)).sum(dim=-1).mean()
        _kl_s = _kl_s + kl
        _kl_n += 1
        if aw > 0:
            va = vargmax[:dr.shape[0]]
            sup = mtp_labels != IGNORE_INDEX
            agree = F.cross_entropy(dlg[sup], va[sup]) if sup.any() else None
            if agree is not None:
                _ag_s = _ag_s + agree
                _ag_n += 1
    if dw > 0 and _kl_n > 0:
        out = out + dw * (_kl_s / _kl_n)
    if aw > 0 and _ag_n > 0:
        out = out + aw * (_ag_s / _ag_n)
    model._last_kl = (float(_kl_s.detach()) / _kl_n) if _kl_n else None
    model._last_agree = (float(_ag_s.detach()) / _ag_n) if _ag_n else None
    return out
