#!/usr/bin/env python
"""V8 Phase 4: the V5.4a serving_chain recipe on the mcore V8 model.

Faithful port of branch-1 `_mtp_loss_serving_chain`
(modeling_glm53_w4a8_pipe_mtp.py:1713) + chunked_supervised_ce (:800).
Geometry preserved EXACTLY (this is the plan §3 "most expensive asset"):
- chunk-local depth chains (chunk=128): hd_stack KV chaining across depths,
  hd_seq = stack(previous depths' input_layernorm'd tokens + current) →
  attention over the depth axis, take the LAST row (vendor causal_mask(k) is
  row-causal, so the last row attends all k — internal-causal ≡ no-mask+last)
- per-depth rot BEFORE hnorm (V8 mtp_rot; vendor :1126/:1355 semantics)
- own embed, zero at the ABSOLUTE pos==0 (pos_now = c0+i+k)
- eh_proj(cat(enorm(emb), hnorm(prev))); positions for the depth-axis
  attention = c0+i+(j+1) (one ahead of the input position — vendor convention)
- normed_k = final_layernorm(out_k); CE head = mtp_head (vendor draft head)
- labels slice = ids[c0+k+1 : c0+k+s_k] (predict-one-ahead per depth)

Losses (branch-1 exact, env knobs identical):
- CE: mean over supervised rows (sum/count; -100 ignored), chunked
- margin: mean(relu(m − (logit_tgt − max_other))) × GLM53_ACC_MARGIN_WEIGHT
  (0.3), margin m=0.5, depth weights GLM53_ACC_DEPTH_WEIGHTS (3,4,3,1,1 →
  normalized) — measured semantics from chunked_supervised_ce
- KL: p_v·(log p_v − log p_d) at matched rows (verifier row c0+k ↔ draft row
  c0+k..) × GLM53_MTP_DISTILL_WEIGHT (2.33). Verifier = main lm_head on h0,
  detached, computed once. NOTE: code authority — the docstring "KL(draft||v)"
  in branch-1 is stale; the code computes KL(verifier||draft).
- agree-CE: CE chasing the verifier's argmax on supervised rows
  × GLM53_AGREE_CE_WEIGHT (0.5)
- final: mean_rows(CE + margin_term) + dw·mean_kl + aw·mean_agree

mcore asset map (branch-1 concept → V8):
  mtp.embed_tokens        → model.mtp.mtp_embed (+ zero-quirk inline)
  self.rot                → model.mtp.mtp_rot (V7 contract: returns (out, bias))
  mtp.enorm / hnorm       → mtp.layers[0].enorm / .hnorm
  mtp.eh_proj             → mtp.layers[0].eh_proj (+ TP16 explicit gather —
                            the slim eh_proj keeps gather_output=False; the
                            mcore-internal gather at :903 only serves the
                            NATIVE mtp path, not this chain)
  mtp.input_layernorm     → layers[0].transformer_layer.input_layernorm
  mtp.self_attn           → ...self_attention (mcore MLA: rope internally from
                            position_ids, internal causal ≡ vendor mask)
  mtp.post_attention_layernorm → ...pre_mlp_layernorm
  mtp.mlp                 → ...mlp
  mtp.shared_head.norm    → mtp.layers[0].final_layernorm
  mtp.shared_head.head    → model.mtp_head (+ TP16 vocab gather for full-logit CE)
  base lm_head (verifier) → model.output_layer (+ TP16 vocab gather)
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
    """[rows, V/tp] → [rows, V] (all-gather over TP, cat along vocab)."""
    tp, rank = _tp()
    if tp == 1:
        return sharded
    import torch.distributed as dist
    from megatron.core import parallel_state
    group = parallel_state.get_tensor_model_parallel_group()
    parts = [torch.empty_like(sharded) for _ in range(tp)]
    dist.all_gather(parts, sharded.contiguous(), group=group)
    return torch.cat(parts, dim=-1)


def _head_full_logits(head, rows, chunk):
    """head(rows) with TP16 vocab gather, row-chunked (HBM discipline)."""
    outs = []
    for c0 in range(0, rows.shape[0], chunk):
        outs.append(_vocab_gather(head(rows[c0:c0 + chunk])))
    return torch.cat(outs, dim=0) if len(outs) > 1 else outs[0]


def chunked_ce_margin(head, normed_rows, targets, chunk=128,
                      want_margin=False, margin=0.5, tp_gather=False):
    """Port of chunked_supervised_ce: mean CE over supervised rows (sum/count)
    + optional margin mean(relu(m − (logit_tgt − max_other))). Exact math."""
    sup = targets != IGNORE_INDEX
    zero = normed_rows.sum(dtype=torch.float32) * 0.0
    if not torch.any(sup):
        return (zero, zero) if want_margin else zero
    hs, tg = normed_rows[sup], targets[sup]

    def apply(r):
        lg = _head_out(head, r)
        return _vocab_gather(lg) if tp_gather else lg

    ce_sum, cnt, mg_sum = None, 0, None
    for c0 in range(0, hs.shape[0], max(1, chunk)):
        logits = apply(hs[c0:c0 + chunk]).float()
        part = F.cross_entropy(logits, tg[c0:c0 + chunk], ignore_index=IGNORE_INDEX,
                               reduction="sum")
        ce_sum = part if ce_sum is None else ce_sum + part
        if want_margin:
            tgt = tg[c0:c0 + chunk]
            t_logit = logits.gather(1, tgt.unsqueeze(1)).squeeze(1)
            top2 = logits.topk(2, dim=1)
            other = torch.where(top2.indices[:, 0] == tgt,
                                top2.values[:, 1], top2.values[:, 0])
            mg = F.relu(margin - (t_logit - other)).sum()
            mg_sum = mg if mg_sum is None else mg_sum + mg
        cnt += tg[c0:c0 + chunk].numel()
    ce = ce_sum / max(1, cnt)
    if want_margin:
        return ce, mg_sum / max(1, cnt)
    return ce


def serving_chain_recipe_loss(model, input_ids, labels, base_hidden,
                              self_feed=False):
    """The V5.4a serving_chain recipe. input_ids/labels: [B, S] (B=1 enforced,
    same as branch-1 — the KL/agree verifier stream is built from the h0 flow).
    base_hidden: [S, B, H] the main decoder's last hidden (mcore layout;
    detached inside).

    Returns the scalar loss. Per-depth CE ladder via model._last_depth_losses,
    KL/agree via model._last_kl / model._last_agree (GLM53_LOSS_DECOMP=1).
    """
    mtp = model.mtp
    layer = mtp.layers[0]
    dev = input_ids.device
    # LAYOUT TRUTH (parity + embedding-source converged): the batcher feeds
    # BATCH-FIRST [B, S] (GPTModel.forward's contract); the recipe keeps
    # branch-1's batch-first indexing (ids[0] = the S-flow) — AS-IS, no
    # transpose. base_hidden = the DECODER output = seq-first [S, B, H]
    # (the embedding transposes [B,S,H]→[S,B,H] for the seq-first decoder;
    # the logits transpose back to [B,S,V] — parity-verified).
    ids = input_ids.to(dev)
    labels = labels.to(dev)
    B, S = ids.shape
    assert B == 1, "serving_chain KL/agree-CE require per-device B=1"
    tp, rank = _tp()
    h0 = base_hidden.detach()  # [B, S, H]
    assert B == 1, "serving_chain KL/agree-CE require per-device B=1"
    tp, rank = _tp()
    h0 = base_hidden.detach()  # [S, B, H]

    # env knobs (branch-1 identical)
    dw = float(os.environ.get("GLM53_MTP_DISTILL_WEIGHT", "0") or 0)
    aw = float(os.environ.get("GLM53_AGREE_CE_WEIGHT", "0") or 0)
    if aw > 0 and dw <= 0:
        raise RuntimeError("GLM53_AGREE_CE_WEIGHT requires the verifier "
                           "stream (GLM53_MTP_DISTILL_WEIGHT>0) — branch-1 "
                           ":1765 invariant; silent skip would change the "
                           "recipe semantics")
    acc_w = float(os.environ.get("GLM53_ACC_MARGIN_WEIGHT", "0") or 0)
    acc_margin = float(os.environ.get("GLM53_ACC_MARGIN", "0.5") or 0)
    chunk = int(os.environ.get("GLM53_HEAD_CE_CHUNK", "512") or 512)
    chain_chunk = 128  # vendor hardcoded: chunk-local depth chains
    depths = int(os.environ.get("GLM53_MTP_DEPTHS", "5"))
    K = max(1, min(depths, S - 2))
    _acc_ws = None
    if acc_w > 0:
        _raw = os.environ.get("GLM53_ACC_DEPTH_WEIGHTS", "3,4,3,1,1")
        _ws = [float(x) for x in _raw.split(",") if x.strip()]
        _t = sum(_ws)
        _acc_ws = [w / _t for w in _ws] if _t > 0 else None

    # verifier stream (once, detached): main lm_head on h0 flow — ROW-CHUNKED
    # (branch-1 streams vocab slices; row-chunking achieves the same HBM cap:
    # [chunk_rows, V] fp32 transient ≈ 80MB, the full vlp bf16 = 634MB resident)
    vlp = vargmax = None
    if dw > 0:
        rows_v = h0[:, 0].detach()  # [S, H]
        _vrows = int(os.environ.get("GLM53_VERIFIER_ROWS", "128"))
        vlp_rows, varg_rows = [], []
        for r0 in range(0, rows_v.shape[0], _vrows):
            vlg = _head_out(model.output_layer, rows_v[r0:r0 + _vrows])
            vlg = _vocab_gather(vlg) if tp > 1 else vlg
            vlg = vlg.float()
            varg_rows.append(vlg.argmax(dim=-1).detach())
            vlp_rows.append(torch.log_softmax(vlg, dim=-1).to(torch.bfloat16))
            del vlg
        vargmax = torch.cat(varg_rows, dim=0).detach()
        vlp = torch.cat(vlp_rows, dim=0).detach()

    embed = mtp.mtp_embed
    rot = mtp.mtp_rot
    tl = layer.mtp_model_layer  # MultiTokenPredictionLayer wraps the block here
    head = model.mtp_head

    def chain_step(prev_norm, hd_stack, ids_flat, label_rows, c0, k, s_k):
        _dbg = os.environ.get("V8_DEBUG_SHAPES") == "1"
        prev = rot(prev_norm[:s_k])[0]
        emb = embed(ids_flat[c0 + k:c0 + k + s_k])
        if _dbg:
            print(f"[chain] c0={c0} k={k} s_k={s_k} prev={tuple(prev.shape)} "
                  f"emb={tuple(emb.shape)} hnorm={tuple(layer.hnorm(prev).shape)} "
                  f"enorm={tuple(layer.enorm(emb).shape)}", flush=True)
        pos_now = torch.arange(c0, c0 + s_k, device=dev) + k
        emb = torch.where((pos_now == 0).unsqueeze(-1), 0, emb)
        fused = layer.eh_proj(torch.cat([layer.enorm(emb), layer.hnorm(prev)],
                                        dim=-1))
        fused = fused[0] if isinstance(fused, tuple) else fused  # slim (y, bias)
        tp_size, _ = _tp()
        if tp_size > 1:
            from megatron.core import tensor_parallel
            fused = tensor_parallel.gather_from_tensor_model_parallel_region(fused)
        residual = fused
        hd = tl.input_layernorm(fused)
        hd_stack = list(hd_stack) + [hd]
        hd_seq = torch.stack(hd_stack, dim=1)  # [B=s_k, S=k, H] in branch-1 terms
        # LAYOUT TRUTH (rope-probe verified, rel 0.0): the mcore attention is
        # SEQ-FIRST [S, B, H] — hd_seq needs the transpose; position_ids [S, B].
        # The MLA computes rope from rotary_seq_len (sequential 0..k, IGNORING
        # position_ids) — equivalent to the vendor's absolute depth-grid
        # c0+i+j+1 by rope's relative property (q depth k vs keys 0..k-1:
        # relative offsets k-j are identical; the absolute offset drops out).
        pos = (torch.arange(c0, c0 + s_k, device=dev).unsqueeze(1)
               + torch.arange(1, k + 1, device=dev).unsqueeze(0))  # [s_k, k]
        attn_out = tl.self_attention(hidden_states=hd_seq.transpose(0, 1),
                                     attention_mask=None,
                                     position_ids=pos.transpose(0, 1).contiguous())
        attn_out = attn_out[0] if isinstance(attn_out, tuple) else attn_out
        hd = residual + attn_out[-1]
        residual = hd
        hd = tl.pre_mlp_layernorm(hd)
        hd = tl.mlp(hd)
        hd = hd[0] if isinstance(hd, tuple) else hd
        out_k = residual + hd
        normed_k = layer.final_layernorm(out_k)
        mtp_labels = label_rows[c0 + k + 1:c0 + k + s_k]
        ce, mg = chunked_ce_margin(head, normed_k[:-1], mtp_labels,
                                   chunk=max(1, chunk), want_margin=acc_w > 0,
                                   margin=acc_margin, tp_gather=tp_size > 1)
        cnt = (mtp_labels != IGNORE_INDEX).sum()
        ce_sum = ce * torch.clamp(cnt, min=1).float()
        kl_k = agree_k = None
        if vlp is not None:
            dr = normed_k[:-1]
            vr = vlp[c0 + k:c0 + k + dr.shape[0]]
            if _dbg:
                print(f"[chain] kl rows: dr={tuple(dr.shape)} vr={tuple(vr.shape)} "
                      f"vlp={tuple(vlp.shape)} c0={c0} k={k}", flush=True)
            dlg = _head_out(head, dr)
            dlg = _vocab_gather(dlg) if tp_size > 1 else dlg
            dlg = dlg.float()
            dlp = torch.log_softmax(dlg, dim=-1)
            pv = vr.float().exp()
            kl_k = (pv * (vr.float() - dlp)).sum(dim=-1).mean()
            if aw > 0:
                va = vargmax[c0 + k:c0 + k + dr.shape[0]]
                sup = mtp_labels != IGNORE_INDEX
                agree_k = F.cross_entropy(dlg[sup], va[sup]) if sup.any() else None
        mg_out = mg if acc_w > 0 else ce_sum * 0.0
        return layer.final_layernorm(out_k), ce_sum, cnt, tuple(hd_stack), \
            mg_out, kl_k, agree_k

    loss_rows = []
    _kl_s = torch.zeros((), device=dev, dtype=torch.float32)
    _kl_n = 0
    _ag_s = torch.zeros((), device=dev, dtype=torch.float32)
    _ag_n = 0
    _d_sums, _d_cnts = {}, {}
    decomp = os.environ.get("GLM53_LOSS_DECOMP") == "1" and rank == 0

    i = 0  # B=1 (mcore layout [S, B, H] → the B=0 flow)
    for c0 in range(0, S - 2, chain_chunk):
        prev_norm = h0[c0:, 0]
        hd_stack = ()
        k = 0
        while k < K:
            k += 1
            s_k = min(c0 + chain_chunk, S - k) - c0
            if s_k < 2:
                break
            hd_stack = tuple(t[:s_k] for t in hd_stack)
            (prev_norm, ce_sum_k, cnt_k, hd_stack, mg_k, kl_k, agree_k) = \
                torch.utils.checkpoint.checkpoint(
                    chain_step, prev_norm, hd_stack, ids[i], labels[i],
                    c0, k, s_k, use_reentrant=False)
            if kl_k is not None:
                _kl_s = _kl_s + kl_k
                _kl_n += 1
            if agree_k is not None:
                _ag_s = _ag_s + agree_k
                _ag_n += 1
            if decomp:
                _p = _d_sums.setdefault(k, torch.zeros((), device=dev))
                _d_sums[k] = _p + ce_sum_k.detach()
                _q = _d_cnts.setdefault(k, torch.zeros((), device=dev))
                _d_cnts[k] = _q + cnt_k.detach().float()
            loss_rows.append((ce_sum_k, cnt_k, mg_k, k))

    # aggregate: mean rows CE + margin term (depth-weighted) + KL + agree
    ce_total = sum(r[0] for r in loss_rows)
    cnt_total = sum(r[1] for r in loss_rows)
    out = ce_total / torch.clamp(cnt_total, min=1).float()
    if acc_w > 0:
        acc_term = torch.zeros((), device=dev, dtype=torch.float32)
        by_k = {}
        for ce_s, cnt, mg, k in loss_rows:
            e = by_k.setdefault(k, [torch.zeros((), device=dev), 0.0])
            e[0] = e[0] + mg * torch.clamp(cnt, min=1).float()
            e[1] = e[1] + cnt
        for k, (mg_s, c_s) in by_k.items():
            w = (_acc_ws[k - 1] if (_acc_ws and k - 1 < len(_acc_ws)) else 1.0)
            acc_term = acc_term + acc_w * w * (mg_s / torch.clamp(c_s, min=1).float())
        out = out + acc_term
    if dw > 0 and _kl_n > 0:
        out = out + dw * (_kl_s / _kl_n)
    if aw > 0 and _ag_n > 0:
        out = out + aw * (_ag_s / _ag_n)
    model._last_kl = (float(_kl_s.detach()) / _kl_n) if _kl_n else None
    model._last_agree = (float(_ag_s.detach()) / _ag_n) if _ag_n else None
    if decomp and _d_sums:
        model._last_depth_losses = [float(_d_sums[k].detach()) / max(1.0, float(_d_cnts[k]))
                                    for k in sorted(_d_sums)]
    return out
