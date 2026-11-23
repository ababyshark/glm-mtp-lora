"""chunked_ce_margin — the shared V5.4a asset, VERBATIM from the V8 line's
scripts/v8/v8_recipe.py (V10 self-containment: the V8 dir is not vendored
here; only this dependency surface was carried over — the W4A8
serving_chain_recipe_loss geometry was NOT merged, per the do-not-merge rule).

Geometry-independent: head + normed rows + targets.
"""
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
