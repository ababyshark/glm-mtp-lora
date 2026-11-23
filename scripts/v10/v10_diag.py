"""V10 diagnostics primitives — the gate-proven measurement path as reusable
code, plus change-detection utilities with known invariances.

Why this exists: on the same text+weights, the gate's manual path measured
main CE=0.886 while mcore's internal vocab-parallel CE printed 17.09
(docs/V10_MTP_MECHANISM.md §二). Until that is adjudicated, every loss NUMBER
V10 reports must come from THIS file's path (all_gather + fp32 softmax),
never from a partial-vocab softmax.

Invariance lesson baked in: tensor-similarity checks use max-abs elementwise
diff — a plain sum is invariant to half-swaps (the first eh-probe's toggle
check was a no-op test because of exactly this).
"""
import torch
import torch.distributed as dist


def world_size(group=None):
    return dist.get_world_size(group) if dist.is_initialized() else 1


def gather_full_logits(logits_sharded, dim=-1, group=None):
    """All-gather a vocab-sharded tensor along `dim` and return fp32 full.
    METRICS ONLY — non-differentiable plain all_gather; NEVER use on a
    grad-carrying tensor (severs the gradient — see error-ledger #18).

    logits_sharded: [S, B, V/tp] (output_layer with parallel_output=True).
    Gate-proven assumption (launch_ce_gate.py): ColumnParallelLinear rows are
    CONTIGUOUS vocab ranges per rank — cat along dim restores vocab order.
    """
    t = logits_sharded.detach().contiguous().float()
    w = world_size(group)
    if w == 1:
        return t
    parts = [torch.empty_like(t) for _ in range(w)]
    dist.all_gather(parts, t, group=group)
    return torch.cat(parts, dim=dim)


def full_vocab_ce(logits_full, targets):
    """Per-position CE. logits_full [S, V] fp32, targets [S] long → [S]."""
    return -(torch.log_softmax(logits_full, dim=-1)
             .gather(-1, targets.unsqueeze(-1)).squeeze(-1))


def _ce_chunk(ls, tg):
    """One chunk: differentiable TP-gather + fp32 softmax + target gather.
    (A torch-native all_gather swap was attempted on an UNPROVEN hypothesis
    and reverted per root-cause discipline — 2026-11-20.)"""
    import torch.distributed as dist
    if dist.is_initialized():
        from megatron.core import tensor_parallel
        full = tensor_parallel.gather_from_tensor_model_parallel_region(
            ls.contiguous()).float()
    else:                                   # world=1 / no dist: logits are full
        full = ls.contiguous().float()
    logp = torch.log_softmax(full, dim=-1)
    return -logp.gather(-1, tg.unsqueeze(-1)).squeeze(-1)


def chunked_full_vocab_ce(logits_sharded, targets, chunk=128):
    """DIFFERENTIABLE full-vocab CE without retaining the gathered fp32
    buffers — per-chunk torch.utils.checkpoint recomputes gather+softmax in
    backward, so the resident cost is only the sharded logits themselves.

    The internal vocab_parallel_cross_entropy is blacklisted on the .40 stack
    (error-ledger #10) — every TRAINING loss number goes through THIS path.
    NPU caveat: do NOT wrap the chunks in torch.utils.checkpoint — the
    checkpoint-recompute of the HCCL gather poisons grads (probe #nan).
    logits_sharded [S, B, V/tp] (grad ✓), targets [S, B] long → scalar mean.
    """
    S = logits_sharded.shape[0]
    total = None
    n = 0
    for s0 in range(0, S, chunk):
        ls = logits_sharded[s0:s0 + chunk]
        tg = targets[s0:s0 + chunk]
        # NO torch.utils.checkpoint here: the NPU checkpoint-recompute of the
        # HCCL gather produced NON-FINITE grads (probe 2026-11-20: 510/1117
        # tensors poisoned at step 1, loss=nan from step 2). Plain per-chunk
        # autograd — retained buffers ~1.9GB/CE at B=1, fits.
        ce = _ce_chunk(ls, tg)
        total = ce.sum() if total is None else total + ce.sum()
        n += ce.numel()
    return total / max(n, 1)


def _kl_chunk(m_sharded, d_sharded):
    """One chunk: differentiable TP-gather + fp32 KL(main‖draft) per row."""
    import torch.distributed as dist
    if dist.is_initialized():
        from megatron.core import tensor_parallel
        m = tensor_parallel.gather_from_tensor_model_parallel_region(
            m_sharded.contiguous()).float()
        d = tensor_parallel.gather_from_tensor_model_parallel_region(
            d_sharded.contiguous()).float()
    else:                                   # world=1 / no dist: logits are full
        m = m_sharded.contiguous().float()
        d = d_sharded.contiguous().float()
    lm = torch.log_softmax(m, dim=-1)
    ld = torch.log_softmax(d, dim=-1)
    return (lm.exp() * (lm - ld)).sum(dim=-1)   # [c, B]


def chunked_vocab_kl(main_sharded, draft_sharded, chunk=128):
    """DIFFERENTIABLE KL(main‖draft) over the full vocab, chunked over
    positions — the serving-verdict fix (V10_TRAIN §11.4): the corpus-CE-only
    recipe pulled the draft OFF the target model's distribution (serving K̄
    base 3.52 → cp500 2.49, −29%). main is the FIXED reference — the caller
    must pass it detached (trains the draft only). Same chunk pattern as
    chunked_full_vocab_ce (NO torch.utils.checkpoint: the NPU HCCL-gather
    recompute poisons grads, probe 2026-11-20).
    main_sharded / draft_sharded [S, B, V/tp] (draft carries grad) → scalar.
    """
    S = main_sharded.shape[0]
    total = None
    n = 0
    for s0 in range(0, S, chunk):
        kl = _kl_chunk(main_sharded[s0:s0 + chunk],
                       draft_sharded[s0:s0 + chunk])
        total = kl.sum() if total is None else total + kl.sum()
        n += kl.numel()
    return total / max(n, 1)


def chunked_agree_margin(draft_sharded, va, margin=0.5, chunk=128):
    """Argmax-shaping terms (V10.2 expert patch, 2026-11-23):
      agree-CE = CE(draft, va)     — chase the DETACHED main-model argmax
      margin   = relu(m − (draft[va] − runner-up_to_va))
    BOTH anchored on va — the V8 gold-anchored margin FOUGHT the agree-CE
    whenever main_argmax != gold (expert consult 2026-11-23: the two terms
    pulled the draft toward different tokens). draft_sharded [S, B, V/tp]
    (grad ✓); va [S, B] long (detached). Returns (agree_ce, margin_mean)."""
    import torch.nn.functional as F
    S = draft_sharded.shape[0]
    ce_sum = None
    cnt = 0
    mg_sum = None
    mg_n = 0
    for s0 in range(0, S, chunk):
        d = draft_sharded[s0:s0 + chunk]
        if dist.is_initialized():
            from megatron.core import tensor_parallel
            full = tensor_parallel.gather_from_tensor_model_parallel_region(
                d.contiguous()).float()
        else:
            full = d.contiguous().float()               # [c, B, V]
        c, B_, V = full.shape
        flat = full.reshape(-1, V)
        vf = va[s0:s0 + chunk].reshape(-1)
        ce_sum = (ce_sum if ce_sum is not None else 0.0) + \
            F.cross_entropy(flat, vf, reduction="sum")
        cnt += flat.shape[0]
        t_logit = flat.gather(1, vf.unsqueeze(1)).squeeze(1)
        top2 = flat.topk(2, dim=1)
        other = torch.where(top2.indices[:, 0] == vf,
                            top2.values[:, 1], top2.values[:, 0])
        mg = F.relu(margin - (t_logit - other))
        mg_sum = mg.sum() if mg_sum is None else mg_sum + mg.sum()
        mg_n += flat.shape[0]
    agree = ce_sum / max(cnt, 1)
    marg = mg_sum / max(mg_n, 1)
    return agree, marg


def lm_head_ce(model, hidden, targets):
    """Full measurement path: lm_head(hidden [S,B,H]) → gather → CE [S]."""
    lg, _ = model.output_layer(hidden)
    lg = lg[0] if isinstance(lg, tuple) else lg
    if lg.dim() == 3:
        lg = lg[:, 0]                       # [S, B, V/tp] → [S, V/tp] (B=1)
    full = gather_full_logits(lg, dim=-1)
    return full_vocab_ce(full, targets)


def sig(t):
    """(abs-sum, max-abs, shape) — informative only, NEVER a change test."""
    f = t.detach().float()
    return (float(f.abs().sum()), float(f.abs().max()), tuple(t.shape))


def max_abs_diff(a, b):
    """Exact change detector (elementwise; no float-sum aliasing)."""
    if a.shape != b.shape:
        return float("inf")
    return float((a.detach().float() - b.detach().float()).abs().max())


def rolled_targets(ids):
    """Draft targets: process_mtp_loss rolls labels -1 → position t's draft
    logits pair with token t+1. For a manual [S-1]-position CE (excluding the
    rolled-in tail) that equals CE(logits[t], ids[t+1])."""
    return ids[0, 1:]


def chunked_topk_kl(main_sharded, draft_sharded, topk=128, chunk=256):
    """TOP-K truncated forward KL(main‖draft) — the V10.2 expert patch
    (2026-11-23): the full-vocab KL spent gradient+time on the 1e-5-prob
    long tail; restricting to the MAIN's top-K tokens focuses the single
    MTP layer on the head competition AND kills the per-chunk full-vocab
    draft softmax (the 110s/step cost driver).
    main_sharded: DETACHED [S, B, V/tp] (reference only — no grad needed);
    draft_sharded: [S, B, V/tp] grad ✓. The KL is computed over the main's
    top-K support with BOTH distributions renormalized on that support.
    Returns scalar (mean over positions/batch)."""
    import torch.distributed as dist
    S = main_sharded.shape[0]
    total = None
    n = 0
    for s0 in range(0, S, chunk):
        m = main_sharded[s0:s0 + chunk]
        d = draft_sharded[s0:s0 + chunk]
        if dist.is_initialized():
            from megatron.core import tensor_parallel
            mf = tensor_parallel.gather_from_tensor_model_parallel_region(
                m.contiguous()).float()                      # [c, B, V] (no grad)
            df = tensor_parallel.gather_from_tensor_model_parallel_region(
                d.contiguous()).float()                      # [c, B, V] (grad ✓)
        else:
            mf = m.contiguous().float()
            df = d.contiguous().float()
        c, B, V = mf.shape
        mflat = mf.reshape(-1, V)
        dflat = df.reshape(-1, V)
        topv, topi = mflat.topk(topk, dim=1)                 # [N, K]
        m_ren = torch.log_softmax(topv, dim=1)               # main renormalized
        d_g = dflat.gather(1, topi)                          # draft at main's top-K
        d_ren = torch.log_softmax(d_g, dim=1)                # renormalized
        kl = (m_ren.exp() * (m_ren - d_ren)).sum(dim=1)      # [N]
        total = kl.sum() if total is None else total + kl.sum()
        n += kl.numel()
    return total / max(n, 1)
