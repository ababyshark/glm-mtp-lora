"""V10 DSA (DeepSeek Sparse Attention) layer — GLM Flash variant.

Indexer math ported from the parity-validated mcore reference backend
(scripts/v10/reference/mcore020/dsa.py @ commit 45b68f914, the exact version
the reference project validated tiny/medium parity against), kpool=4
compress path. Attention core is UNABSORBED explicit k/v (HF Glm5NextText
semantics 1:1 — no W_UK absorption; exact for tests, serving-equivalent).

GLM Flash specifics (mapping §3):
- MLA nope-only: qk_rope=0, no q/k norm, no RoPE anywhere, scaling 256^-0.5
- indexer fully frozen (@no_grad forward, requires_grad=False params), no
  aux loss, no Hadamard rotate, no fp8
- kpool=4: pool_keys = softmax(gate+ape over pool slots) weighted k-sum;
  scores = relu(q@pool_k * 128^-0.5) (ReLU AFTER scaling); weights mixed
  across 32 index heads; topk pools expanded to 2048 + causal tail (<=3)
- kv_b_proj [32768,512] = 64 heads x (k_nope 256 | v 256)

TP scheme (vLLM-serving-equivalent MLA TP): q_a/kv_a REPLICATED (lora dims),
q_b/kv_b Column (head-split), o_proj Row, indexer replicated. Correct at any
TP — V8's lora-Column gap (mapping §8.8) avoided by construction.
"""
import math

import torch

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

from v10_modules import GlmRMSNorm, causal_ends_from_mask

# ---------------------------------------------------------------------------
# kpool reference-torch functions (ported from reference/mcore020/dsa.py)
# ---------------------------------------------------------------------------


def _kpool_compress_keys(k, gate_score, ape, pool_size):
    """Softmax-weighted pool keys, fp32 accumulate, bf16 return.

    k/gate_score [tokens, batch, head_dim]; ape [pool_size, head_dim].
    Only complete pools compressed (trailing incomplete pool dropped; its
    tokens re-enter via always_select_tail).
    """
    seqlen, bsz, head_dim = k.shape
    num_pools = seqlen // pool_size
    usable = num_pools * pool_size
    k_p = k[:usable].reshape(num_pools, pool_size, bsz, head_dim)
    if gate_score is not None:
        g = gate_score[:usable].reshape(num_pools, pool_size, bsz, head_dim).float()
    else:
        g = torch.zeros((num_pools, pool_size, bsz, head_dim),
                        dtype=torch.float32, device=k.device)

    ape_f = ape.to(dtype=torch.float32, device=k.device)
    score = g + ape_f.unsqueeze(0).unsqueeze(2)
    score_max = score.max(dim=1, keepdim=True).values
    prob = torch.exp(score - score_max)
    k_f = k_p.float()
    k_pooled = (prob * k_f).sum(dim=1, keepdim=True) / \
        prob.sum(dim=1, keepdim=True).clamp(min=1e-12)
    k_pooled = k_pooled.squeeze(1)                      # [num_pools, batch, head_dim]
    return k_pooled.to(torch.bfloat16)


def _expand_pools_to_tokens(pool_ids, pool_valid, topk_tokens, pool_size):
    """Expand fixed-width pool IDs to token IDs, preserving -1 padding."""
    starts = pool_ids * pool_size
    offsets = torch.arange(pool_size, device=pool_ids.device)
    tokens = starts.unsqueeze(-1) + offsets
    tokens = tokens.masked_fill(~pool_valid.unsqueeze(-1), -1)
    return tokens.reshape(pool_ids.shape[0], topk_tokens).to(torch.int32)


def _append_tail_to_topk(topk_result, seq_lens, pool_size):
    """Append each query's incomplete causal pool (global token coords)."""
    tail_count = seq_lens.to(torch.int32).remainder(pool_size)
    tail_start = seq_lens.to(torch.int32) - tail_count
    offsets = torch.arange(pool_size - 1, device=topk_result.device)
    tail = tail_start[:, None] + offsets
    tail = tail.masked_fill(offsets >= tail_count[:, None], -1).to(topk_result.dtype)
    return torch.cat((topk_result, tail), dim=-1)


def _compute_index_scores(q, weights, k, use_relu=True):
    """Index scoring: q @ k^T -> optional ReLU -> head-weight mix -> head sum.

    q [sq, batch, n_heads, head_dim]; weights [sq, batch, n_heads];
    k [sq_or_pools, batch, head_dim]. Returns fp32 [batch, sq, sk].
    """
    sq, batch, n_heads, head_dim = q.shape
    sk = k.size(0)
    if sq == 0:
        return torch.empty(batch, 0, sk, dtype=torch.float32, device=q.device)
    k_fp32 = k.float()
    scores = torch.einsum('sbhd,tbd->sbht', q.float(), k_fp32)
    if use_relu:
        scores.relu_()
    scores = scores * weights.unsqueeze(-1).float()
    index_scores = scores.sum(dim=2)                     # [sq, batch, sk]
    return index_scores.transpose(0, 1).contiguous()


@torch.no_grad()
def fused_qk_topk_kpool_causal(q, k, weights, index_topk, pool_size,
                               gate_score, ape, causal_ends, use_relu=True):
    """Causal-training kpool topk (GLM Flash: no fp8, no rotate, no packed).

    q [sq, batch, heads, dim]; k [sk, batch, dim]; weights [sq, batch, heads];
    causal_ends [sq] (shared) or [batch, sq] per-item exclusive key bound
    (= position+1 for right-aligned causal; padding-safe callers pass real
    lengths). Returns int32 topk_indices
    [batch, sq, index_topk + pool_size - 1] with -1 padding.
    """
    sq, batch = q.shape[0], q.shape[1]
    ce = causal_ends.to(device=q.device, dtype=torch.long)
    if ce.dim() == 1:
        ce = ce.unsqueeze(0)
    if ce.shape[0] == 1 and batch > 1:
        ce = ce.expand(batch, -1)                      # [B, sq] (mask [1,1,S,S])
    if ce.shape[0] != batch:
        raise ValueError(f"causal_ends batch {ce.shape[0]} != hidden batch {batch}")
    sk = k.size(0)
    num_pools = sk // pool_size
    k_pooled = _kpool_compress_keys(k, gate_score, ape, pool_size)
    pool_token_base = torch.arange(num_pools, device=k.device,
                                   dtype=torch.int64) * pool_size

    # A pool is causal for query t iff its LAST token < causal_ends[t].
    pool_last = pool_token_base + (pool_size - 1)          # [num_pools]
    index_scores = _compute_index_scores(q, weights, k_pooled, use_relu=use_relu)
    if num_pools > 0:
        # [B, sq, num_pools]; mask pools whose last token >= query bound
        pool_valid = pool_last.view(1, 1, -1) < ce.view(batch, sq, 1)
        index_scores = index_scores.masked_fill(~pool_valid, float("-inf"))

        budget = index_topk // pool_size
        select_k = min(budget, num_pools)
        topk_scores, pool_topk = index_scores.topk(select_k, dim=-1)
        pool_topk = pool_topk.masked_fill(topk_scores == float("-inf"), -1)
        if pool_topk.shape[-1] < budget:
            pad = torch.full(index_scores.shape[:-1] + (budget - pool_topk.shape[-1],),
                             -1, dtype=torch.int64, device=index_scores.device)
            pool_topk = torch.cat([pool_topk, pad], dim=-1)
    else:
        pool_topk = torch.full(index_scores.shape[:-1] + (index_topk // pool_size,),
                               -1, dtype=torch.int64, device=index_scores.device)

    rows = pool_topk.shape[0] * pool_topk.shape[1]
    pool_flat = pool_topk.reshape(rows, -1)
    pool_valid = pool_flat >= 0
    token_topk = _expand_pools_to_tokens(pool_flat.clamp(min=0), pool_valid,
                                         index_topk, pool_size)
    token_topk = token_topk.reshape(pool_topk.shape[0], pool_topk.shape[1],
                                    index_topk)

    # always_select_tail: append the incomplete causal pool per query.
    # lengths must be row-aligned with token_topk (batch-major rows: b*sq).
    lengths = ce.clamp(max=sk).reshape(-1)
    token_topk = _append_tail_to_topk(token_topk.reshape(rows, -1),
                                      lengths, pool_size)
    return token_topk.reshape(pool_topk.shape[0], pool_topk.shape[1], -1)


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------


class GlmLayerNorm(torch.nn.Module):
    """DSAIndexer.k_norm — LayerNorm(128, eps=1e-6) WITH bias (probe: the
    checkpoint carries k_norm.weight AND k_norm.bias)."""

    def __init__(self, dim=128, eps=1e-6, **kwargs):
        super().__init__()
        self.norm = torch.nn.LayerNorm(dim, eps=eps)

    def forward(self, x):
        return self.norm(x)


class V10DSAIndexer(MegatronModule):
    """GLM Flash DSA indexer — fully frozen (mapping §3.2: @no_grad, no aux
    loss). Replicated at TP (inputs are lora-replicated by construction).
    Param names follow the mcore reference: bare compress_gate/ape (no
    .weight suffix — vLLM-aligned), k_norm as GlmLayerNorm (state_dict keys
    k_norm.norm.{weight,bias} — fill maps the checkpoint's k_norm.{weight,bias}).
    """

    def __init__(self, config, submodules=None, layer_number=None, **kwargs):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        hidden = config.hidden_size
        q_lora = config.q_lora_rank
        n_heads = getattr(config, "dsa_indexer_n_heads", 32)
        head_dim = getattr(config, "dsa_indexer_head_dim", 128)
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.index_topk = getattr(config, "dsa_indexer_topk", 2048)
        self.pool_size = getattr(config, "dsa_indexer_kpool", 4)
        self.always_select_tail = getattr(config, "dsa_indexer_always_select_tail", True)

        pd = config.params_dtype
        # duplicated linears (mcore build kwargs-compatible; (out, bias) contract
        # at use sites). Meta-build safe: NO explicit init here — the ambient
        # meta context makes params meta skeletons; fill provides total coverage
        # (V8 lesson: init on meta is a no-op/failure, values come from fill).
        self.linear_wq_b = torch.nn.Linear(q_lora, n_heads * head_dim,
                                           bias=False, dtype=pd)    # [4096,1536]
        self.linear_wk = torch.nn.Linear(hidden, head_dim,
                                         bias=False, dtype=pd)      # [128,4096]
        self.linear_weights_proj = torch.nn.Linear(hidden, n_heads,
                                                   bias=False, dtype=pd)  # [32,4096]
        self.k_norm = GlmLayerNorm(head_dim, eps=1e-6)
        # bare params (no .weight suffix), frozen, bf16 per checkpoint probe
        self.index_kpool_compress_gate = torch.nn.Parameter(
            torch.ones(head_dim, hidden, dtype=pd), requires_grad=False)
        self.index_kpool_compress_ape = torch.nn.Parameter(
            torch.zeros(self.pool_size, head_dim, dtype=torch.float32),
            requires_grad=False)
        for m in (self.linear_wq_b, self.linear_wk, self.linear_weights_proj,
                  self.k_norm):
            for p in m.parameters():
                p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x, q_resid, causal_ends):
        """x [sq,b,hidden] (post input_layernorm hidden); q_resid [sq,b,q_lora]
        (q_a_layernorm output); causal_ends [sq] or [batch, sq] per-query
        exclusive bound. Returns int32 topk indices
        [batch, sq, topk + pool_size - 1]."""
        sq, bsz, _ = x.shape
        q_idx = self.linear_wq_b(q_resid).view(sq, bsz, self.n_heads, self.head_dim)
        k_idx = self.k_norm(self.linear_wk(x)).view(sq, bsz, self.head_dim)
        gate = x @ self.index_kpool_compress_gate.t()          # [sq,b,128]
        weights = (x @ self.linear_weights_proj.weight.t().to(x.dtype)) \
            * (self.n_heads ** -0.5)                           # [sq,b,32]
        topk = fused_qk_topk_kpool_causal(
            q_idx, k_idx, weights, self.index_topk, self.pool_size,
            gate_score=gate, ape=self.index_kpool_compress_ape,
            causal_ends=causal_ends, use_relu=True)
        return topk

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, **kwargs):
        metadata = metadata or {}
        sd = {}
        direct = {"index_kpool_compress_gate": self.index_kpool_compress_gate,
                  "index_kpool_compress_ape": self.index_kpool_compress_ape}
        sd.update(make_sharded_tensors_for_checkpoint(
            direct, prefix, sharded_offsets=sharded_offsets))
        for name, module in self._modules.items():
            if hasattr(module, "sharded_state_dict"):
                sd.update(module.sharded_state_dict(
                    prefix=f"{prefix}{name}.", sharded_offsets=sharded_offsets))
            elif isinstance(module, torch.nn.Linear):
                sd.update(make_sharded_tensors_for_checkpoint(
                    {"weight": module.weight}, f"{prefix}{name}.",
                    sharded_offsets=sharded_offsets))
        return sd


class V10DSAAttention(MegatronModule):
    """GLM Flash DSA attention (MLA nope-only, unabsorbed explicit k/v).

    Linears: q_a/kv_a_with_mqa REPLICATED (lora dims — vLLM TP scheme),
    q_b/kv_b Column (head-split), o_proj Row. q_layernorm/kv_layernorm
    eps=rms_norm_eps (1e-5). No rope, no q/k norm.
    """

    def __init__(self, config, submodules=None, layer_number=None,
                 cp_comm_type=None, **kwargs):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        hidden = config.hidden_size
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.num_heads = config.num_attention_heads
        self.qk_nope = getattr(config, "qk_head_dim", 256)  # nope only (rope=0)
        self.v_head_dim = config.v_head_dim
        tp = config.tensor_model_parallel_size
        self.tp_heads = self.num_heads // tp
        self.softmax_scale = self.qk_nope ** -0.5

        pd = config.params_dtype
        self.indexer = V10DSAIndexer(config, layer_number=layer_number)

        # replicated lora projections (plain nn.Linear on params_dtype)
        self.q_a_proj = torch.nn.Linear(hidden, self.q_lora_rank,
                                        bias=False, dtype=pd)
        self.kv_a_proj_with_mqa = torch.nn.Linear(hidden, self.kv_lora_rank,
                                                  bias=False, dtype=pd)
        # head-split projections — mcore Column/RowParallelLinear slices by
        # parallel_state TP itself: pass FULL dims (num_heads*...), NOT the
        # per-rank size (double-split probe: per-rank kv_b came out (128,512)
        # = 2048/16). tp_heads is only for the forward's view math.
        col = tensor_parallel.ColumnParallelLinear
        row = tensor_parallel.RowParallelLinear
        self.q_b_proj = col(self.q_lora_rank, self.num_heads * self.qk_nope,
                            config=config, init_method=config.init_method,
                            bias=False, gather_output=False, skip_bias_add=True)
        self.kv_b_proj = col(self.kv_lora_rank,
                             self.num_heads * (self.qk_nope + self.v_head_dim),
                             config=config, init_method=config.init_method,
                             bias=False, gather_output=False, skip_bias_add=True)
        self.o_proj = row(self.num_heads * self.v_head_dim, hidden, config=config,
                          init_method=config.output_layer_init_method,
                          bias=False, input_is_parallel=True, skip_bias_add=True)
        # eps = rms_norm_eps (V8 lesson: eps by name — q_a/kv_a norms use the
        # model rms_norm_eps=1e-5 in GLM Flash; the 1e-6 transformers-default
        # trap applies to W4A8's glm_moe_dsa, NOT here — mapping §3.1: RMS 1e-5)
        eps = config.layernorm_epsilon
        self.q_a_layernorm = GlmRMSNorm(hidden_size=self.q_lora_rank, eps=eps)
        self.kv_a_layernorm = GlmRMSNorm(hidden_size=self.kv_lora_rank, eps=eps)

    def forward(self, hidden_states, attention_mask=None, causal_ends=None, **kwargs):
        """Glm5NextLayer self-attention contract: [S,B,H] -> (output, bias).

        causal_ends [sq] or [B, sq] long: per-query exclusive key bound
        (= position+1 for right-aligned causal batches). When None it is
        derived from attention_mask (prefix-validated — left/middle padding
        is unsupported stack-wide, see v10_modules.causal_ends_from_mask),
        else arange(1..S+1).
        """
        sq, bsz, _ = hidden_states.shape
        x = hidden_states                                     # [S,B,H] seq-first

        q_resid = self.q_a_layernorm(self.q_a_proj(x))        # [S,B,1536]
        q = self.q_b_proj(q_resid)[0]                         # [S,B,tp*256]
        q = q.view(sq, bsz, self.tp_heads, self.qk_nope)

        kv_lat = self.kv_a_layernorm(self.kv_a_proj_with_mqa(x))  # [S,B,512]
        kv = self.kv_b_proj(kv_lat)[0]                        # [S,B,tp*(256+256)]
        kv = kv.view(sq, bsz, self.tp_heads, self.qk_nope + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope, self.v_head_dim], dim=-1)

        if causal_ends is None:
            if attention_mask is not None:
                causal_ends = causal_ends_from_mask(attention_mask, sq)
            else:
                causal_ends = torch.arange(1, sq + 1, device=x.device)
        topk = self.indexer(x, q_resid, causal_ends)          # [B,S,topk+3] int32

        # sparse attention over selected keys (fp32 softmax; NPU bool-mask rule
        # applies to masked_fill)
        sk = k_nope.shape[0]
        neg = torch.full((bsz, sq, sk), float("-inf"), dtype=torch.float32,
                         device=x.device)
        # Scatter 0.0 ONLY (never -inf): invalid slots are all -1 -> clamp to 0,
        # and position 0 is causally valid for EVERY query (either via pool 0
        # or the always_select_tail), so granting it is a no-op. A mixed
        # 0.0/-inf scatter at duplicate indices is a nondeterministic race —
        # it left all--inf rows -> softmax NaN (probe 2026-11-19).
        idx = topk.clamp(min=0).to(torch.long)
        neg.scatter_(2, idx, 0.0)

        # [S,B,h,d] -> [B,h,S,d]
        qh = q.permute(1, 2, 0, 3).float()
        kh = k_nope.permute(1, 2, 0, 3).float()
        scores = torch.matmul(qh, kh.transpose(-1, -2)) * self.softmax_scale
        scores = scores + neg.unsqueeze(1)                    # [B,h,S,sk]
        probs = torch.softmax(scores, dim=-1).to(v.dtype)
        out = torch.matmul(probs, v.permute(1, 2, 0, 3))      # [B,h,S,vd]
        out = out.permute(2, 0, 1, 3).reshape(sq, bsz, -1).contiguous()
        out, bias = self.o_proj(out)
        return out, bias

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, **kwargs):
        metadata = metadata or {}
        sd = {}
        for name, module in self._modules.items():
            if hasattr(module, "sharded_state_dict"):
                sd.update(module.sharded_state_dict(
                    prefix=f"{prefix}{name}.", sharded_offsets=sharded_offsets,
                    metadata=metadata))
            elif isinstance(module, torch.nn.Linear):
                sd.update(make_sharded_tensors_for_checkpoint(
                    {"weight": module.weight}, f"{prefix}{name}.",
                    sharded_offsets=sharded_offsets))
        return sd
