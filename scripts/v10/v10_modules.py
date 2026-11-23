"""V10 norm/mHC modules — ports from the parity-validated reference
(AIfromScratch/glm53flash-mtp code/glm_mhc.py + glm5next_spec.py, tiny/medium
parity rel<=1.7e-06 vs HF Glm5Next transformers 5.16.1).

Semantics (docs reference ARCHITECTURE_MAPPING.md):
- GlmRMSNorm: fp32 rsqrt(mean(x^2)+eps) * weight (HF Glm5NextTextRMSNorm).
- GlmHyperConnectionModule: per-sublayer mHC (n=4 streams):
    flat = RMSNorm_no_weight(streams.flatten(2))   # fp32
    w    = flat @ fn.T                              # [b,s,n^2+2n]
    pre  = sigmoid(w_pre*s + b) + eps               # collapse weights
    post = 2*sigmoid(w_post*s + b)                  # expand weights
    comb = sinkhorn20(softmax(w_comb*s + b))        # stream mixing
  checkpoint dtypes (probe 2026-11-19): fn BF16 [24,16384], base/scale F32.
  The module holds fn in fp32 (fill upcasts bf16 -> fp32, value-preserving);
  all gate math runs fp32 (reference line-identical).
"""
import torch

try:
    from megatron.core.utils import mark_keep_in_fp32
    HAS_MARK_FP32 = True
except ImportError:  # 0.18.2 has no mark_keep_in_fp32
    HAS_MARK_FP32 = False


def _mark_fp32(p):
    if HAS_MARK_FP32:
        mark_keep_in_fp32(p)
    return p


def causal_ends_from_mask(attention_mask, sq):
    """Per-query exclusive key bounds from a bool keep-mask ([B,1,S,S] or
    [B,S,S], True = keep). V10's attention stack is right-aligned-causal ONLY:
    the KDA recurrence and the DSA indexer pool encoding cannot express
    left/middle padding, so every row must be a contiguous prefix (keep set
    exactly [0:end)). Returns ends [B, sq] int64; raises ValueError otherwise
    (fail fast instead of silently corrupting — V8 lesson)."""
    m = attention_mask
    if m.dim() == 4:
        m = m[:, 0]
    if m.dim() != 3 or m.shape[-2] != sq or m.shape[-1] != sq:
        raise ValueError(f"unsupported attention_mask shape "
                         f"{tuple(attention_mask.shape)} for sq={sq} "
                         f"(want [B,1,S,S] or [B,S,S] bool keep-mask)")
    if m.dtype != torch.bool:
        raise ValueError(f"attention_mask must be a bool keep-mask, got {m.dtype}")
    ends = m.sum(dim=-1)                                    # [B, sq]
    ar = torch.arange(sq, device=m.device)
    if not bool((m == (ar.view(1, -1) < ends.unsqueeze(-1))).all()):
        raise ValueError("attention_mask rows are not causal prefixes "
                         "(left/middle padding unsupported: the KDA recurrence "
                         "and DSA indexer assume right-aligned sequences)")
    return ends.to(torch.long)


class GlmRMSNorm(torch.nn.Module):
    """HF Glm5NextTextRMSNorm: fp32 mean/rsqrt, weight-multiply.
    eps comes from the CALLER (norm position determines eps: main norms
    rms_norm_eps=1e-5; q_a/kv_a norms transformers-default 1e-6 — V8 lesson
    'norm eps 按名字分配')."""

    def __init__(self, config=None, hidden_size=None, eps=None, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size or config.hidden_size
        self.eps = eps if eps is not None else (
            config.layernorm_epsilon if config is not None else 1e-5)
        self.weight = torch.nn.Parameter(torch.ones(self.hidden_size))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class RMSNormGated(torch.nn.Module):
    """Glm5NextTextRMSNormGated: fp32 RMSNorm(last D dims) * weight * sigmoid(gate).
    eps = rms_norm_eps (1e-5), NOT the class-default 1e-6 (mapping §1.5)."""

    def __init__(self, dim, eps=1e-5, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x, gate):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * self.weight.float()
        out = x * torch.sigmoid(gate.float())
        return out.to(dtype)


class GlmHyperConnectionModule(torch.nn.Module):
    """Single-sublayer mHC (pure torch; per GLM checkpoint: fn/base/scale).

    Param shapes (n streams, h hidden): fn [(2+n)*n, n*h], base [(2+n)*n],
    scale [3] (pre/post/comb scales). n=4, h=4096 -> fn [24, 16384].
    """

    def __init__(self, config=None, n_streams=None, hidden=None, eps=None,
                 sinkhorn_iters=None, norm_eps=None, **kwargs):
        super().__init__()
        n = n_streams if n_streams is not None else getattr(config, "glm_mhc_streams", 4)
        h = hidden if hidden is not None else config.hidden_size
        self.n = n
        self.hidden = h
        # gate eps = hc_eps (1e-6); the HC INPUT norm eps = layernorm_epsilon
        # (1e-5) — the golden reference hardcodes 1e-5 there (hf_reference.py
        # hf_hyper_connection_gates), NOT hc_eps. Two different eps by design.
        self.eps = eps if eps is not None else getattr(config, "glm_hc_eps", 1e-6)
        self.norm_eps = norm_eps if norm_eps is not None else \
            (getattr(config, "layernorm_epsilon", 1e-5) if config is not None else 1e-5)
        self.iters = sinkhorn_iters if sinkhorn_iters is not None else \
            getattr(config, "glm_mhc_sinkhorn_iters", 20)
        mix = (2 + n) * n
        self.fn = torch.nn.Parameter(torch.zeros(mix, n * h, dtype=torch.float32))
        self.base = torch.nn.Parameter(torch.zeros(mix, dtype=torch.float32))
        self.scale = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
        _mark_fp32(self.fn)
        _mark_fp32(self.base)
        _mark_fp32(self.scale)

    def _rmsnorm_noweight(self, x, eps=None):
        # HC INPUT norm eps = norm_eps (1e-5), NOT the gate eps
        eps = self.norm_eps if eps is None else eps
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)

    def compute_gates(self, streams):
        """streams [b,s,n,h] -> (pre [b,s,n], post [b,s,n], comb [b,s,n,n])."""
        b, s, n, h = streams.shape
        eps = self.eps
        flat = self._rmsnorm_noweight(streams.flatten(2).float())  # [b,s,n*h]
        w = torch.nn.functional.linear(flat, self.fn.float())           # [b,s,n^2+2n]
        pre_w, post_w, comb_w = w.split([n, n, n * n], dim=-1)
        pre_b, post_b, comb_b = self.base.float().split([n, n, n * n], dim=0)
        pre_s, post_s, comb_s = self.scale.float().unbind(0)

        pre = torch.sigmoid(pre_w * pre_s + pre_b) + eps
        post = 2.0 * torch.sigmoid(post_w * post_s + post_b)
        comb = torch.softmax(comb_w.view(b, s, n, n) * comb_s + comb_b.view(n, n),
                             dim=-1) + eps
        # Sinkhorn-Knopp (HF-aligned: column-norm x1 then (row,col) x (iters-1))
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(self.iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb

    @staticmethod
    def expand_streams(x_sbh, n):
        """[s,b,h] -> [s,b,n*h] (pure copy expand, no params; HF unsqueeze.expand)."""
        s, b, h = x_sbh.shape
        return x_sbh.unsqueeze(2).expand(s, b, n, h).contiguous().view(s, b, n * h)

    @staticmethod
    def mean_contract(streams_bsnh):
        """[b,s,n,h] -> [b,s,h] unconditional mean (HF Glm5NextTextHyperHead)."""
        return streams_bsnh.mean(dim=2)
