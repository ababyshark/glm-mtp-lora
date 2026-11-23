"""V10 grouped MoE experts — pure-torch pad+bmm implementation.

Replaces mcore's SequentialMLP (the 288-expert python loop, ~24k kernel
launches/step = the CPU-bound root cause, error-ledger #17) with:
  segment-pad -> ONE bmm per fc -> unpad

Why bmm and not torch_npu.npu_grouped_matmul (probed 2026-11-20):
  - the npu op's forward works ([x1] + [w3d] + split_item=2, vllm-ascend's
    exact pattern) but has NO registered backward (x.grad=None) — inference-only
  - the dw backward would need the separated-x form, rejected by the kernel
    (aclnnGroupedMatmulV5 161002, per-group varying contraction dims)
  - torch.bmm is natively differentiable (dx/dw automatic) — no custom
    autograd Function, no op-API risk, and testable on CPU

Interface mirrors SequentialMLP exactly (drop-in via the module spec):
  forward(permuted_local_hidden_states [T,H], tokens_per_expert [E],
          permuted_probs [T]) -> (out [T,H], None)
The tokens arrive SORTED by expert (the mcore dispatcher contract); the
routing probs are applied to the activation output (mcore MLP semantics).

Weights (per-rank TP shards, by construction — the fill writes the slices):
  w1 [E, H, 2*f_loc]  ([in,out] transposed storage; gate|up interleaved
                       per rank, the fill's fc1 pattern)
  w2 [E, f_loc, H]    ([in,out] transposed storage)
"""
import os

import torch
import torch.nn as nn

from megatron.core.transformer.module import MegatronModule


def grouped_enabled():
    """Env knob GLM53_V10_GROUPED=1: swap SequentialMLP -> V10GroupedExperts
    in BOTH the main stack and the MTP block (single source of truth read by
    v10_model.build_layer_specs / v10_mtp.build_mtp_block_spec / v10_fill)."""
    return os.environ.get("GLM53_V10_GROUPED", "0") == "1"


class V10GroupedExperts(MegatronModule):
    """Drop-in replacement for SequentialMLP (EP=1, all experts local)."""

    def __init__(self, num_local_experts, config, submodules=None,
                 pg_collection=None, name=None, **kwargs):
        super().__init__(config)
        self.config = config
        self.num_local_experts = num_local_experts
        self.hidden = config.hidden_size
        tp = config.tensor_model_parallel_size
        self.f_loc = config.moe_ffn_hidden_size // tp           # 2048/16=128
        self.fc1_out = 2 * self.f_loc                            # gate|up
        self.clamp_v = config.activation_func_clamp_value        # swiglu_limit=10
        # transposed-storage weights: y = x_pad @ w1  ([E,max_t,H]@[E,H,2f_loc])
        self.w1 = nn.Parameter(torch.empty(num_local_experts, self.hidden,
                                           self.fc1_out,
                                           dtype=config.params_dtype))
        self.w2 = nn.Parameter(torch.empty(num_local_experts, self.f_loc,
                                           self.hidden,
                                           dtype=config.params_dtype))

    def _pad_index(self, tokens_per_expert, total, device):
        """Token-index map [E, max_t] (-1 pad) + the valid mask, pure ops.
        NOTE: the dispatcher's tokens_per_expert may live on CPU — the map
        tensors are built on `device` (the activations' device); max_t comes
        from the CPU tensor (no NPU sync)."""
        tpe = tokens_per_expert.to(device)
        e_ids = torch.repeat_interleave(
            torch.arange(self.num_local_experts, device=device), tpe)
        starts = torch.cumsum(tpe, 0) - tpe
        j_idx = torch.arange(total, device=device) - starts[e_ids]
        max_t = int(tokens_per_expert.max())
        idx_p = torch.full((self.num_local_experts, max_t), -1,
                           device=device, dtype=torch.long)
        idx_p[e_ids, j_idx] = torch.arange(total, device=device)
        return idx_p, idx_p >= 0

    def _grouped_fc(self, x, w, tokens_per_expert):
        """x [T, in] -> per-expert x @ w_e -> [T, out] (pad+bmm+unpad).
        w [E, in, out] (transposed storage, matches w1/w2); bmm over the
        last dim — NO transpose here (a transpose here expects [out,in]
        params and breaks fc2 shapes + the fc1 math when H == fc1_out).
        The pad is processed in CHUNKS along the max_t axis (GLM53_V10_
        GROUPED_CHUNK, default 64): a skewed batch can push one expert's
        token count to ~3k, and the full [E, max_t, in] gather would
        allocate ~12GB — chunking caps the workspace at [E, chunk, in]
        (~265MB) while the unpad uses the MASK-based index (the valid rows
        in the (e, j) row-major order = the original expert-sorted [T]
        order; the clamped index would drag the padding rows back in)."""
        total = x.shape[0]
        idx_p, _ = self._pad_index(tokens_per_expert, total, x.device)
        chunk = int(os.environ.get("GLM53_V10_GROUPED_CHUNK", "64"))
        out = x.new_empty((total, w.shape[-1]))
        for s in range(0, idx_p.shape[1], chunk):
            idx_c = idx_p[:, s:s + chunk]                    # [E, c]
            m_c = idx_c >= 0
            x_c = x[idx_c.clamp(min=0)] * m_c.unsqueeze(-1)  # [E, c, in]
            y_c = torch.bmm(x_c, w)                          # [E, c, out]
            sel = idx_c[m_c]                                 # [n_valid_c]
            out[sel] = y_c.reshape(-1, y_c.shape[-1])[m_c.reshape(-1)]
        return out                                           # [T, out]

    def backward_dw(self):
        """No-op: mcore's MoELayer wgrad-deferral path may call this; the
        pad+bmm path produces dw via native autograd (no backward_dw hook)."""
        pass

    def forward(self, permuted_local_hidden_states, tokens_per_expert,
                permuted_probs=None, **kwargs):
        x = permuted_local_hidden_states                          # [T, H]
        counts = tokens_per_expert
        # fc1 + swiglu with the vendor clamp (gate max-only, up double-sided)
        y = self._grouped_fc(x, self.w1, counts)                  # [T, 2f_loc]
        gate, up = y.chunk(2, dim=-1)
        if self.clamp_v is not None:
            gate = gate.clamp(max=self.clamp_v)
            up = up.clamp(min=-self.clamp_v, max=self.clamp_v)
        inter = torch.nn.functional.silu(gate) * up               # [T, f_loc]
        if permuted_probs is not None:
            inter = inter * permuted_probs.to(inter.dtype).unsqueeze(-1)
        # fc2 — UNREDUCED partial output (per-rank ffn slice): RowParallelLinear
        # for experts sets explicit_expert_comm=True (layers.py:1211,1341) and
        # SKIPS the internal reduce; the dispatcher's token_combine owns the
        # cross-rank reduce-scatter over tp_ep. Reducing here TOO = 16x
        # overshoot -> saturation -> garbage (probed 2026-11-21, grouped
        # baseline CE ~10.8 = uniform).
        out = self._grouped_fc(inter, self.w2, counts)            # [T, H]
        return out, None


def world_size():
    import torch.distributed as dist
    return dist.get_world_size() if dist.is_initialized() else 1
