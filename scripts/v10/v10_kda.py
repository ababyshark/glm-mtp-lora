"""V10 KDA (Kimi Delta Attention) mixer — port of the parity-validated
reference (AIfromScratch/glm53flash-mtp code/glm_kda.py, aligned line-by-line
with HF Glm5NextTextLinearAttention, transformers 5.16.1).

Forward math (mapping §1):
  q,k,v = q/k/v_proj(x)                      [B,S,64*128]
  qkv  = depthwise-causal-conv1d(cat(qkv), k=4, pad=3) + silu   (fp32 conv)
  g    = -5.0 * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))   ∈ (-5,0) per head-channel
  beta = sigmoid(b_proj(x))                  per head
  KDA chunk update (chunk=64, fp32, Householder triangular correction) — differentiable
  out  = o_proj( RMSNormGated(core) * sigmoid(g_b(g_a(x))) )

NOT GatedDeltaNet2 (mcore's GDN math is inequivalent: f/z/b shapes, o_norm
gating, g formula all differ — reference decision, mapping §6).

Checkpoint facts (probe 2026-11-19):
  q/k/v_proj [8192,4096] bf16; conv1d q/k/v [8192,1,4] bf16 (fused to
  [24576,1,4] fp32 param at fill); A_log [64] fp32; dt_bias [8192] fp32;
  b_proj [64,4096]; f_a/g_a [128,4096]; f_b/g_b [8192,128] (all bf16);
  o_norm [128]; o_proj [4096,8192].
"""
import torch
import torch.nn.functional as F

from megatron.core import tensor_parallel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
from megatron.core.utils import make_viewless_tensor

from v10_modules import RMSNormGated, causal_ends_from_mask


class GlmKDASubmodules:
    """KDA mixer submodule declaration (mcore spec compatible)."""

    def __init__(self, linear_q_proj=None, linear_k_proj=None, linear_v_proj=None,
                 linear_b_proj=None, linear_fb_proj=None, linear_gb_proj=None,
                 linear_o_proj=None, out_norm=None):
        self.linear_q_proj = linear_q_proj
        self.linear_k_proj = linear_k_proj
        self.linear_v_proj = linear_v_proj
        self.linear_b_proj = linear_b_proj
        self.linear_fb_proj = linear_fb_proj
        self.linear_gb_proj = linear_gb_proj
        self.linear_o_proj = linear_o_proj
        self.out_norm = out_norm


def _kda_l2norm(x, dim=-1, eps=1e-6):
    # HF/FLA aligned: sqrt + divide, NOT F.normalize
    return x / torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def kda_chunk_forward(query, key, value, g, beta, chunk_size=64):
    """chunk_kimi_delta_attention — line-by-line port from HF modeling_glm5_next
    (pure torch, differentiable).

    Inputs [B,S,H,D] (q/k/v), g [B,S,H,Dk], beta [B,S,H]; output [B,S,H,Dv].
    """
    initial_dtype = query.dtype
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    query = _kda_l2norm(query, dim=-1, eps=1e-6)
    key = _kda_l2norm(key, dim=-1, eps=1e-6)

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    scale = 1 / (query.shape[-1] ** 0.5)
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

    query = F.pad(query, (0, 0, 0, pad_size)) * scale
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    query, key, value, g, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, g, k_beta, v_beta)
    ]

    g = g.cumsum(dim=-2)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                 device=query.device), diagonal=0)
    # NAN ROOT CAUSE (anomaly-traced 2026-11-20): the RAW exponent is POSITIVE
    # in the triu region (cumsum differences up to +320) -> exp = inf -> the
    # forward masks it, but the MUL backward computes grad(0 at masked
    # positions) * inf = NaN, poisoning every LoRA grad through the decoder
    # backward. clamp(max=0) makes the masked operands finite: the forward is
    # bit-identical (masked anyway) and the backward is 0*finite = 0.
    decay_mask = (g.unsqueeze(-2) - g.unsqueeze(-3)).clamp(max=0.0).exp().float()
    attn = -(k_beta.unsqueeze(-2) * key.unsqueeze(-3) * decay_mask).sum(dim=-1) \
        .masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp())

    last_recurrent_state = torch.zeros(
        batch_size, num_heads, k_head_dim, value.shape[-1], dtype=value.dtype,
        device=value.device
    )
    core_attn_out = torch.zeros_like(value)

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool,
                                 device=query.device), diagonal=1)
    for i in range((sequence_length + pad_size) // chunk_size):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        g_i = g[:, :, i]

        attn_inter = (q_i * g_i.exp()) @ last_recurrent_state
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]) \
            .sum(dim=-1).masked_fill(mask, 0)
        v_prime = k_cumdecay[:, :, i] @ last_recurrent_state
        v_new = v_i - v_prime

        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = (
            last_recurrent_state * g_i[:, :, -1].exp().unsqueeze(-1)
            + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new
        )

    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out


def kda_causal_conv1d(x_bsc, weight_dc1, seq_len, kernel_size, activation="silu"):
    """depthwise causal conv1d + silu. x [B, conv_dim, S], weight [conv_dim, 1, k]."""
    out = F.conv1d(x_bsc, weight_dc1, bias=None, padding=kernel_size - 1,
                   groups=x_bsc.shape[1])
    out = out[:, :, :seq_len]
    if activation == "silu":
        out = F.silu(out)
    return out


class GlmKDAAttention(MegatronModule):
    """KDA mixer, slots into Glm5NextLayer.self_attention.

    TP split: q/k/v/b/fb/gb Column(dim0), o Row(dim1), fa/ga duplicated,
    A_log/dt_bias/conv TP dim0 (per-head / per-channel layouts).
    """

    def __init__(self, config, submodules: GlmKDASubmodules = None,
                 layer_number=None, cp_comm_type=None, **kwargs):
        super().__init__(config)
        self.config = config
        self.submodules = submodules or GlmKDASubmodules()
        self.layer_number = layer_number
        self.hidden_size = config.hidden_size
        self.num_heads = getattr(config, "glm_linear_num_heads", 64)
        self.head_dim = getattr(config, "glm_linear_head_dim", 128)
        self.qkv_dim = self.head_dim * self.num_heads
        self.conv_kernel = getattr(config, "glm_linear_conv_kernel_dim", 4)
        self.gate_lower_bound = getattr(config, "glm_gate_lower_bound", -5.0)
        self.activation = getattr(config, "glm_linear_activation", "silu")

        cp_kw = {"cp_comm_type": cp_comm_type} if cp_comm_type is not None else {}
        col = self.submodules.linear_q_proj or tensor_parallel.ColumnParallelLinear
        row = self.submodules.linear_o_proj or tensor_parallel.RowParallelLinear

        self.linear_q_proj = col(self.hidden_size, self.qkv_dim, config=config,
                                 init_method=config.init_method, bias=False,
                                 gather_output=False, skip_bias_add=True, **cp_kw)
        self.linear_k_proj = col(self.hidden_size, self.qkv_dim, config=config,
                                 init_method=config.init_method, bias=False,
                                 gather_output=False, skip_bias_add=True, **cp_kw)
        self.linear_v_proj = col(self.hidden_size, self.qkv_dim, config=config,
                                 init_method=config.init_method, bias=False,
                                 gather_output=False, skip_bias_add=True, **cp_kw)
        self.linear_b_proj = col(self.hidden_size, self.num_heads, config=config,
                                 init_method=config.init_method, bias=False,
                                 gather_output=False, skip_bias_add=True, **cp_kw)
        # decay-gate low-rank bottleneck: fa duplicated (head_dim full), fb Column
        # (plain nn.Linear defaults fp32 — force params_dtype; checkpoint fa/ga BF16)
        self.linear_fa_proj = torch.nn.Linear(self.hidden_size, self.head_dim,
                                              bias=False).to(
            dtype=self.config.params_dtype)
        self.linear_fb_proj = col(self.head_dim, self.qkv_dim, config=config,
                                  init_method=config.init_method, bias=False,
                                  gather_output=False, skip_bias_add=True, **cp_kw)
        # output gate low-rank bottleneck: ga duplicated, gb Column
        self.linear_ga_proj = torch.nn.Linear(self.hidden_size, self.head_dim,
                                              bias=False).to(
            dtype=self.config.params_dtype)
        self.linear_gb_proj = col(self.head_dim, self.qkv_dim, config=config,
                                  init_method=config.init_method, bias=False,
                                  gather_output=False, skip_bias_add=True, **cp_kw)
        self.linear_o_proj = row(self.qkv_dim, self.hidden_size, config=config,
                                 init_method=config.output_layer_init_method,
                                 bias=False, input_is_parallel=True,
                                 skip_bias_add=True, **cp_kw)
        self.o_norm = (self.submodules.out_norm or RMSNormGated)(
            self.head_dim, eps=config.layernorm_epsilon)

        tp = self.config.tensor_model_parallel_size
        self.tp_heads = self.num_heads // tp
        self.tp_qkv = self.qkv_dim // tp

        # fp32 small params (checkpoint: A_log/dt_bias F32; conv bf16 upcast at
        # fill). conv1d_weight holds the FUSED q|k|v conv [3*tp_qkv, 1, k].
        self.A_log = torch.nn.Parameter(torch.zeros(self.tp_heads, dtype=torch.float32))
        self.dt_bias = torch.nn.Parameter(torch.zeros(self.tp_qkv, dtype=torch.float32))
        self.conv1d_weight = torch.nn.Parameter(
            torch.zeros(3 * self.tp_qkv, 1, self.conv_kernel, dtype=torch.float32))

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """Glm5NextLayer self-attention contract: [S,B,H] -> (output [S,B,H], bias=None).

        attention_mask (when provided) is prefix-VALIDATED only — the KDA
        recurrence cannot express left/middle padding (right-aligned batches
        need no mask; see v10_modules.causal_ends_from_mask)."""
        bsz, seq_len = hidden_states.shape[1], hidden_states.shape[0]
        if attention_mask is not None:
            causal_ends_from_mask(attention_mask, seq_len)
        x_bsh = hidden_states.transpose(0, 1).contiguous()          # [B,S,H]

        q, _ = self.linear_q_proj(x_bsh)
        k, _ = self.linear_k_proj(x_bsh)
        v, _ = self.linear_v_proj(x_bsh)
        mixed = torch.cat([q, k, v], dim=-1).float().transpose(1, 2)  # [B,3qkv,S]
        mixed = kda_causal_conv1d(mixed, self.conv1d_weight, seq_len,
                                  self.conv_kernel, self.activation)
        mixed = mixed.transpose(1, 2).to(x_bsh.dtype)                # [B,S,3qkv]
        q, k, v = torch.split(mixed, [self.tp_qkv] * 3, dim=-1)

        shape = (bsz, seq_len, self.tp_heads, self.head_dim)
        query = q.view(shape)
        key = k.view(shape)
        value = v.view(shape)

        # g = -lb * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)) — per head-channel
        f_a = self.linear_fa_proj(x_bsh)
        f_b, _ = self.linear_fb_proj(f_a.to(x_bsh.dtype))
        g_raw = f_b.float() + self.dt_bias.float().view(1, 1, -1)
        g = self.gate_lower_bound * torch.sigmoid(
            self.A_log.float().exp().view(1, 1, self.tp_heads, 1)
            * g_raw.view(bsz, seq_len, self.tp_heads, self.head_dim)
        ).view(bsz, seq_len, -1)                                     # [B,S,tp_qkv]

        beta = torch.sigmoid(self.linear_b_proj(x_bsh)[0])           # [B,S,tp_heads]

        core = kda_chunk_forward(query, key, value,
                                 g=g.view(bsz, seq_len, self.tp_heads, self.head_dim),
                                 beta=beta)                          # [B,S,tp_heads,head_dim]

        gate = self.linear_ga_proj(x_bsh)
        gate, _ = self.linear_gb_proj(gate.to(x_bsh.dtype))
        gate = gate.view(bsz, seq_len, self.tp_heads, self.head_dim)
        out = self.o_norm(core, gate)                                # [B,S,tp_heads,head_dim]
        out = out.reshape(bsz, seq_len, -1)
        out, bias = self.linear_o_proj(out)
        out = make_viewless_tensor(out.transpose(0, 1).contiguous(), True, True)  # [S,B,H]
        return out, None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None,
                           tp_group=None, **kwargs):
        """Linears recurse; A_log/dt_bias/conv1d_weight TP-dim0 ShardedTensor;
        fa/ga (plain nn.Linear, duplicated) must be emitted explicitly or
        dist-ckpt silently drops them (reference session-log bug #3)."""
        metadata = metadata or {}
        sd = {}
        direct = {
            "A_log": self.A_log,
            "dt_bias": self.dt_bias,
            "conv1d_weight": self.conv1d_weight,
        }
        sd.update(make_sharded_tensors_for_checkpoint(
            direct, prefix, tensor_parallel_layers_axis_map={
                "A_log": 0, "dt_bias": 0, "conv1d_weight": 0,
            }, sharded_offsets=sharded_offsets))
        for name, module in self._modules.items():
            if hasattr(module, "sharded_state_dict"):
                sd.update(module.sharded_state_dict(
                    prefix=f"{prefix}{name}.", sharded_offsets=sharded_offsets,
                    metadata=metadata))
            elif isinstance(module, torch.nn.Linear):
                d = {"weight": module.weight}
                if module.bias is not None:
                    d["bias"] = module.bias
                sd.update(make_sharded_tensors_for_checkpoint(
                    d, f"{prefix}{name}.", sharded_offsets=sharded_offsets))
        return sd

    def __repr__(self):
        return (f"GlmKDAAttention(hidden={self.hidden_size}, heads={self.num_heads}, "
                f"head_dim={self.head_dim}, conv_k={self.conv_kernel}, "
                f"gate_lb={self.gate_lower_bound}, layer={self.layer_number})")
