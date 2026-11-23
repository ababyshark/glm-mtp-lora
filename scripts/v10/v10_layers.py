"""V10 Glm5NextLayer — one GLM decoder layer = mHC(attn segment) + mHC(ffn
segment), full custom forward (HF modeling_glm5_next MG:1293-1327 semantics,
parity-validated by the reference project).

Layer I/O: streams [S, B, n*H] between layers (n=4, bf16 throughout the
stream math); the FIRST layer expands the single post-embedding stream
(no-param copy); the LAST layer contracts via unconditional mean (HF
Glm5NextTextHyperHead). MTP layer has NO hc (checkpoint: no hc_* keys) and
is a plain layer — not this module.

Residual semantics: carried by the HC comb streams (NOT a standard
post-layer residual); sublayer outputs are net outputs by construction
(no BDA bypass trick needed — the 0.20-wrapper y_net−collapsed hack is
avoided because this module never uses mcore's TransformerLayer BDA).

Per-segment math (mapping §2.2/2.3):
  pre, post, comb = hc(streams)                  # fp32 gates
  collapsed  = Σ_h pre[h]·streams[h]             # [b,s,h] (cast bf16)
  y          = self_attn(input_layernorm(collapsed))   (or mlp(pre_mlp_ln(...)))
  streams    = post·y + comb^T @ streams         # post/comb cast bf16 first
"""
import torch

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import build_module

from v10_modules import GlmHyperConnectionModule


class Glm5NextLayerSubmodules:
    def __init__(self, input_layernorm=None, self_attention=None,
                 pre_mlp_layernorm=None, mlp=None):
        self.input_layernorm = input_layernorm
        self.self_attention = self_attention
        self.pre_mlp_layernorm = pre_mlp_layernorm
        self.mlp = mlp


class Glm5NextLayer(MegatronModule):
    """GLM Flash decoder layer (mHC dual-gate + attention + mlp)."""

    def __init__(self, config, submodules: Glm5NextLayerSubmodules,
                 layer_number=None, is_first=False, is_last=False, **kwargs):
        super().__init__(config)
        self.config = config
        self.layer_number = layer_number
        self.is_first = is_first
        self.is_last = is_last
        self.n_streams = getattr(config, "glm_mhc_streams", 4)
        self.hidden_size = config.hidden_size

        self.attn_hc = GlmHyperConnectionModule(config)
        self.ffn_hc = GlmHyperConnectionModule(config)
        self.input_layernorm = build_module(
            submodules.input_layernorm, config=config,
            hidden_size=config.hidden_size, eps=config.layernorm_epsilon)
        self.self_attention = build_module(
            submodules.self_attention, config=config, layer_number=layer_number)
        self.pre_mlp_layernorm = build_module(
            submodules.pre_mlp_layernorm, config=config,
            hidden_size=config.hidden_size, eps=config.layernorm_epsilon)
        # 0.18.2 get_moe_module_spec returns functools.partial(MoELayer,
        # submodules=...) (venv snapshot), NOT a ModuleSpec — build_module
        # cannot handle a partial; call it directly. ModuleSpec (dense MLP)
        # goes through build_module normally.
        import functools
        from megatron.core.transformer.spec_utils import ModuleSpec as _MS
        if isinstance(submodules.mlp, _MS):
            self.mlp = build_module(submodules.mlp, config=config)
        elif isinstance(submodules.mlp, functools.partial):
            self.mlp = submodules.mlp(config=config)
        else:  # plain class spec
            self.mlp = build_module(submodules.mlp, config=config)
        if hasattr(self.mlp, "set_layer_number"):
            self.mlp.set_layer_number(layer_number)

    # ------------------------------------------------------------------
    def _expand(self, hidden_states):
        """[s,b,H] single stream -> [b,s,n,h] (pure copy)."""
        s, b, h = hidden_states.shape
        return hidden_states.view(s, b, 1, h).expand(
            s, b, self.n_streams, h).permute(1, 0, 2, 3).contiguous()

    def _segment(self, streams, hc, norm, sublayer, attention_mask=None,
                 pass_mask=False):
        """One mHC segment: collapse -> norm -> sublayer -> expand-back."""
        pre, post, comb = hc.compute_gates(streams)            # fp32 gates
        collapsed = (pre.unsqueeze(-1) * streams).sum(dim=2).to(streams.dtype)
        y_sbH = collapsed.transpose(0, 1).contiguous()         # [s,b,h]
        y_in = norm(y_sbH)
        if pass_mask:
            out, _ = sublayer(y_in, attention_mask=attention_mask)
        else:
            out, _ = sublayer(y_in)                            # MLP: no mask arg
        y_bsh = out.transpose(0, 1)                            # [b,s,h] net output
        post = post.to(streams.dtype)
        comb = comb.to(streams.dtype)
        new_streams = post.unsqueeze(-1) * y_bsh.unsqueeze(2) \
            + torch.matmul(comb.transpose(-1, -2), streams)    # [b,s,n,h]
        return new_streams

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        """[S,B,H] (first) or [S,B,n*H] -> (streams out, bias=None)."""
        s, b, nh = hidden_states.shape
        if nh == self.hidden_size:
            streams = self._expand(hidden_states)              # first layer
        else:
            streams = hidden_states.view(s, b, self.n_streams,
                                         self.hidden_size).permute(1, 0, 2, 3) \
                .contiguous()

        streams = self._segment(streams, self.attn_hc, self.input_layernorm,
                                self.self_attention, attention_mask,
                                pass_mask=True)
        streams = self._segment(streams, self.ffn_hc, self.pre_mlp_layernorm,
                                self.mlp)

        if self.is_last:
            out = GlmHyperConnectionModule.mean_contract(streams)  # [b,s,h]
            return out.transpose(0, 1).contiguous(), None
        return streams.permute(1, 0, 2, 3).contiguous().view(
            s, b, self.n_streams * self.hidden_size), None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None, **kwargs):
        metadata = metadata or {}
        sd = {}
        for hc_name, hc in (("attn_hc", self.attn_hc), ("ffn_hc", self.ffn_hc)):
            sd.update(make_hc_sharded(hc, f"{prefix}{hc_name}.",
                                      sharded_offsets))
        for name, module in self._modules.items():
            if name in ("attn_hc", "ffn_hc"):
                continue
            if hasattr(module, "sharded_state_dict"):
                sd.update(module.sharded_state_dict(
                    prefix=f"{prefix}{name}.", sharded_offsets=sharded_offsets,
                    metadata=metadata))
        return sd


def make_hc_sharded(hc, prefix, sharded_offsets):
    from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint
    return make_sharded_tensors_for_checkpoint(
        {"fn": hc.fn, "base": hc.base, "scale": hc.scale},
        prefix, sharded_offsets=sharded_offsets)
