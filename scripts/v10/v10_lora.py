"""V10 LoRA — V10-owned injection (no PEFT, no V8 reuse).

Design (user decision 2026-11-20, option A): LoRA adapters on the BF16 base
decoder + full MTP training. Base weights stay frozen BF16 (iron rule: BF16
direct storage, zero quantization — "QLoRA" is legacy project naming only).

Injection = forward HOOKS, not module replacement:
  - the base module keeps its name/weight/state_dict → the fill target names
    and the coverage ledger stay byte-identical
  - adapter params are plain attributes (lora_A/lora_B) on the base module →
    named_parameters exposes "...linear_q_proj.lora_A" for the optimizer/CP
  - B is ZERO-initialized → step 0 is EXACTLY the fill-validated model
  - Column targets: adapter added locally (B [out_local, r] per-rank shard)
  - Row targets: adapter partial reduced over TP (A [r, in_local] per-rank,
    B [out, r] replicated) via mcore's differentiable reduce
  - A init: fixed seed, normal(0, 0.02) — deterministic across ranks for
    replicated A (column mode)

Default targets: the big attention projections + dense/shared-expert MLP.
Routed experts are NOT adapted by default (routing stability; adapter count);
indexer/norms/hc/conv1d stay frozen by design.
"""
import os

import torch
import torch.nn as nn

from megatron.core import tensor_parallel

# suffix match on module names inside decoder.layers.*
DEFAULT_TARGETS = (
    "linear_q_proj", "linear_k_proj", "linear_v_proj", "linear_o_proj",
    "q_b_proj", "kv_b_proj", "o_proj",
    "linear_fc1", "linear_fc2",
)


def _is_row(mod):
    from megatron.core.tensor_parallel.layers import RowParallelLinear
    return isinstance(mod, RowParallelLinear)


def _dims(mod):
    """(in_dim_for_A, out_dim_for_B) per mode, from the LOCAL weight shape."""
    w = mod.weight
    if _is_row(mod):
        return w.shape[1], w.shape[0]        # A [r, in_local]; B [out_full, r]
    return w.shape[1], w.shape[0]            # column: A [r, in_full]; B [out_local, r]


def _make_hook(A, B, scale, row_mode):
    def hook(module, inputs, output):
        x = inputs[0]
        if x.dim() == 3 and x.shape[0] != A.shape[-1] and False:
            pass
        upd = (x @ A.t()) @ B.t() * scale     # [S, B, out] (local for column)
        if row_mode and world_size() > 1:
            upd = tensor_parallel.reduce_from_tensor_model_parallel_region(upd)
        out = output[0] if isinstance(output, tuple) else output
        new = out + upd.to(out.dtype)
        return (new, output[1]) if isinstance(output, tuple) else new
    return hook


def world_size():
    import torch.distributed as dist
    return dist.get_world_size() if dist.is_initialized() else 1


def apply_lora(model, r=16, alpha=32, targets=None, seed=20261120):
    """Register LoRA hooks on decoder target linears. Returns handled names.

    Idempotent: modules already carrying lora_A are skipped. The model's
    behavior at injection is EXACTLY unchanged (B == 0)."""
    r = int(os.environ.get("GLM53_V10_LORA_R", r))
    alpha = float(os.environ.get("GLM53_V10_LORA_ALPHA", alpha))
    targets = tuple(os.environ.get("GLM53_V10_LORA_TARGETS", "").split(",")) \
        if os.environ.get("GLM53_V10_LORA_TARGETS") else (targets or DEFAULT_TARGETS)
    scale = alpha / r
    handled = []
    for name, mod in model.named_modules():
        # apply_lora receives the decoder ROOT (trainer) or the full model
        # (tests) — names are RELATIVE to that root: accept both spellings.
        # (probe 2026-11-20: the absolute-only filter silently injected
        # NOTHING on the production path — the unit assertion caught it)
        if not ("decoder.layers." in name or name.startswith("layers.")):
            continue
        if "mtp" in name or "indexer" in name:
            continue
        # routed experts excluded by design (docstring contract: routing
        # stability + adapter count). GLM53_V10_LORA_EXPERTS=1 opts in.
        if ".mlp.experts." in name and \
                os.environ.get("GLM53_V10_LORA_EXPERTS", "0") != "1":
            continue
        if not name.rsplit(".", 1)[-1] in targets:
            continue
        if hasattr(mod, "lora_A"):
            continue                            # idempotent
        if not hasattr(mod, "weight"):
            continue
        row_mode = _is_row(mod)
        in_dim, out_dim = _dims(mod)
        dev, dt = mod.weight.device, mod.weight.dtype
        torch.manual_seed(seed)
        A = nn.Parameter(torch.empty(r, in_dim, dtype=dt, device=dev))
        torch.nn.init.normal_(A, mean=0.0, std=0.02)
        B = nn.Parameter(torch.zeros(out_dim, r, dtype=dt, device=dev))
        mod.register_parameter("lora_A", A)
        mod.register_parameter("lora_B", B)
        mod.register_forward_hook(_make_hook(A, B, scale, row_mode))
        handled.append(f"{name}[{mode_tag(row_mode)},r={r}]")
    return handled


def mode_tag(row_mode):
    return "row" if row_mode else "column"
