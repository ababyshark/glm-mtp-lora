#!/usr/bin/env python
"""v10_lora tests — hook-based injection invariants (CPU, world=1):
zero-init B ⇒ EXACT base behavior; grads reach A/B; fill names untouched."""
import os
import sys

import pytest
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
V10 = os.path.dirname(HERE)
for p in (V10, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from v10_lora import apply_lora  # noqa: E402


def _tiny_model():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            layer = nn.Module()
            sa = nn.Module()
            sa.linear_q_proj = nn.Linear(8, 6, bias=False)
            sa.q_b_proj = nn.Linear(8, 6, bias=False)
            sa.linear_o_proj = nn.Linear(6, 8, bias=False)
            layer.self_attention = sa
            mlp = nn.Module()
            mlp.linear_fc1 = nn.Linear(8, 6, bias=False)
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([layer])
            layer.mlp = mlp
            # a non-target frozen param to verify the whitelist boundary
            self.embed = nn.Linear(8, 8, bias=False)
    return M()


def test_apply_lora_targets_and_idempotent():
    m = _tiny_model()
    handled = apply_lora(m, r=4, alpha=8)
    assert len(handled) == 4, handled                 # q/o/fc1 + q_b
    assert not any("embed" in h for h in handled)
    names = {n for n, _ in m.named_parameters() if "lora_" in n}
    assert len(names) == 8                            # A+B per target
    handled2 = apply_lora(m, r=4, alpha=8)
    assert handled2 == [], "double injection must be a no-op"


def test_zero_init_b_is_exact_identity():
    m = _tiny_model()
    apply_lora(m, r=4, alpha=8)
    lin = m.decoder.layers[0].self_attention.linear_q_proj
    x = torch.randn(3, 5, 8)
    with torch.no_grad():
        out_wrapped = lin(x)
        out_base = torch.nn.functional.linear(x, lin.weight)
    assert torch.equal(out_wrapped, out_base), \
        "B=0 init must leave the base behavior bit-exact"


def test_grads_reach_lora_params():
    m = _tiny_model()
    apply_lora(m, r=4, alpha=8)
    # simulate build_trainable's whitelist: only lora params trainable
    for n, p in m.named_parameters():
        if "lora_" not in n:
            p.requires_grad_(False)
    lin = m.decoder.layers[0].mlp.linear_fc1
    with torch.no_grad():
        lin.lora_B.normal_(0, 0.1)                    # break the zero init
    x = torch.randn(2, 4, 8)
    out = lin(x)
    out.sum().backward()
    assert lin.lora_A.grad is not None and lin.lora_A.grad.abs().sum() > 0
    assert lin.lora_B.grad is not None
    # base weight untouched (frozen in the trainer contract)
    assert lin.weight.grad is None


def test_fill_names_untouched():
    """The fill ledger keys are base-weight names — hooks must not rename
    anything (named_parameters keeps ...linear_q_proj.weight, plus NEW
    lora_* entries)."""
    m = _tiny_model()
    before = {n for n, _ in m.named_parameters()}
    apply_lora(m, r=4, alpha=8)
    after = {n for n, _ in m.named_parameters()}
    assert before <= after, "base names must survive injection"
    assert all("base" not in n for n in after), "no module replacement"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
