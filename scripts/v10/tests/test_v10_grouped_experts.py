#!/usr/bin/env python
"""V10GroupedExperts tests — the pad+bmm grouped MoE math vs the manual
per-expert loop (the SequentialMLP semantics), CPU, world=1."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10 = os.path.dirname(HERE)
for p in (V10, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from v10_grouped_experts import V10GroupedExperts  # noqa: E402
from v10_fill import flush_grouped, shard_slice  # noqa: E402


def _ref_experts(x, counts, w1, w2, probs=None, clamp=10.0):
    """The manual per-expert loop = the SequentialMLP semantics."""
    outs = []
    s = 0
    for e, c in enumerate(counts):
        xe = x[s:s + c]
        y = xe @ w1[e]
        gate, up = y.chunk(2, dim=-1)
        gate = gate.clamp(max=clamp)
        up = up.clamp(min=-clamp, max=clamp)
        inter = torch.nn.functional.silu(gate) * up
        if probs is not None:
            inter = inter * probs[s:s + c].unsqueeze(-1)
        outs.append(inter @ w2[e])
        s += c
    return torch.cat(outs)


def _mod(E=4, H=16, f=8, tp=1):
    class Cfg:
        pass
    c = Cfg()
    c.hidden_size = H
    c.moe_ffn_hidden_size = f
    c.tensor_model_parallel_size = tp
    c.params_dtype = torch.bfloat16
    c.activation_func_clamp_value = 10.0
    torch.manual_seed(3)
    return V10GroupedExperts(E, c)


def test_forward_matches_manual_loop():
    m = _mod()
    counts = [3, 0, 5, 2]                    # includes an EMPTY expert
    T = sum(counts)
    x = torch.randn(T, 16, dtype=torch.bfloat16)
    probs = torch.rand(T, dtype=torch.bfloat16)
    with torch.no_grad():
        m.w1.normal_(0, 0.02)
        m.w2.normal_(0, 0.02)
    out, bias = m(x, torch.tensor(counts), probs)
    ref = _ref_experts(x.float(), counts, m.w1.float(), m.w2.float(),
                       probs.float()).to(x.dtype)
    assert bias is None
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2), \
        f"maxdiff={(out.float()-ref.float()).abs().max().item()}"


def test_empty_expert_group():
    m = _mod()
    counts = [0, 0, 4, 0]
    x = torch.randn(4, 16, dtype=torch.bfloat16)
    out, _ = m(x, torch.tensor(counts))
    assert out.shape == (4, 16) and torch.isfinite(out.float()).all()


def test_grads_flow_native_autograd():
    m = _mod()
    counts = [3, 1, 2, 2]
    x = torch.randn(8, 16, dtype=torch.bfloat16, requires_grad=True)
    probs = torch.rand(8, dtype=torch.bfloat16)
    out, _ = m(x, torch.tensor(counts), probs)
    out.float().sum().backward()
    assert m.w1.grad is not None and torch.isfinite(m.w1.grad).all()
    assert m.w2.grad is not None and torch.isfinite(m.w2.grad).all()
    assert x.grad is not None


def test_chunked_bmm_matches_single_chunk(monkeypatch=None):
    """GLM53_V10_GROUPED_CHUNK forces the multi-chunk path (skew guard) —
    the chunked scatter unpad must reproduce the single-shot result."""
    import os
    m = _mod()
    counts = [3, 0, 5, 2]                    # max_t=5 -> 3 chunks at chunk=2
    T = sum(counts)
    x = torch.randn(T, 16, dtype=torch.bfloat16)
    probs = torch.rand(T, dtype=torch.bfloat16)
    with torch.no_grad():
        m.w1.normal_(0, 0.02)
        m.w2.normal_(0, 0.02)
    old = os.environ.get("GLM53_V10_GROUPED_CHUNK")
    os.environ["GLM53_V10_GROUPED_CHUNK"] = "2"
    try:
        out, _ = m(x, torch.tensor(counts), probs)
    finally:
        if old is None:
            os.environ.pop("GLM53_V10_GROUPED_CHUNK", None)
        else:
            os.environ["GLM53_V10_GROUPED_CHUNK"] = old
    ref = _ref_experts(x.float(), counts, m.w1.float(), m.w2.float(),
                       probs.float()).to(x.dtype)
    assert torch.allclose(out.float(), ref.float(), atol=1e-2, rtol=1e-2), \
        f"maxdiff={(out.float()-ref.float()).abs().max().item()}"


def test_zero_init_b_is_not_used():
    """The weights are torch.empty — the FILL owns them; the module must not
    silently initialize (the fill ledger's unwritten check relies on it)."""
    m = _mod()
    assert m.w1.is_leaf and m.w2.is_leaf


class _FakeCkpt:
    """Duck-typed SlicedCheckpoint: dict-backed load with real TP slicing."""

    def __init__(self, tensors):
        self.t = tensors

    def load(self, key, mode, tp_rank, tp_size):
        return shard_slice(self.t[key], mode, tp_rank, tp_size)


def test_flush_grouped_matches_sequential_fill_transposed():
    """flush_grouped w1/w2 must be the BIT-IDENTICAL transpose of the
    SequentialMLP fill content (fc1 = cat(shard(gate), shard(up));
    fc2 = shard(down, row))."""
    H, FT, TP, E, LI = 16, 8, 2, 3, 0        # FT = full ffn dim = f_loc*tp
    torch.manual_seed(7)
    vendor = {}
    for e in range(E):
        pfx = f"model.language_model.layers.{LI}.mlp.experts.{e}."
        vendor[pfx + "gate_proj.weight"] = torch.randn(FT, H)
        vendor[pfx + "up_proj.weight"] = torch.randn(FT, H)
        vendor[pfx + "down_proj.weight"] = torch.randn(H, FT)
    ckpt = _FakeCkpt(vendor)
    lp = "decoder.layers.0."
    f_loc = FT // TP

    def _entries():
        return {(lp, LI): {e: {
            "gate_proj": f"model.language_model.layers.{LI}.mlp.experts.{e}.gate_proj.weight",
            "up_proj": f"model.language_model.layers.{LI}.mlp.experts.{e}.up_proj.weight",
            "down_proj": f"model.language_model.layers.{LI}.mlp.experts.{e}.down_proj.weight",
        } for e in range(E)}}

    for rank in range(TP):                     # each rank fills its OWN w1/w2
        w1_t = torch.empty(E, H, 2 * f_loc, dtype=torch.float32)
        w2_t = torch.empty(E, f_loc, H, dtype=torch.float32)
        targets = {lp + "mlp.experts.w1": w1_t, lp + "mlp.experts.w2": w2_t}
        written = {}
        grouped = _entries()
        flush_grouped(grouped, LI, lp, ckpt, targets, written, rank, TP)
        assert written == {lp + "mlp.experts.w1": tuple(w1_t.shape),
                           lp + "mlp.experts.w2": tuple(w2_t.shape)}
        assert not grouped                    # entry popped
        for e in range(E):
            pfx = f"model.language_model.layers.{LI}.mlp.experts.{e}."
            # sequential fill content (fc1 composite / fc2 assign), transposed
            fc1 = torch.cat([vendor[pfx + "gate_proj.weight"][rank * f_loc:(rank + 1) * f_loc],
                             vendor[pfx + "up_proj.weight"][rank * f_loc:(rank + 1) * f_loc]], 0)
            fc2 = vendor[pfx + "down_proj.weight"][:, rank * f_loc:(rank + 1) * f_loc]
            assert torch.equal(w1_t[e], fc1.transpose(0, 1)), f"rank {rank} w1"
            assert torch.equal(w2_t[e], fc2.transpose(0, 1)), f"rank {rank} w2"


def test_flush_grouped_noop_without_entries():
    targets = {"mlp.experts.w1": None, "mlp.experts.w2": None}
    written = {}
    flush_grouped({}, 0, "mlp.experts.", _FakeCkpt({}), targets, written, 0, 1)
    assert written == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
