#!/usr/bin/env python
"""chunked_vocab_kl tests (CPU, world=1 → gather is identity).

KL(main‖draft): identical dists -> 0; shifted -> positive & correct;
gradient flows into the draft input only (main detached by the caller).
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10 = os.path.dirname(HERE)
for p in (V10, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from v10_diag import chunked_vocab_kl  # noqa: E402


def _logits(seed, V=64):
    torch.manual_seed(seed)
    return torch.randn(12, 1, V)


def test_identical_zero():
    lg = _logits(0)
    kl = chunked_vocab_kl(lg, lg.clone())
    assert abs(float(kl)) < 1e-6, kl


def test_shifted_positive_and_value():
    torch.manual_seed(0)
    main = torch.randn(10, 1, 64)
    draft = torch.randn(10, 1, 64)
    pm = torch.log_softmax(main, -1).exp()
    pd = torch.log_softmax(draft, -1)
    ref = (pm * (torch.log_softmax(main, -1) - pd)).sum(-1).mean()
    got = chunked_vocab_kl(main, draft, chunk=4)
    assert float(got) > 0
    assert abs(float(got) - float(ref)) < 1e-5, (got, ref)


def test_grad_flows_to_draft_only():
    main = _logits(1).requires_grad_(True)
    draft = _logits(2).requires_grad_(True)
    # the CALLER detaches main (trainer contract) — the graph starts at draft
    chunked_vocab_kl(main.detach(), draft).backward()
    assert draft.grad is not None and torch.isfinite(draft.grad).all()
    assert main.grad is None                      # detach cut the graph


if __name__ == "__main__":
    test_identical_zero()
    test_shifted_positive_and_value()
    test_grad_flows_to_draft_only()
    print("kl tests PASS")
