#!/usr/bin/env python
"""chunked_agree_margin (V10.2 signature: (draft, va, margin)) + chunked_topk_kl
tests (CPU, world=1 -> gather is identity)."""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10 = os.path.dirname(HERE)
for p in (V10, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from v10_diag import chunked_agree_margin, chunked_topk_kl  # noqa: E402


def test_agree_zero_when_argmax_matches():
    V = 64
    d = torch.full((8, 1, V), -10.0)
    va = torch.randint(0, V, (8, 1))
    for i in range(8):
        d[i, 0, va[i, 0]] = 10.0
    agree, _ = chunked_agree_margin(d, va, margin=0.5)
    assert float(agree) < 1e-3, agree


def test_agree_positive_when_diverged():
    torch.manual_seed(1)
    d = torch.randn(8, 1, 64)
    va = torch.randint(0, 64, (8, 1))
    agree, _ = chunked_agree_margin(d, va)
    assert float(agree) > 0


def test_margin_anchored_on_va():
    V = 64
    d = torch.full((1, 1, V), -10.0)
    va_token, other_token = 7, 42
    d[0, 0, va_token] = 0.0                 # va top, runner-up far below
    _, mg_ok = chunked_agree_margin(d, va=torch.tensor([[[va_token]]]), margin=0.5)
    assert float(mg_ok) == 0.0
    d2 = torch.full((1, 1, V), -10.0)
    d2[0, 0, va_token] = -9.8               # gap 0.2 < m=0.5 -> violated
    d2[0, 0, other_token] = -10.0
    _, mg_viol = chunked_agree_margin(d2, va=torch.tensor([[[va_token]]]), margin=0.5)
    assert abs(float(mg_viol) - 0.3) < 1e-5, mg_viol


def test_margin_grad_flows_to_draft():
    d = torch.randn(8, 1, 64).requires_grad_(True)
    va = torch.randint(0, 64, (8, 1))
    agree, marg = chunked_agree_margin(d, va)
    (agree + marg).backward()
    assert d.grad is not None and torch.isfinite(d.grad).all()


def test_topk_kl_identical_zero():
    lg = torch.randn(12, 1, 192)
    kl = chunked_topk_kl(lg, lg.clone(), topk=32)
    assert abs(float(kl)) < 1e-6, kl


def test_topk_kl_positive_and_focused():
    torch.manual_seed(0)
    main = torch.randn(10, 1, 192)
    draft = torch.randn(10, 1, 192)
    kl128 = chunked_topk_kl(main, draft, topk=128)
    assert float(kl128) > 0
    # 支撑内 KL 不随 topk 增大而降为 0; topk=V 全词表值 >= topk=128 值
    kl_full = chunked_topk_kl(main, draft, topk=192)
    assert float(kl_full) >= float(kl128) - 1e-4


def test_topk_kl_grad_flows_to_draft():
    main = torch.randn(10, 1, 192)
    draft = torch.randn(10, 1, 192).requires_grad_(True)
    chunked_topk_kl(main, draft, topk=32).backward()
    assert draft.grad is not None and torch.isfinite(draft.grad).all()


if __name__ == "__main__":
    for f in (test_agree_zero_when_argmax_matches, test_agree_positive_when_diverged,
              test_margin_anchored_on_va, test_margin_grad_flows_to_draft,
              test_topk_kl_identical_zero, test_topk_kl_positive_and_focused,
              test_topk_kl_grad_flows_to_draft):
        f()
    print("agree_margin + topk_kl tests PASS")
