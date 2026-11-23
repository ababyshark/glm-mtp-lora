#!/usr/bin/env python
"""fixed_diag_kernel regression test (ledger #23).

The #19 squeeze fix introduced an off-by-one: the chunk loop's range()
start bound alone let the LAST chunk overrun tgt by one row and the
(am == tgt) broadcast exploded (1024 vs 1023). The kernel must:
  - iterate EXACTLY len(tgt) rows (an extra poisoned logits row ignored)
  - clamp the last chunk (n_pos not a multiple of chunk)
  - handle both [S,B,V] (B=1) and [S,V] logits layouts
Echo contract: argmax(logits[t]) == tgt[t] => agree 1.0 / ce ~0.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10 = os.path.dirname(HERE)
for p in (V10, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from v10_trainer import fixed_diag_kernel  # noqa: E402

V = 512


def _echo_logits(tgt, batched):
    """[n_pos+1, 1, V] or [n_pos+1, V]; rows < n_pos echo tgt, LAST row
    poisoned (argmax = 404) — the kernel must never read it."""
    n_pos = tgt.shape[0]
    shape = (n_pos + 1, 1, V) if batched else (n_pos + 1, V)
    logits = torch.full(shape, -10.0)
    for t in range(n_pos):
        if batched:
            logits[t, 0, int(tgt[t])] = 10.0
        else:
            logits[t, int(tgt[t])] = 10.0
    if batched:
        logits[n_pos, 0, 404 % V] = 10.0
    else:
        logits[n_pos, 404 % V] = 10.0
    return logits


def test_echo_batched_extra_row_ignored():
    torch.manual_seed(0)
    tgt = torch.randint(0, V - 2, (1023,))          # real eval shape S-1
    logits = _echo_logits(tgt, batched=True)
    ce, agree = fixed_diag_kernel(logits, tgt, gather_fn=lambda x: x)
    assert abs(agree - 1.0) < 1e-9, agree
    assert ce < 1e-3, ce


def test_chunk_boundary_not_multiple():
    torch.manual_seed(1)
    tgt = torch.randint(0, V - 2, (300,))           # 300 = 256 + 44
    logits = _echo_logits(tgt, batched=False)
    ce, agree = fixed_diag_kernel(logits, tgt, gather_fn=lambda x: x, chunk=256)
    assert abs(agree - 1.0) < 1e-9, agree
    assert ce < 1e-3, ce


def test_single_row_chunk():
    tgt = torch.randint(0, V - 2, (1,))
    logits = _echo_logits(tgt, batched=True)
    ce, agree = fixed_diag_kernel(logits, tgt, gather_fn=lambda x: x, chunk=256)
    assert abs(agree - 1.0) < 1e-9 and ce < 1e-3


def test_wrong_predictions_low_agree():
    torch.manual_seed(2)
    tgt = torch.randint(0, V - 2, (100,))
    shape = (101, 1, V)
    logits = torch.full(shape, -10.0)
    for t in range(100):                            # argmax = tgt + 1 (wrong)
        logits[t, 0, int((tgt[t] + 1) % (V - 2))] = 10.0
    ce, agree = fixed_diag_kernel(logits, tgt, gather_fn=lambda x: x)
    assert agree == 0.0 and ce > 5.0
