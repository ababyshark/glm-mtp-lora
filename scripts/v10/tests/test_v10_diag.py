#!/usr/bin/env python
"""v10_diag helper tests — the measurement backbone (CPU, one local test uses
a 2-process gloo gather to validate the all_gather path end-to-end)."""
import os
import subprocess
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10 = os.path.dirname(HERE)
for p in (V10, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from v10_diag import (full_vocab_ce, gather_full_logits,  # noqa: E402
                      max_abs_diff, rolled_targets)

_GATHER_CHILD = r"""
import os, sys, torch
import torch.distributed as dist
sys.path.insert(0, sys.argv[1])
dist.init_process_group(backend="gloo", init_method="env://")
from v10_diag import gather_full_logits, full_vocab_ce

rank = dist.get_rank()
# full logits [S=6, V=8]; rank r owns rows [r*4:(r+1)*4] of a KNOWN matrix
torch.manual_seed(3)
full = torch.randn(6, 8)
shard = full[:, rank * 4:(rank + 1) * 4].contiguous()
got = gather_full_logits(shard, dim=-1)
assert torch.equal(got, full), f"rank{rank} gather mismatch"
ce = full_vocab_ce(got, torch.tensor([1, 2, 3, 4, 5, 6]))
ref = -(torch.log_softmax(full, dim=-1).gather(-1, torch.arange(1, 7).unsqueeze(-1)).squeeze(-1))
assert torch.equal(ce, ref), f"rank{rank} CE mismatch"
print(f"GATHER_CE_OK rank{rank}", flush=True)
dist.barrier()
dist.destroy_process_group()
"""


def test_gather_full_logits_and_ce_two_rank_gloo():
    import socket
    import tempfile

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    with tempfile.TemporaryDirectory() as td:
        script = os.path.join(td, "child.py")
        with open(script, "w") as f:
            f.write(_GATHER_CHILD)
        env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                   WORLD_SIZE="2")
        procs = [subprocess.Popen([sys.executable, script, V10],
                                  env={**env, "RANK": str(r)},
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                 for r in range(2)]
        outs = []
        for r, p in enumerate(procs):
            so, se = p.communicate(timeout=120)
            assert p.returncode == 0, \
                f"rank{r} rc={p.returncode}\n{so.decode()}\n{se.decode()}"
            outs.append(so.decode())
        assert "GATHER_CE_OK rank0" in outs[0]
        assert "GATHER_CE_OK rank1" in outs[1]


def test_max_abs_diff_detects_half_swap_sum_does_not():
    """The first eh-probe lesson, pinned: sum is invariant to a half swap."""
    g = torch.randn(4, 8)
    a = torch.cat([g, -g], dim=1)
    b = torch.cat([-g, g], dim=1)          # halves swapped
    assert abs(a.abs().sum().item() - b.abs().sum().item()) < 1e-4, \
        "precondition: sum must be (near-)invariant here"
    assert max_abs_diff(a, b) > 0, "max-abs diff MUST detect the swap"


def test_full_vocab_ce_matches_manual():
    torch.manual_seed(1)
    full = torch.randn(5, 16)
    tgt = torch.tensor([2, 0, 15, 7, 9])
    got = full_vocab_ce(full, tgt)
    ref = -(torch.log_softmax(full, dim=-1)[torch.arange(5), tgt])
    assert torch.allclose(got, ref, atol=1e-6)
    # CE of the max-logit position is ~ -log(p) and never below 0
    assert (got >= -1e-5).all()


def test_rolled_targets_alignment():
    ids = torch.tensor([[5, 6, 7, 8]])
    assert torch.equal(rolled_targets(ids), torch.tensor([6, 7, 8]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def _npu_ok():
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except Exception:
        return True  # torch_npu not installed -> clean CPU path


@pytest.mark.skipif(not _npu_ok(),
                    reason="torch_npu present but ACL broken on this host "
                           "(dev box 507008, fork-detected); verified on .40 "
                           "and pure-CPU hosts")
def test_chunked_ce_matches_direct_world1():
    """world=1 (no gather): the chunked/checkpointed path must equal the
    direct full-vocab CE bit-for-bit-ish (fp32, same math)."""
    from v10_diag import chunked_full_vocab_ce, full_vocab_ce
    torch.manual_seed(2)
    full = torch.randn(10, 1, 32, requires_grad=True)   # [S, B=1, V]
    tgt = torch.randint(0, 32, (10, 1))                 # [S, B]
    ref = full_vocab_ce(full[:, 0], tgt[:, 0]).mean()
    got = chunked_full_vocab_ce(full, tgt, chunk=4)
    assert torch.allclose(got, ref, atol=1e-6)
    got.sum().backward()
    assert full.grad is not None and full.grad.abs().sum() > 0
