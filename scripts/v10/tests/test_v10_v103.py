#!/usr/bin/env python
"""V10.3 recipe-change tests (CPU, world=1).

Covers the three code-level deltas of the V10.3 single-variable arm:
  1. _mtp_frozen freeze surface — router + every RMSNorm inside mtp.* frozen,
     eh_proj trainable by default (knob flips it), attention linears + expert
     MLPs stay trainable. Real mcore key names from the export mapping.
  2. _kl_reduce selector — tk>0 keeps the legacy topk path; tk=0 selects the
     FULL-vocab chunked_vocab_kl; gradient flows into the draft input only.
  3. (regression) chunked_vocab_kl sanity is owned by test_v10_kl.py.
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10 = os.path.dirname(HERE)
for p in (V10, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from v10_trainer import _mtp_frozen, _kl_reduce, _mtp_grad_bucket  # noqa: E402

MT = "mtp.layers.0."

FROZEN_KEYS = [
    MT + "enorm.weight",
    MT + "hnorm.weight",
    MT + "final_layernorm.weight",
    MT + "mtp_model_layer.input_layernorm.weight",
    MT + "mtp_model_layer.pre_mlp_layernorm.weight",
    MT + "mtp_model_layer.mlp.router.weight",
    MT + "mtp_model_layer.self_attention.q_a_layernorm.weight",
    MT + "mtp_model_layer.self_attention.kv_a_layernorm.weight",
    MT + "mtp_model_layer.self_attention.indexer.k_norm.weight",
    MT + "mtp_model_layer.self_attention.indexer.k_norm.bias",
]

TRAINABLE_KEYS = [
    MT + "eh_proj.weight",
    MT + "mtp_model_layer.self_attention.q_a_proj.weight",
    MT + "mtp_model_layer.self_attention.q_b_proj.weight",
    MT + "mtp_model_layer.self_attention.kv_a_proj_with_mqa.weight",
    MT + "mtp_model_layer.self_attention.kv_b_proj.weight",
    MT + "mtp_model_layer.self_attention.o_proj.weight",
    MT + "mtp_model_layer.self_attention.indexer.wk.weight",
    MT + "mtp_model_layer.mlp.shared_experts.linear_fc1.weight",
    MT + "mtp_model_layer.mlp.experts.local_experts.0.linear_fc1.weight",
    MT + "mtp_model_layer.mlp.experts.local_experts.287.linear_fc2.weight",
]


def test_frozen_surface_v103_pinned():
    """V10.3 launcher pins ROUTER=1/NORMS=1 — the surface as designed."""
    os.environ["GLM53_V10_FREEZE_MTP_ROUTER"] = "1"
    os.environ["GLM53_V10_FREEZE_MTP_NORMS"] = "1"
    try:
        for k in FROZEN_KEYS:
            assert _mtp_frozen(k), f"{k} must be frozen (router+norms surface)"
        for k in TRAINABLE_KEYS:
            assert not _mtp_frozen(k), f"{k} must stay trainable"
    finally:
        del os.environ["GLM53_V10_FREEZE_MTP_ROUTER"]
        del os.environ["GLM53_V10_FREEZE_MTP_NORMS"]


def test_replay_safety_defaults_off():
    """No V10.3 envs set → nothing frozen, legacy TOPK default — a relaunch of
    the V10.2 launcher on the new code replays byte-identically."""
    assert not _mtp_frozen(MT + "enorm.weight")
    assert not _mtp_frozen(MT + "mtp_model_layer.mlp.router.weight")
    assert not _mtp_frozen(MT + "eh_proj.weight")


def test_knobs():
    os.environ["GLM53_V10_FREEZE_EH_PROJ"] = "1"
    try:
        assert _mtp_frozen(MT + "eh_proj.weight")
    finally:
        del os.environ["GLM53_V10_FREEZE_EH_PROJ"]
    os.environ["GLM53_V10_FREEZE_MTP_ROUTER"] = "0"
    try:
        assert not _mtp_frozen(MT + "mtp_model_layer.mlp.router.weight")
    finally:
        del os.environ["GLM53_V10_FREEZE_MTP_ROUTER"]
    os.environ["GLM53_V10_FREEZE_MTP_NORMS"] = "0"
    try:
        assert not _mtp_frozen(MT + "enorm.weight")
        assert not _mtp_frozen(MT + "mtp_model_layer.mlp.router.weight")
    finally:
        del os.environ["GLM53_V10_FREEZE_MTP_NORMS"]
    # decoder LoRA / non-mtp names are never classified by this surface
    assert not _mtp_frozen("decoder.layers.0.mlp.router.weight")


def test_kl_reduce_full_vocab_default():
    torch.manual_seed(0)
    main = torch.randn(8, 1, 32)
    draft = torch.randn(8, 1, 32, requires_grad=True)
    kl = _kl_reduce(main.detach(), draft, tk=0)
    assert kl.dim() == 0 and float(kl) > 0
    kl.backward()
    assert draft.grad is not None and bool(torch.isfinite(draft.grad).all())
    assert main.grad is None  # no grad path into the (detached) reference
    # selector output == direct full-vocab call
    from v10_diag import chunked_vocab_kl
    d2 = draft.detach().requires_grad_(True)
    assert torch.allclose(_kl_reduce(main.detach(), d2, tk=0),
                          chunked_vocab_kl(main.detach(), d2))


def test_kl_reduce_topk_legacy_path():
    from v10_diag import chunked_topk_kl
    torch.manual_seed(1)
    main = torch.randn(8, 1, 32)
    draft = torch.randn(8, 1, 32)
    assert torch.allclose(_kl_reduce(main.detach(), draft, tk=8),
                          chunked_topk_kl(main.detach(), draft, topk=8))


def test_mtp_grad_bucket_semantic_split():
    """R8 audit buckets must be SEMANTIC — one aggregate mtp_model_layer
    bucket would hide a recompute-dropped region (experts) behind surviving
    attention grads. Real mcore key names from the export mapping."""
    B = lambda *tail: ".".join(("mtp.layers.0.mtp_model_layer",) + tail)  # noqa: E731
    cases = {
        B("self_attention.q_a_proj.weight"):
            B("self_attention"),
        B("self_attention.o_proj.weight"):
            B("self_attention"),
        B("self_attention.q_a_layernorm.weight"):
            B("self_attention"),
        B("self_attention.indexer.wk.weight"):
            B("self_attention.indexer"),
        B("self_attention.indexer.k_norm.norm.weight"):
            B("self_attention.indexer"),
        B("mlp.router.weight"):
            B("mlp.router"),
        B("mlp.shared_experts.linear_fc1.weight"):
            B("mlp.shared_experts"),
        B("mlp.experts.local_experts.0.linear_fc1.weight"):
            B("mlp.experts"),
        B("mlp.experts.local_experts.287.linear_fc2.weight"):
            B("mlp.experts"),
        "mtp.layers.0.eh_proj.weight": "mtp.layers.0.eh_proj",
        "mtp.layers.0.enorm.weight": "mtp.layers.0.enorm",
        "mtp.layers.0.hnorm.weight": "mtp.layers.0.hnorm",
        "mtp.layers.0.final_layernorm.weight": "mtp.layers.0.final_layernorm",
    }
    for name, bucket in cases.items():
        assert _mtp_grad_bucket(name) == bucket, name
    # the R8 masking scenario: attention vs routed experts MUST be separate
    buckets = {_mtp_grad_bucket(n) for n in
               (B("self_attention.q_b_proj.weight"),
                B("mlp.experts.local_experts.0.linear_fc2.weight"))}
    assert len(buckets) == 2


def test_depth_accept_guards_fail_loud():
    """Config drift (B>1 / mtp_num_layers≠1) must AssertionError, never
    silently print wrong observability proxies."""
    from types import SimpleNamespace
    from v10_trainer import _depth_accept
    ids_b1 = torch.zeros(1, 6, dtype=torch.long)
    dec = torch.zeros(6, 1, 4)
    # B>1 rejected at entry (model never touched)
    try:
        _depth_accept(None, torch.zeros(2, 6, dtype=torch.long), dec)
        raise SystemExit("B>1 must AssertionError")
    except AssertionError:
        pass
    # wrong mtp row count (2*S=12 expected) rejected before chunking
    stub = SimpleNamespace(mtp=lambda **kw: torch.zeros(10, 1, 4),
                           embedding=None)
    try:
        _depth_accept(stub, ids_b1, dec)
        raise SystemExit("wrong mtp_out rows must AssertionError")
    except AssertionError:
        pass
