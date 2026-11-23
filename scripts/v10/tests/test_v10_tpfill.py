#!/usr/bin/env python
"""V10 TP-math tests — pure-tensor pins for the sharding semantics.

The two Phase-2a production bugs (kv_b double-split; conv/fc1 cat-then-shard)
only manifest at TP>1 and were silent (shapes coincided). These tests pin the
sharding math WITHOUT needing a 16-rank run: any regression to cat-then-shard
or a full-dims/double-split drift fails here on CPU.

Run on .12: python3 -m pytest scripts/v10/tests/test_v10_tpfill.py -q -p no:warnings
"""
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10 = os.path.dirname(HERE)
for p in (V10, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from test_v10_build import TINY_VC, cpu_env, _seed_params  # noqa: E402,F401


def _mcore_mtp_uses_mtp_model_layer() -> bool:
    """Version probe (0.18.2 has mtp_model_layer; 0.12.x names it
    transformer_layer) — MTP-build-dependent tests skip cleanly on old mcores."""
    try:
        import inspect

        from megatron.core.transformer.multi_token_prediction import (
            MultiTokenPredictionLayerSubmodules)
        return "mtp_model_layer" in inspect.signature(
            MultiTokenPredictionLayerSubmodules.__init__).parameters
    except Exception:
        return False


_MTP_OK = _mcore_mtp_uses_mtp_model_layer()

from v10_fill import (composite_fuse, param_shard_mode,  # noqa: E402
                      shard_slice, swap_eh_halves)


# ---------------------------------------------------------------------------
# shard-mode coverage
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _MTP_OK, reason="requires mcore 0.18.2 (mtp_model_layer)")
def test_param_shard_mode_covers_every_param(cpu_env):
    """param_shard_mode must classify EVERY model param + expert_bias buffer
    (an unknown mode raises at fill — this walks the whole tiny mtp model so
    any new module's params are covered the moment they're built)."""
    from v10_model import build_v10_model

    model, cfg = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=True)
    names = [n for n, _ in model.named_parameters()]
    names += [n for n, b in model.named_buffers()
              if n.endswith("mlp.router.expert_bias")]
    assert len(names) > 50
    for n in names:
        mode = param_shard_mode(n)  # raises ValueError on an unknown name
        assert mode in ("column", "row", "vocab", "replicated"), \
            f"bad mode {mode} for {n}"


def test_shard_mode_mtp_and_dsa_specifics():
    """The Flash TP scheme (vLLM-equivalent MLA): lora dims + indexer
    REPLICATED; head dims column; o_proj row; MTP eh_proj column;
    MTP enorm/hnorm replicated; expert_bias replicated."""
    assert param_shard_mode("decoder.layers.3.self_attention.q_b_proj.weight") == "column"
    assert param_shard_mode("decoder.layers.3.self_attention.kv_b_proj.weight") == "column"
    assert param_shard_mode("decoder.layers.3.self_attention.o_proj.weight") == "row"
    assert param_shard_mode("decoder.layers.3.self_attention.q_a_proj.weight") == "replicated"
    assert param_shard_mode(
        "decoder.layers.3.self_attention.kv_a_proj_with_mqa.weight") == "replicated"
    assert param_shard_mode(
        "decoder.layers.3.self_attention.indexer.linear_wq_b.weight") == "replicated"
    assert param_shard_mode(
        "decoder.layers.3.self_attention.indexer.index_kpool_compress_ape") == "replicated"
    assert param_shard_mode(
        "decoder.layers.3.self_attention.indexer.index_kpool_compress_gate") == "replicated"
    assert param_shard_mode("decoder.layers.0.self_attention.o_norm.weight") == "replicated"
    assert param_shard_mode("decoder.layers.0.self_attention.A_log") == "column"
    assert param_shard_mode("decoder.layers.0.self_attention.dt_bias") == "column"
    assert param_shard_mode("decoder.layers.0.self_attention.conv1d_weight") == "column"
    assert param_shard_mode("decoder.layers.0.self_attention.linear_q_proj.weight") == "column"
    assert param_shard_mode("decoder.layers.0.self_attention.linear_o_proj.weight") == "row"
    assert param_shard_mode("decoder.layers.0.self_attention.linear_fa_proj.weight") == "replicated"
    assert param_shard_mode("decoder.layers.0.attn_hc.fn") == "replicated"
    assert param_shard_mode("decoder.layers.0.mlp.router.weight") == "replicated"
    assert param_shard_mode("decoder.layers.0.mlp.router.expert_bias") == "replicated"
    assert param_shard_mode("decoder.layers.3.mlp.experts.local_experts.0.linear_fc1.weight") == "column"
    assert param_shard_mode("decoder.layers.3.mlp.experts.local_experts.0.linear_fc2.weight") == "row"
    assert param_shard_mode("decoder.layers.3.mlp.shared_experts.linear_fc1.weight") == "column"
    assert param_shard_mode("mtp.layers.0.eh_proj.weight") == "column"
    assert param_shard_mode("mtp.layers.0.enorm.weight") == "replicated"
    assert param_shard_mode("mtp.layers.0.hnorm.weight") == "replicated"
    assert param_shard_mode("mtp.layers.0.final_layernorm.weight") == "replicated"
    assert param_shard_mode("embedding.word_embeddings.weight") == "vocab"
    assert param_shard_mode("output_layer.weight") == "vocab"


# ---------------------------------------------------------------------------
# shard_slice algebra
# ---------------------------------------------------------------------------
def test_shard_slice_concat_back_column_row_vocab():
    """cat of per-rank shards == the full tensor (the fill's inverse property);
    per-rank shapes = full/tp on the sharded dim."""
    torch.manual_seed(5)
    for shape, mode, dim in (((64, 16), "column", 0), ((64, 16), "vocab", 0),
                             ((16, 64), "row", 1)):
        t = torch.randn(*shape)
        tp = 4
        shards = [shard_slice(t, mode, r, tp) for r in range(tp)]
        assert all(s.shape[dim] == shape[dim] // tp for s in shards)
        assert torch.equal(torch.cat(shards, dim=dim), t), f"{mode} not invertible"


def test_shard_slice_replicated_and_tp1_passthrough():
    t = torch.randn(8, 8)
    assert shard_slice(t, "replicated", 3, 8) is t
    assert shard_slice(t, "column", 2, 1) is t


# ---------------------------------------------------------------------------
# composite per-rank interleave (the two production-bug regressions)
# ---------------------------------------------------------------------------
def test_conv_composite_per_rank_interleave():
    """REGRESSION (Phase-2a bug #1): conv1d = cat(q,k,v) of three INDEPENDENT
    column outputs — rank r must get [q_r; k_r; v_r]. cat-then-shard would
    hand rank 0 a pure-q block; this test fails under that regression."""
    torch.manual_seed(6)
    per, k, tp = 16, 4, 4  # full conv [3*16, 1, 4] built from three [16,1,4]
    q = torch.randn(per, 1, k)
    c_q = torch.randn(per, 1, k)
    c_k = torch.randn(per, 1, k)
    c_v = torch.randn(per, 1, k)
    fused_full = torch.cat([c_q, c_k, c_v], dim=0)  # what cat-then-shard gives
    for r in range(tp):
        got = composite_fuse([c_q, c_k, c_v], "column", r, tp)
        s = per // tp
        want = torch.cat([c_q[r * s:(r + 1) * s], c_k[r * s:(r + 1) * s],
                          c_v[r * s:(r + 1) * s]], dim=0)
        assert torch.equal(got, want), f"rank {r}: conv interleave broken"
        # the WRONG form (cat-then-shard) must DIFFER (probe-sensitive values)
        wrong = fused_full[r * (3 * per // tp):(r + 1) * (3 * per // tp)]
        assert not torch.equal(got, wrong), \
            "interleave degenerated to cat-then-shard"


def test_fc1_composite_matches_glu_local_chunk():
    """REGRESSION (Phase-2a, the fc1 variant): mcore glu() chunks the TP-LOCAL
    fc1 output dim=-1 -> the local gate rows are the FIRST HALF of the LOCAL
    block. So rank r's fc1 = [gate_r; up_r] (per-rank interleave), and the
    forward's glu split of that local block must reproduce gate_r/up_r."""
    torch.manual_seed(7)
    f, h, tp = 16, 8, 4
    gate = torch.randn(f, h)
    up = torch.randn(f, h)
    s = f // tp
    for r in range(tp):
        fc1_r = composite_fuse([gate, up], "column", r, tp)
        assert fc1_r.shape == (2 * s, h)
        # forward glu semantics on the LOCAL block: chunk(2, dim=-1) of the
        # activation [.., 2*s] -> gate rows = fc1_r[:s], linear rows = fc1_r[s:]
        g_loc, u_loc = torch.chunk(fc1_r, 2, dim=0)
        assert torch.equal(g_loc, gate[r * s:(r + 1) * s]), \
            f"rank {r}: local gate rows wrong (cat-then-shard regression)"
        assert torch.equal(u_loc, up[r * s:(r + 1) * s])


# ---------------------------------------------------------------------------
# eh_proj swap algebra
# ---------------------------------------------------------------------------
def test_swap_eh_halves_column_slice_order_invariance():
    """The fill swaps [h|e]->[e|h] on the FULL tensor, THEN shard_slice slices
    rows (dim0) — order independent. Pin: swap-then-slice == slice-then-swap."""
    torch.manual_seed(8)
    out, h2 = 32, 16  # [out, 2h] with 2h=16
    w = torch.randn(out, h2)
    tp = 4
    for r in range(tp):
        a = shard_slice(swap_eh_halves(w), "column", r, tp)
        b = swap_eh_halves(shard_slice(w, "column", r, tp))
        assert torch.equal(a, b), f"rank {r}: swap/slice order matters (bug)"


def test_swap_eh_halves_rejects_bad_shapes():
    with pytest.raises(ValueError):
        swap_eh_halves(torch.randn(4, 7))          # odd 2h
    with pytest.raises(ValueError):
        swap_eh_halves(torch.randn(2, 4, 8))       # not 2D


# ---------------------------------------------------------------------------
# vocab-parallel CE math (launch_ce_gate)
# ---------------------------------------------------------------------------
def test_vocab_parallel_ce_gather_math():
    """The gate's CE: all_gather partial-vocab logits, cat dim=-1 -> full-vocab
    softmax == directly-computed full softmax."""
    torch.manual_seed(9)
    R, V, tp = 12, 48, 4
    full = torch.randn(R, V)
    parts = [full[:, r * V // tp:(r + 1) * V // tp].contiguous() for r in range(tp)]
    gathered = torch.cat(parts, dim=-1)
    assert torch.equal(gathered, full)
    tgt = torch.randint(0, V, (R,))
    ce_full = torch.nn.functional.cross_entropy(full, tgt, reduction="none")
    ce_gather = torch.nn.functional.cross_entropy(gathered, tgt, reduction="none")
    assert torch.allclose(ce_full, ce_gather, atol=1e-6)


# ---------------------------------------------------------------------------
# chunked_ce_margin contract (shared V5.4a asset)
# ---------------------------------------------------------------------------
def test_chunked_ce_margin_return_shapes_and_ignore_index():
    """want_margin=False -> scalar; True -> (ce, margin). IGNORE_INDEX rows
    contribute to NEITHER ce nor margin (sum/count semantics)."""
    from v10_ce_margin import chunked_ce_margin

    torch.manual_seed(10)
    R, V = 10, 32
    head = torch.nn.Linear(8, V, bias=False)
    rows = torch.randn(R, 8)
    targets = torch.randint(0, V, (R,))
    targets[3] = -100
    targets[7] = -100
    ce0 = chunked_ce_margin(head, rows, targets, want_margin=False)
    assert ce0.ndim == 0
    ce1, mg1 = chunked_ce_margin(head, rows, targets, want_margin=True, margin=0.5)
    # manual: supervised rows only
    sup = targets != -100
    logits = head(rows[sup]).float()
    ce_manual = torch.nn.functional.cross_entropy(logits, targets[sup],
                                                  reduction="mean")
    assert torch.allclose(ce1.float(), ce_manual, atol=1e-5)
    t_logit = logits.gather(1, targets[sup].unsqueeze(1)).squeeze(1)
    top2 = logits.topk(2, dim=1)
    other = torch.where(top2.indices[:, 0] == targets[sup].unsqueeze(1).squeeze(1),
                        top2.values[:, 1], top2.values[:, 0])
    mg_manual = torch.nn.functional.relu(0.5 - (t_logit - other)).mean()
    assert torch.allclose(mg1.float(), mg_manual, atol=1e-5)
    # all-ignored -> zero tensor, finite
    tg2 = torch.full((R,), -100)
    ce2 = chunked_ce_margin(head, rows, tg2, want_margin=False)
    assert ce2.ndim == 0 and ce2.item() == 0.0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


# ---------------------------------------------------------------------------
# causal_ends_from_mask contract (Phase-2a fix #2 — rejection paths)
# ---------------------------------------------------------------------------
def test_causal_ends_from_mask_accepts_prefix_masks():
    from v10_modules import causal_ends_from_mask
    S, B = 8, 2
    keep = torch.ones(B, 1, S, S, dtype=torch.bool).tril()
    ends = causal_ends_from_mask(keep, S)
    assert ends.shape == (B, S)
    assert torch.equal(ends, torch.arange(1, S + 1).unsqueeze(0).expand(B, -1))


def test_causal_ends_from_mask_rejects_non_prefix():
    """Left padding / middle holes must FAIL LOUD (V8 lesson: silent corrupt)."""
    from v10_modules import causal_ends_from_mask
    S = 8
    # a hole: tril keep-mask with column 3 dropped — rows stop being prefixes
    bad = torch.ones(S, S, dtype=torch.bool).tril()
    bad[:, 3] = False
    with pytest.raises(ValueError):
        causal_ends_from_mask(bad, S)
    # left-padded keep (keep starts at column 2): suffix-causal, not prefix
    left = torch.ones(S, S, dtype=torch.bool).tril()
    left[:, :2] = False
    with pytest.raises(ValueError):
        causal_ends_from_mask(left, S)
    # float mask
    with pytest.raises(ValueError):
        causal_ends_from_mask(torch.ones(1, S, S), S)
    # wrong shape
    with pytest.raises(ValueError):
        causal_ends_from_mask(torch.ones(S, S, dtype=torch.bool), S)


# ---------------------------------------------------------------------------
# CP restore: no silent partial restore
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _MTP_OK, reason="requires mcore 0.18.2 (mtp_model_layer)")
def test_cp_load_missing_key_raises(cpu_env):
    import tempfile

    from v10_model import build_v10_model
    from v10_trainer import _opt_param_names, build_trainable as freeze_base, load_cp, save_cp

    model, _ = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=True)
    _seed_params(model, seed=31)
    freeze_base(model)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    with tempfile.TemporaryDirectory() as td:
        save_cp(model, opt, 1, td)
        blob = torch.load(os.path.join(td, "cp1_rank0.pt"), weights_only=False)
        # drop one key -> load must RAISE, not silently skip
        blob["shard"].pop(next(iter(blob["shard"])))
        with tempfile.TemporaryDirectory() as td2:
            torch.save(blob, os.path.join(td2, "cp1_rank0.pt"))
            model2, _ = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=True)
            freeze_base(model2)
            params2 = [p for n, p in model2.named_parameters() if p.requires_grad]
            opt2 = torch.optim.AdamW(params2, lr=1e-3)
            with pytest.raises(KeyError):
                load_cp(model2, opt2, os.path.join(td2, "cp1_rank0.pt"))


@pytest.mark.skipif(not _MTP_OK, reason="requires mcore 0.18.2 (mtp_model_layer)")
def test_cp_load_shape_mismatch_raises(cpu_env):
    import tempfile

    from v10_model import build_v10_model
    from v10_trainer import build_trainable as freeze_base, load_cp, save_cp

    model, _ = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=True)
    _seed_params(model, seed=32)
    freeze_base(model)
    params = [p for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    with tempfile.TemporaryDirectory() as td:
        save_cp(model, opt, 1, td)
        blob = torch.load(os.path.join(td, "cp1_rank0.pt"), weights_only=False)
        k = next(iter(blob["shard"]))
        blob["shard"][k] = torch.randn(3, 3)  # wrong shape
        with tempfile.TemporaryDirectory() as td2:
            torch.save(blob, os.path.join(td2, "cp1_rank0.pt"))
            model2, _ = build_v10_model(vc=dict(TINY_VC), device="cpu", mtp=True)
            freeze_base(model2)
            params2 = [p for n, p in model2.named_parameters() if p.requires_grad]
            opt2 = torch.optim.AdamW(params2, lr=1e-3)
            with pytest.raises(ValueError):
                load_cp(model2, opt2, os.path.join(td2, "cp1_rank0.pt"))
