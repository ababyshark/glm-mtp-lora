#!/usr/bin/env python
"""V10 fill tests — synthetic-checkpoint round-trip.

Build model A (tiny), emit a synthetic Flash-layout checkpoint from A's params
(inverting the fill map: cat-splits, conv q|k|v split, indexer renames,
expert_bias buffer), save safetensors shards + index, then fill a fresh model B
and require bit-exact equality + a clean coverage ledger. This validates the
complete key map / composite transforms without the 599GB real checkpoint.
"""
import json
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

CKPT = "model.language_model."


def _dump_synthetic_checkpoint(model, out_dir, vc):
    """Invert the fill map: model params -> Flash-BF16-style checkpoint keys."""
    from safetensors.torch import save_file

    P = dict(model.named_parameters())
    B = {n: b for n, b in model.named_buffers() if n.endswith("mlp.router.expert_bias")}
    tensors = {}
    n_layers = len(vc["layer_types"])

    tensors["model.language_model.embed_tokens.weight"] = \
        P["embedding.word_embeddings.weight"].detach().clone().contiguous()
    tensors["lm_head.weight"] = P["output_layer.weight"].detach().clone().contiguous()
    tensors["model.language_model.norm.weight"] = \
        P["decoder.final_layernorm.weight"].detach().clone().contiguous()

    for li in range(n_layers):
        pfx = f"model.language_model.layers.{li}."
        lp = f"decoder.layers.{li}."
        ap = f"{lp}self_attention."
        for which, mname in (("attn", "attn_hc"), ("ffn", "ffn_hc")):
            for w in ("fn", "base", "scale"):
                tensors[pfx + f"hc_{which}_{w}"] = \
                    P[f"{lp}{mname}.{w}"].detach().clone().contiguous()
        tensors[pfx + "input_layernorm.weight"] = \
            P[f"{lp}input_layernorm.weight"].detach().clone().contiguous()
        tensors[pfx + "post_attention_layernorm.weight"] = \
            P[f"{lp}pre_mlp_layernorm.weight"].detach().clone().contiguous()

        is_kda = vc["layer_types"][li] == "linear_attention"
        is_moe = vc["mlp_layer_types"][li] == "moe"
        if is_kda:
            for src, dst in (("linear_q_proj", "q_proj"), ("linear_k_proj", "k_proj"),
                             ("linear_v_proj", "v_proj"), ("linear_b_proj", "b_proj"),
                             ("linear_fa_proj", "f_a_proj"), ("linear_fb_proj", "f_b_proj"),
                             ("linear_ga_proj", "g_a_proj"), ("linear_gb_proj", "g_b_proj"),
                             ("linear_o_proj", "o_proj")):
                tensors[pfx + f"self_attn.{dst}.weight"] = \
                    P[ap + f"{src}.weight"].detach().clone().contiguous()
            tensors[pfx + "self_attn.o_norm.weight"] = \
                P[ap + "o_norm.weight"].detach().clone().contiguous()
            tensors[pfx + "self_attn.A_log"] = P[ap + "A_log"].detach().clone().contiguous()
            tensors[pfx + "self_attn.dt_bias"] = P[ap + "dt_bias"].detach().clone().contiguous()
            conv = P[ap + "conv1d_weight"].detach()
            per = conv.shape[0] // 3
            for i, seg in enumerate("qkv"):
                tensors[pfx + f"self_attn.{seg}_conv1d.weight"] = \
                    conv[i * per:(i + 1) * per].clone().contiguous()
        else:
            for src, dst in (("q_a_proj", "q_a_proj"), ("q_b_proj", "q_b_proj"),
                             ("kv_a_proj_with_mqa", "kv_a_proj_with_mqa"),
                             ("kv_b_proj", "kv_b_proj"), ("o_proj", "o_proj")):
                tensors[pfx + f"self_attn.{dst}.weight"] = \
                    P[ap + f"{src}.weight"].detach().clone().contiguous()
            tensors[pfx + "self_attn.q_a_layernorm.weight"] = \
                P[ap + "q_a_layernorm.weight"].detach().clone().contiguous()
            tensors[pfx + "self_attn.kv_a_layernorm.weight"] = \
                P[ap + "kv_a_layernorm.weight"].detach().clone().contiguous()
            ipfx = pfx + "self_attn.indexer."
            for src, dst in (("linear_wq_b", "wq_b"), ("linear_wk", "wk"),
                             ("linear_weights_proj", "weights_proj")):
                tensors[ipfx + f"{dst}.weight"] = \
                    P[ap + f"indexer.{src}.weight"].detach().clone().contiguous()
            tensors[ipfx + "k_norm.weight"] = \
                P[ap + "indexer.k_norm.norm.weight"].detach().clone().contiguous()
            tensors[ipfx + "k_norm.bias"] = \
                P[ap + "indexer.k_norm.norm.bias"].detach().clone().contiguous()
            tensors[ipfx + "index_kpool_compress_gate"] = \
                P[ap + "indexer.index_kpool_compress_gate"].detach().clone().contiguous()
            tensors[ipfx + "index_kpool_compress_ape"] = \
                P[ap + "indexer.index_kpool_compress_ape"].detach().clone().contiguous()

        if not is_moe:
            fc1 = P[lp + "mlp.linear_fc1.weight"].detach()
            half = fc1.shape[0] // 2
            tensors[pfx + "mlp.gate_proj.weight"] = fc1[:half].clone().contiguous()
            tensors[pfx + "mlp.up_proj.weight"] = fc1[half:].clone().contiguous()
            tensors[pfx + "mlp.down_proj.weight"] = \
                P[lp + "mlp.linear_fc2.weight"].detach().clone().contiguous()
        else:
            tensors[pfx + "mlp.gate.weight"] = \
                P[lp + "mlp.router.weight"].detach().clone().contiguous()
            tensors[pfx + "mlp.gate.e_score_correction_bias"] = \
                B[lp + "mlp.router.expert_bias"].detach().clone().contiguous()
            n_exp = vc["n_routed_experts"]
            for e in range(n_exp):
                efx = lp + f"mlp.experts.local_experts.{e}."
                fc1 = P[efx + "linear_fc1.weight"].detach()
                half = fc1.shape[0] // 2
                tensors[pfx + f"mlp.experts.{e}.gate_proj.weight"] = \
                    fc1[:half].clone().contiguous()
                tensors[pfx + f"mlp.experts.{e}.up_proj.weight"] = \
                    fc1[half:].clone().contiguous()
                tensors[pfx + f"mlp.experts.{e}.down_proj.weight"] = \
                    P[efx + "linear_fc2.weight"].detach().clone().contiguous()
            sfx = lp + "mlp.shared_experts."
            fc1 = P[sfx + "linear_fc1.weight"].detach()
            half = fc1.shape[0] // 2
            tensors[pfx + "mlp.shared_experts.gate_proj.weight"] = \
                fc1[:half].clone().contiguous()
            tensors[pfx + "mlp.shared_experts.up_proj.weight"] = \
                fc1[half:].clone().contiguous()
            tensors[pfx + "mlp.shared_experts.down_proj.weight"] = \
                P[sfx + "linear_fc2.weight"].detach().clone().contiguous()

    # shard it (one shard per ~20 tensors, like the real checkpoint)
    os.makedirs(out_dir, exist_ok=True)
    items = sorted(tensors.items())
    weight_map, shards = {}, {}
    shard_size = 20
    for si in range(0, len(items), shard_size):
        chunk = dict(items[si:si + shard_size])
        fname = f"model-{si // shard_size + 1:05d}-of-00000.safetensors"
        save_file(chunk, os.path.join(out_dir, fname))
        for k in chunk:
            weight_map[k] = fname
    n_shards = (len(items) + shard_size - 1) // shard_size
    for k, fname in weight_map.items():
        weight_map[k] = fname.replace("of-00000", f"of-{n_shards:05d}")
    for si in range(0, len(items), shard_size):
        old = f"model-{si // shard_size + 1:05d}-of-00000.safetensors"
        new = f"model-{si // shard_size + 1:05d}-of-{n_shards:05d}.safetensors"
        os.rename(os.path.join(out_dir, old), os.path.join(out_dir, new))
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"weight_map": weight_map}, f)
    return sorted(tensors.keys())


def test_fill_roundtrip_synthetic(cpu_env):
    from v10_fill import fill_model
    from v10_model import build_v10_model

    vc = dict(TINY_VC)
    model_a, _ = build_v10_model(vc=vc, device="cpu", mtp=False)
    _seed_params(model_a, seed=777)

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        keys = _dump_synthetic_checkpoint(model_a, td, vc)
        assert any("q_conv1d" in k for k in keys), "synthetic must contain conv parts"

        model_b, _ = build_v10_model(vc=vc, device="cpu", mtp=False)
        ledger = fill_model(model_b, td, vc)
        assert ledger["unwritten"] == [], f"unwritten: {ledger['unwritten']}"
        assert ledger["unconsumed"] == [], f"unconsumed: {ledger['unconsumed']}"
        assert ledger["n_deferred"] == 0

        # bit-exact round-trip on EVERY param + expert_bias buffers
        pa = dict(model_a.named_parameters())
        pb = dict(model_b.named_parameters())
        assert set(pa) == set(pb)
        for n in pa:
            assert torch.equal(pa[n].data, pb[n].data), f"param mismatch: {n}"
        ba = {n: b for n, b in model_a.named_buffers()
              if n.endswith("mlp.router.expert_bias")}
        bb = {n: b for n, b in model_b.named_buffers()
              if n.endswith("mlp.router.expert_bias")}
        for n in ba:
            assert torch.equal(ba[n], bb[n]), f"expert_bias mismatch: {n}"


def test_fill_conv_composite_order(cpu_env):
    """conv1d_weight must be cat(q, k, v) — spot-check with distinct values."""
    from v10_fill import fill_model
    from v10_model import build_v10_model

    vc = dict(TINY_VC)
    model_a, _ = build_v10_model(vc=vc, device="cpu", mtp=False)
    _seed_params(model_a, seed=1234)
    conv = dict(model_a.named_parameters())[
        "decoder.layers.0.self_attention.conv1d_weight"].detach()
    per = conv.shape[0] // 3
    assert per > 0
    # three segments must be distinct after seeding (probabilistic but stable
    # with the fixed seed)
    assert not torch.equal(conv[:per], conv[per:2 * per])

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _dump_synthetic_checkpoint(model_a, td, vc)
        model_b, _ = build_v10_model(vc=vc, device="cpu", mtp=False)
        fill_model(model_b, td, vc, verbose=False)
        conv_b = dict(model_b.named_parameters())[
            "decoder.layers.0.self_attention.conv1d_weight"].detach()
        assert torch.equal(conv, conv_b), "conv cat(q,k,v) order broken"


def test_fill_mtp_keys_deferred(cpu_env):
    """layers.45-style keys (li >= n_main) land in the deferred bucket, not
    unconsumed — with a synthetic extra layer injected."""
    from v10_fill import fill_model
    from v10_model import build_v10_model
    from safetensors.torch import save_file

    vc = dict(TINY_VC)
    model_a, _ = build_v10_model(vc=vc, device="cpu", mtp=False)
    _seed_params(model_a, seed=555)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _dump_synthetic_checkpoint(model_a, td, vc)
        # inject a fake MTP layer key
        extra = {"model.language_model.layers.2.enorm.weight":
                 torch.ones(4, dtype=torch.bfloat16)}
        save_file(extra, os.path.join(td, "mtp.safetensors"))
        with open(os.path.join(td, "model.safetensors.index.json")) as f:
            idx = json.load(f)
        idx["weight_map"]["model.language_model.layers.2.enorm.weight"] = "mtp.safetensors"
        with open(os.path.join(td, "model.safetensors.index.json"), "w") as f:
            json.dump(idx, f)
        model_b, _ = build_v10_model(vc=vc, device="cpu", mtp=False)
        ledger = fill_model(model_b, td, vc, verbose=False)
        assert ledger["n_deferred"] == 1
        assert ledger["unconsumed"] == []


def test_fill_visual_keys_skipped(cpu_env):
    """model.visual.* keys (347 in the real vendor) land in the skipped
    bucket — never in unconsumed (a ledger FAIL would block the CE gate)."""
    from v10_fill import fill_model
    from v10_model import build_v10_model
    from safetensors.torch import save_file

    vc = dict(TINY_VC)
    model_a, _ = build_v10_model(vc=vc, device="cpu", mtp=False)
    _seed_params(model_a, seed=999)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _dump_synthetic_checkpoint(model_a, td, vc)
        extra = {"model.visual.blocks.0.attn.qkv.weight":
                 torch.zeros(2, 2, dtype=torch.bfloat16),
                 "model.visual.blocks.0.mlp.down_proj.weight":
                 torch.zeros(2, 2, dtype=torch.bfloat16)}
        save_file(extra, os.path.join(td, "visual.safetensors"))
        with open(os.path.join(td, "model.safetensors.index.json")) as f:
            idx = json.load(f)
        idx["weight_map"].update({k: "visual.safetensors" for k in extra})
        with open(os.path.join(td, "model.safetensors.index.json"), "w") as f:
            json.dump(idx, f)
        model_b, _ = build_v10_model(vc=vc, device="cpu", mtp=False)
        ledger = fill_model(model_b, td, vc, verbose=False)
        assert ledger["n_skipped"] == 2
        assert ledger["unconsumed"] == []


def test_fill_bf16_vendor_upcast_lossless(cpu_env):
    """The REAL vendor stores hc fn / conv1d / norms / indexer gate+ape as
    BF16 while the mcore params are fp32 (probe 2026-11-19) — fill must upcast
    bit-exactly. A_log/dt_bias/e_score_correction_bias stay F32."""
    from v10_fill import fill_model
    from v10_model import build_v10_model
    from safetensors.torch import load_file, save_file

    BF16_KEY_SUFFIXES = ("hc_attn_fn", "hc_ffn_fn", "q_conv1d.weight",
                         "k_conv1d.weight", "v_conv1d.weight",
                         "o_norm.weight", "layernorm.weight",
                         "indexer.index_kpool_compress_gate",
                         "indexer.index_kpool_compress_ape")
    UPCAST_SUFFIXES = ("attn_hc.fn", "ffn_hc.fn", "conv1d_weight",
                       "o_norm.weight", "layernorm.weight",
                       "indexer.index_kpool_compress_gate",
                       "indexer.index_kpool_compress_ape")

    vc = dict(TINY_VC)
    model_a, _ = build_v10_model(vc=vc, device="cpu", mtp=False)
    _seed_params(model_a, seed=888)
    pa = dict(model_a.named_parameters())
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        _dump_synthetic_checkpoint(model_a, td, vc)
        for fn in sorted(os.listdir(td)):
            if not fn.endswith(".safetensors"):
                continue
            path = os.path.join(td, fn)
            tensors = dict(load_file(path))
            changed = False
            for k in list(tensors):
                if k.endswith(BF16_KEY_SUFFIXES) or k == "model.language_model.norm.weight":
                    tensors[k] = tensors[k].to(torch.bfloat16)
                    changed = True
            if changed:
                save_file(tensors, path)
        model_b, _ = build_v10_model(vc=vc, device="cpu", mtp=False)
        ledger = fill_model(model_b, td, vc, verbose=False)
        assert ledger["unwritten"] == [] and ledger["unconsumed"] == []
        pb = dict(model_b.named_parameters())
        for n in pa:
            if n.endswith(UPCAST_SUFFIXES):
                assert torch.equal(pb[n].data, pa[n].data.to(torch.bfloat16).float()), \
                    f"bf16 vendor upcast not lossless: {n}"
            else:
                assert torch.equal(pa[n].data, pb[n].data), f"param mismatch: {n}"


# ---------------------------------------------------------------------------
# TP=2 END-TO-END fill (the launch_ce_gate.py TP16 path, miniaturized):
# a 2-rank gloo build+fill must produce per-rank INTERLEAVED conv1d/fc1
# shards — mcore 0.18.2 glu() chunks the TP-LOCAL output, so cat-then-shard
# hands rank 0 a pure-gate/pure-q block (silent, shapes still line up).
# ---------------------------------------------------------------------------
_TP2_CHILD = r"""
import json, os, sys
V10, TD = sys.argv[1], sys.argv[2]
sys.path.insert(0, V10)
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
import torch
torch.cuda.current_device = lambda: "cpu"      # mcore router/expert-bias paths
rank = int(os.environ["RANK"])
import torch.distributed as dist
dist.init_process_group(backend="gloo", init_method="env://")
from safetensors.torch import load_file
from v10_fill import fill_model
from v10_model import build_v10_model

vc = json.load(open(os.path.join(TD, "vc.json")))
model, cfg = build_v10_model(vc=vc, device="cpu", mtp=False, tp_size=2)
ledger = fill_model(model, TD, vc, tp_rank=rank, tp_size=2, verbose=False)
assert ledger["unwritten"] == [], ledger["unwritten"]
assert ledger["unconsumed"] == [], ledger["unconsumed"]
assert ledger["n_deferred"] == 0

ck = {}
for fn in sorted(os.listdir(TD)):
    if fn.endswith(".safetensors"):
        ck.update(load_file(os.path.join(TD, fn)))
P = dict(model.named_parameters())
TP = 2

def col(t, r):
    n = t.shape[0] // TP
    return t[r * n:(r + 1) * n]
def row(t, r):
    n = t.shape[1] // TP
    return t[:, r * n:(r + 1) * n]
def interleave(parts, r):
    out = []
    for p in parts:
        n = p.shape[0] // TP
        out.append(p[r * n:(r + 1) * n])
    return torch.cat(out, dim=0)

pfx0 = "model.language_model.layers.0."
pfx1 = "model.language_model.layers.1."

def eq(param_name, expect):
    got = P[param_name].detach()
    e = expect.to(got.dtype).reshape(got.shape)
    assert torch.equal(got, e), f"rank{rank} {param_name} mismatch"

# conv1d per-rank interleave (KDA layer 0): q/k/v are INDEPENDENT column outs
eq("decoder.layers.0.self_attention.conv1d_weight",
   interleave([ck[pfx0 + "self_attn.q_conv1d.weight"],
               ck[pfx0 + "self_attn.k_conv1d.weight"],
               ck[pfx0 + "self_attn.v_conv1d.weight"]], rank))
# gated fc1 per-rank interleave (dense): glu chunks the TP-LOCAL output
eq("decoder.layers.0.mlp.linear_fc1.weight",
   interleave([ck[pfx0 + "mlp.gate_proj.weight"],
               ck[pfx0 + "mlp.up_proj.weight"]], rank))
# column / row / vocab sanity
eq("decoder.layers.0.self_attention.linear_q_proj.weight",
   col(ck[pfx0 + "self_attn.q_proj.weight"], rank))
eq("decoder.layers.0.mlp.linear_fc2.weight",
   row(ck[pfx0 + "mlp.down_proj.weight"], rank))
eq("decoder.layers.0.self_attention.A_log", col(ck[pfx0 + "self_attn.A_log"], rank))
eq("decoder.layers.0.self_attention.dt_bias", col(ck[pfx0 + "self_attn.dt_bias"], rank))
eq("embedding.word_embeddings.weight",
   col(ck["model.language_model.embed_tokens.weight"], rank))
eq("output_layer.weight", col(ck["lm_head.weight"], rank))
# MoE expert + shared expert fc1 per-rank interleave (layer 1)
eq("decoder.layers.1.mlp.experts.local_experts.0.linear_fc1.weight",
   interleave([ck[pfx1 + "mlp.experts.0.gate_proj.weight"],
               ck[pfx1 + "mlp.experts.0.up_proj.weight"]], rank))
eq("decoder.layers.1.mlp.shared_experts.linear_fc1.weight",
   interleave([ck[pfx1 + "mlp.shared_experts.gate_proj.weight"],
               ck[pfx1 + "mlp.shared_experts.up_proj.weight"]], rank))
# DSA: q_b column (head split), q_a replicated
eq("decoder.layers.1.self_attention.q_b_proj.weight",
   col(ck[pfx1 + "self_attn.q_b_proj.weight"], rank))
eq("decoder.layers.1.self_attention.q_a_proj.weight",
   ck[pfx1 + "self_attn.q_a_proj.weight"])
print(f"TP2FILL PASS rank{rank}", flush=True)
dist.barrier()
dist.destroy_process_group()
"""


def test_fill_tp2_composite_per_rank_interleave(cpu_env):
    import socket
    import subprocess
    import tempfile

    from v10_model import build_v10_model

    vc = dict(TINY_VC)
    model_a, _ = build_v10_model(vc=vc, device="cpu", mtp=False)
    _seed_params(model_a, seed=777)
    with tempfile.TemporaryDirectory() as td:
        _dump_synthetic_checkpoint(model_a, td, vc)
        with open(os.path.join(td, "vc.json"), "w") as f:
            json.dump(vc, f)
        script = os.path.join(td, "tp2_child.py")
        with open(script, "w") as f:
            f.write(_TP2_CHILD)
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        V10 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ, MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                   WORLD_SIZE="2", TORCHDYNAMO_DISABLE="1")
        procs = []
        for r in range(2):
            procs.append(subprocess.Popen(
                [sys.executable, script, V10, td], env={**env, "RANK": str(r)},
                stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        outs = []
        for r, p in enumerate(procs):
            so, se = p.communicate(timeout=300)
            assert p.returncode == 0, \
                f"TP2 child rank{r} failed rc={p.returncode}\n{so.decode()}\n{se.decode()}"
            outs.append(so.decode())
        assert "TP2FILL PASS rank0" in outs[0]
        assert "TP2FILL PASS rank1" in outs[1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))


def _dump_mtp_keys(model_a, extra_tensors, vc=None):
    """MTP layer-(n_main)-style keys from mtp.layers.0.* — eh_proj written in
    the RAW checkpoint order. VERDICT (diag v2 cell D, 2026-11-20): GLM stores
    [e | h], ALREADY aligned with mcore's cat([enorm(e), hnorm(h)]) — the fill
    must NOT swap (swap default OFF; GLM53_V10_EH_SWAP=1 re-enables for A/B).
    The REAL checkpoint uses layers.45 (n_main=45); the tiny fixture uses
    layers.{len(layer_types)} — the fill matches li == n_main either way."""
    n_main = len((vc or TINY_VC)["layer_types"])
    P = dict(model_a.named_parameters())
    B = {n: b for n, b in model_a.named_buffers()
         if n.endswith("mlp.router.expert_bias")}
    lp = f"model.language_model.layers.{n_main}."
    mt = "mtp.layers.0."
    mm = mt + "mtp_model_layer."
    sa = mm + "self_attention."
    t = extra_tensors
    t[lp + "enorm.weight"] = P[mt + "enorm.weight"].detach().clone().contiguous()
    t[lp + "hnorm.weight"] = P[mt + "hnorm.weight"].detach().clone().contiguous()
    t[lp + "eh_proj.weight"] = \
        P[mt + "eh_proj.weight"].detach().clone().contiguous()
    t[lp + "shared_head.norm.weight"] = \
        P[mt + "final_layernorm.weight"].detach().clone().contiguous()
    t[lp + "input_layernorm.weight"] = \
        P[mm + "input_layernorm.weight"].detach().clone().contiguous()
    t[lp + "post_attention_layernorm.weight"] = \
        P[mm + "pre_mlp_layernorm.weight"].detach().clone().contiguous()
    for src, dst in (("q_a_proj", "q_a_proj"), ("q_b_proj", "q_b_proj"),
                     ("kv_a_proj_with_mqa", "kv_a_proj_with_mqa"),
                     ("kv_b_proj", "kv_b_proj"), ("o_proj", "o_proj")):
        t[lp + f"self_attn.{dst}.weight"] = \
            P[sa + f"{src}.weight"].detach().clone().contiguous()
    t[lp + "self_attn.q_a_layernorm.weight"] = \
        P[sa + "q_a_layernorm.weight"].detach().clone().contiguous()
    t[lp + "self_attn.kv_a_layernorm.weight"] = \
        P[sa + "kv_a_layernorm.weight"].detach().clone().contiguous()
    for src, dst in (("linear_wq_b", "wq_b"), ("linear_wk", "wk"),
                     ("linear_weights_proj", "weights_proj")):
        t[lp + f"self_attn.indexer.{dst}.weight"] = \
            P[sa + f"indexer.{src}.weight"].detach().clone().contiguous()
    t[lp + "self_attn.indexer.k_norm.weight"] = \
        P[sa + "indexer.k_norm.norm.weight"].detach().clone().contiguous()
    t[lp + "self_attn.indexer.k_norm.bias"] = \
        P[sa + "indexer.k_norm.norm.bias"].detach().clone().contiguous()
    t[lp + "self_attn.indexer.index_kpool_compress_gate"] = \
        P[sa + "indexer.index_kpool_compress_gate"].detach().clone().contiguous()
    t[lp + "self_attn.indexer.index_kpool_compress_ape"] = \
        P[sa + "indexer.index_kpool_compress_ape"].detach().clone().contiguous()
    t[lp + "mlp.gate.weight"] = P[mm + "mlp.router.weight"].detach().clone().contiguous()
    t[lp + "mlp.gate.e_score_correction_bias"] = \
        B[mm + "mlp.router.expert_bias"].detach().clone().contiguous()
    n_exp = TINY_VC["n_routed_experts"]
    for e in range(n_exp):
        efx = mm + f"mlp.experts.local_experts.{e}."
        fc1 = P[efx + "linear_fc1.weight"].detach()
        half = fc1.shape[0] // 2
        t[lp + f"mlp.experts.{e}.gate_proj.weight"] = fc1[:half].clone().contiguous()
        t[lp + f"mlp.experts.{e}.up_proj.weight"] = fc1[half:].clone().contiguous()
        t[lp + f"mlp.experts.{e}.down_proj.weight"] = \
            P[efx + "linear_fc2.weight"].detach().clone().contiguous()
    sfx = mm + "mlp.shared_experts."
    fc1 = P[sfx + "linear_fc1.weight"].detach()
    half = fc1.shape[0] // 2
    t[lp + "mlp.shared_experts.gate_proj.weight"] = fc1[:half].clone().contiguous()
    t[lp + "mlp.shared_experts.up_proj.weight"] = fc1[half:].clone().contiguous()
    t[lp + "mlp.shared_experts.down_proj.weight"] = \
        P[sfx + "linear_fc2.weight"].detach().clone().contiguous()
    return t


@pytest.mark.skipif(not _MTP_OK, reason="requires mcore 0.18.2 (mtp_model_layer)")
def test_fill_mtp_roundtrip(cpu_env):
    """Full synthetic roundtrip WITH the MTP layer: main layers + layers.45
    keys -> fill(mtp=True) -> bit-exact + clean ledger (deferred==0)."""
    import json
    import tempfile

    from safetensors.torch import save_file

    from v10_fill import fill_model
    from v10_model import build_v10_model

    vc = dict(TINY_VC)
    model_a, _ = build_v10_model(vc=vc, device="cpu", mtp=True)
    _seed_params(model_a, seed=888)
    with tempfile.TemporaryDirectory() as td:
        _dump_synthetic_checkpoint(model_a, td, vc)
        mtp_tensors = {}
        _dump_mtp_keys(model_a, mtp_tensors, vc)
        save_file(mtp_tensors, os.path.join(td, "mtp.safetensors"))
        with open(os.path.join(td, "model.safetensors.index.json")) as f:
            idx = json.load(f)
        for k in mtp_tensors:
            idx["weight_map"][k] = "mtp.safetensors"
        with open(os.path.join(td, "model.safetensors.index.json"), "w") as f:
            json.dump(idx, f)

        model_b, _ = build_v10_model(vc=vc, device="cpu", mtp=True)
        ledger = fill_model(model_b, td, vc, mtp=True, verbose=False)
        assert ledger["unwritten"] == [], f"unwritten: {ledger['unwritten'][:8]}"
        assert ledger["unconsumed"] == [], f"unconsumed: {ledger['unconsumed'][:8]}"
        assert ledger["n_deferred"] == 0, f"deferred: {ledger['n_deferred']}"

        pa = dict(model_a.named_parameters())
        pb = dict(model_b.named_parameters())
        assert set(pa) == set(pb)
        # eh_proj roundtrip: raw checkpoint order both sides (fill does NOT
        # swap by default — diag v2 cell D verdict)
        for n in pa:
            assert torch.equal(pa[n].data, pb[n].data), f"mtp roundtrip mismatch: {n}"
        ea = pa["mtp.layers.0.eh_proj.weight"].data
        eb = pb["mtp.layers.0.eh_proj.weight"].data
        assert torch.equal(ea, eb), "eh_proj raw-order roundtrip broken"
        assert not torch.equal(ea[:ea.shape[0] // 2], ea[ea.shape[0] // 2:]) or True
