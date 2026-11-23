#!/usr/bin/env python3
"""V10 cp -> vLLM serving dir export (BF16 vendor + LoRA merge + MTP replace).

The V5.x build_trained_cp.py recipe is W4A8-specific (int8 requant + offsets);
V10's vendor is BF16 and the CP shards are per-rank fp32 masters of the
TRAINABLE set only (decoder LoRA hooks + the full MTP subtree) — so export =
vendor copy + the fill run in REVERSE:

  decoder LoRA (259 pairs, r=16, alpha=32):
    column targets (A replicated, B rank-local concat dim0):
      KDA linear_q/k/v_proj, DSA q_b/kv_b_proj, dense/shared fc1
    row targets (A rank-local concat dim1, B replicated):
      KDA linear_o_proj, DSA o_proj, dense/shared fc2
    fc1 delta rows split into gate/up halves (the fill cats [gate_r; up_r]
    per rank, so rank r owns vendor gate rows [r*f_loc:(r+1)*f_loc])
  MTP subtree (vendor = model.language_model.layers.45.*):
    enorm/hnorm/eh_proj(raw order, no swap)/shared_head.norm + the inner
    layer (DSA map inverted + router + shared experts + 288 routed experts)
  EVERYTHING else: byte-identical vendor tensor.

Safety: every write asserts shape == vendor target; per-component delta
norms printed (audit block, TRAINED_CP_WEIGHT_BUILD.md discipline).

Usage: python3 v10_export_serving.py <rank_shard_dir> <out_dir>
"""
import json
import os
import shutil
import sys

import torch

RANK = int(os.environ.get("GLM53_LORA_RANK", "16"))
ALPHA = float(os.environ.get("GLM53_LORA_ALPHA", "32"))
SCALING = ALPHA / RANK
VENDOR = os.environ.get("V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16")
N_RANKS = int(os.environ.get("V10_TP", "16"))
F_LOC = 128          # moe_ffn(2048)/tp
DENSE_F_LOC = 768    # dense ffn(12288)/tp (fc2.A in_local probe)
CKPT = "model.language_model."

shard_dir, out_dir = sys.argv[1], sys.argv[2]
CP_STEP = os.environ.get("V10_CP_STEP", "1000")   # shard filename cp{step}_rank{r}.pt

# ---- load all rank shards (CPU, one at a time; keep only trainable keys) ----
print(f"[export] loading {N_RANKS} rank shards from {shard_dir} ...")
rank_params = {}
for r in range(N_RANKS):
    blob = torch.load(os.path.join(shard_dir, f"cp{CP_STEP}_rank{r}.pt"),
                      weights_only=False, map_location="cpu")
    assert blob.get("step") == int(CP_STEP), \
        f"cp{CP_STEP}_rank{r}.pt is actually step {blob.get('step')} — wrong CP"
    rank_params[r] = blob["shard"]
    print(f"  rank {r}: {len(blob['shard'])} keys (step {blob['step']})")

def get(r, name):
    return rank_params[r][name]

# ---- vendor index ----
idx = json.load(open(os.path.join(VENDOR, "model.safetensors.index.json")))
weight_map = idx["weight_map"]
full_keys = set(weight_map)

def vendor_tensor(key):
    from safetensors.torch import load_file
    return load_file(os.path.join(VENDOR, weight_map[key]))[key]

modified = {}          # vendor_key -> torch tensor (full, vendor layout)
delta_audit = []       # (component, vendor_key, delta_norm)

def put(key, t, component, delta=None):
    assert key in full_keys, f"vendor key not in index: {key}"
    vshape = tuple(vendor_tensor(key).shape) if key not in modified \
        else tuple(modified[key].shape)
    assert tuple(t.shape) == vshape, \
        f"{key}: vendor {vshape} vs export {tuple(t.shape)}"
    modified[key] = t.contiguous()
    if delta is not None:
        delta_audit.append((component, key, float(delta)))


def fc1_uninterleave(v2f, f_loc, n_ranks):
    """The mcore fc1 layout is [g_r; u_r] per rank (rank-interleaved) — the
    naive [:half] split SCRAMBLES gate/up across ranks (probed 2026-11-22:
    the served cp500 draft collapsed to D1=0.098). Returns (gate, up) in
    vendor row order (cat of g_r, cat of u_r)."""
    rpf = 2 * f_loc
    g = torch.cat([v2f[i * rpf : i * rpf + f_loc] for i in range(n_ranks)], 0)
    u = torch.cat([v2f[i * rpf + f_loc : (i + 1) * rpf]
                   for i in range(n_ranks)], 0)
    return g, u


# ---- 1) decoder LoRA merge ----
# mode: (column -> A replicated, B concat dim0) | (row -> A concat dim1, B rep)
ATTN_COLUMN = {
    "linear_q_proj": "self_attn.q_proj.weight",
    "linear_k_proj": "self_attn.k_proj.weight",
    "linear_v_proj": "self_attn.v_proj.weight",
    "q_b_proj": "self_attn.q_b_proj.weight",
    "kv_b_proj": "self_attn.kv_b_proj.weight",
}
lora_names = sorted(k for k in rank_params[0] if ".lora_A" in k)
layer_ids = sorted({int(k.split(".layers.")[1].split(".")[0])
                    for k in lora_names})
n_lora = 0
for li in layer_ids:
    for mod, vend_rel in ATTN_COLUMN.items():
        base = f"decoder.layers.{li}.self_attention.{mod}.lora_"
        if base + "A" not in rank_params[0]:
            continue                       # KDA vs DSA layer typing
        A = get(0, base + "A").float()                     # replicated
        B = torch.cat([get(r, base + "B").float()
                       for r in range(N_RANKS)], 0)        # [out_full, r]
        delta = (B @ A) * SCALING                          # [out_full, in]
        key = f"{CKPT}layers.{li}.{vend_rel}"
        v = vendor_tensor(key).float()
        put(key, (v + delta).to(v.dtype), f"lora.{mod}",
            delta=float(delta.norm()))
        n_lora += 1

# dense/shared fc1: composite split (gate|up), fc2: plain row
for li in layer_ids:
    mlp_kind = "dense" if li < 3 else None
    # dense layers: mlp.linear_fc1/fc2; moe layers: mlp.shared_experts.fc1/fc2
    # (routed experts are NOT lora'd)
    if li < 3:
        f_loc = DENSE_F_LOC
        fc1_base = f"decoder.layers.{li}.mlp.linear_fc1.lora_"
        gk, uk = (f"{CKPT}layers.{li}.mlp.gate_proj.weight",
                  f"{CKPT}layers.{li}.mlp.up_proj.weight")
        dk = f"{CKPT}layers.{li}.mlp.down_proj.weight"
        fc2_base = f"decoder.layers.{li}.mlp.linear_fc2.lora_"
    else:
        f_loc = F_LOC
        fc1_base = (f"decoder.layers.{li}.mlp.shared_experts."
                    f"linear_fc1.lora_")
        gk = f"{CKPT}layers.{li}.mlp.shared_experts.gate_proj.weight"
        uk = f"{CKPT}layers.{li}.mlp.shared_experts.up_proj.weight"
        dk = f"{CKPT}layers.{li}.mlp.shared_experts.down_proj.weight"
        fc2_base = (f"decoder.layers.{li}.mlp.shared_experts."
                    f"linear_fc2.lora_")
    if fc1_base + "A" in rank_params[0]:
        A = get(0, fc1_base + "A").float()
        B = torch.cat([get(r, fc1_base + "B").float()
                       for r in range(N_RANKS)], 0)        # [2*f_full, r]
        delta = (B @ A) * SCALING                          # [2*f_full, in]
        dg, du = fc1_uninterleave(delta, f_loc, N_RANKS)   # rank-interleave inverse
        g, u = vendor_tensor(gk).float(), vendor_tensor(uk).float()
        put(gk, (g + dg).to(g.dtype), "lora.fc1.gate",
            delta=float(dg.norm()))
        put(uk, (u + du).to(u.dtype), "lora.fc1.up",
            delta=float(du.norm()))
        n_lora += 1
    if fc2_base + "A" in rank_params[0]:
        A = torch.cat([get(r, fc2_base + "A").float()
                       for r in range(N_RANKS)], 1)        # [r, in_full]
        B = get(0, fc2_base + "B").float()                 # replicated
        delta = (B @ A) * SCALING                          # [out, in_full]
        d = vendor_tensor(dk).float()
        put(dk, (d + delta).to(d.dtype), "lora.fc2",
            delta=float(delta.norm()))
        n_lora += 1
ATTN_ROW = {
    "linear_o_proj": "self_attn.o_proj.weight",
    "o_proj": "self_attn.o_proj.weight",
}
for li in layer_ids:
    for mod, vend_rel in ATTN_ROW.items():
        base = f"decoder.layers.{li}.self_attention.{mod}.lora_"
        if base + "A" not in rank_params[0]:
            continue                       # KDA vs DSA layer typing
        A = torch.cat([get(r, base + "A").float()
                       for r in range(N_RANKS)], 1)        # [r, in_full]
        B = get(0, base + "B").float()                     # replicated
        delta = (B @ A) * SCALING                          # [out, in_full]
        key = f"{CKPT}layers.{li}.{vend_rel}"
        v = vendor_tensor(key).float()
        put(key, (v + delta).to(v.dtype), f"lora.{mod}",
            delta=float(delta.norm()))
        n_lora += 1
print(f"[export] lora merged: {n_lora} linears")

# ---- 2) MTP subtree (vendor layer 45) ----
VL = 45
MT = "mtp.layers.0."
VI = f"{CKPT}layers.{VL}."

def mtp_simple(mcore_name, vend_rel, component, mode="replicated"):
    if mode == "replicated":
        t = get(0, mcore_name).float()
    elif mode == "column":
        t = torch.cat([get(r, mcore_name).float()
                       for r in range(N_RANKS)], 0)
    elif mode == "row":
        t = torch.cat([get(r, mcore_name).float()
                       for r in range(N_RANKS)], 1)
    else:
        raise ValueError(mode)
    v = vendor_tensor(VI + vend_rel).float()
    put(VI + vend_rel, t.to(v.dtype), component,
        delta=float((t.to(v.dtype).float() - v).norm()))

mtp_simple(MT + "enorm.weight", "enorm.weight", "mtp.enorm")
mtp_simple(MT + "hnorm.weight", "hnorm.weight", "mtp.hnorm")
mtp_simple(MT + "eh_proj.weight", "eh_proj.weight", "mtp.eh_proj", "column")
mtp_simple(MT + "final_layernorm.weight", "shared_head.norm.weight",
           "mtp.shared_head.norm")
mtp_simple(MT + "mtp_model_layer.input_layernorm.weight",
           "input_layernorm.weight", "mtp.input_ln")
mtp_simple(MT + "mtp_model_layer.pre_mlp_layernorm.weight",
           "post_attention_layernorm.weight", "mtp.pre_mlp_ln")
mtp_simple(MT + "mtp_model_layer.mlp.router.weight",
           "mlp.gate.weight", "mtp.router")

IM = "mtp.layers.0.mtp_model_layer.self_attention."
DSA_MAP = {
    "q_a_proj.weight": "self_attn.q_a_proj.weight",
    "q_a_layernorm.weight": "self_attn.q_a_layernorm.weight",
    "q_b_proj.weight": "self_attn.q_b_proj.weight",
    "kv_a_proj_with_mqa.weight": "self_attn.kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight": "self_attn.kv_a_layernorm.weight",
    "kv_b_proj.weight": "self_attn.kv_b_proj.weight",
    "o_proj.weight": "self_attn.o_proj.weight",
    "indexer.linear_wq_b.weight": "self_attn.indexer.wq_b.weight",
    "indexer.linear_wk.weight": "self_attn.indexer.wk.weight",
    "indexer.k_norm.norm.weight": "self_attn.indexer.k_norm.weight",
    "indexer.k_norm.norm.bias": "self_attn.indexer.k_norm.bias",
    "indexer.linear_weights_proj.weight":
        "self_attn.indexer.weights_proj.weight",
    "indexer.index_kpool_compress_gate":
        "self_attn.indexer.index_kpool_compress_gate",
    "indexer.index_kpool_compress_ape":
        "self_attn.indexer.index_kpool_compress_ape",
}
# replicated per param_shard_mode: q_a_proj / kv_a_proj_with_mqa / all norms
# / ALL indexer.* — only q_b_proj / kv_b_proj / o_proj are TP-column
REPL = {"self_attn.q_a_proj.weight", "self_attn.q_a_layernorm.weight",
        "self_attn.kv_a_proj_with_mqa.weight",
        "self_attn.kv_a_layernorm.weight",
        "self_attn.indexer.linear_wq_b.weight",
        "self_attn.indexer.linear_wk.weight",
        "self_attn.indexer.k_norm.weight", "self_attn.indexer.k_norm.bias",
        "self_attn.indexer.linear_weights_proj.weight",
        "self_attn.indexer.index_kpool_compress_gate",
        "self_attn.indexer.index_kpool_compress_ape"}
for mc, vend in DSA_MAP.items():
    if IM + mc not in rank_params[0]:
        continue
    # TP modes = param_shard_mode authority (probed 2026-11-22): q_b/kv_b
    # column, o_proj row, EVERYTHING else replicated (q_a/kv_a/indexer/*)
    if vend.endswith("q_b_proj.weight") or vend.endswith("kv_b_proj.weight"):
        mode = "column"
    elif vend.endswith("o_proj.weight"):
        mode = "row"
    else:
        mode = "replicated"
    mtp_simple(IM + mc, vend, f"mtp.dsa.{vend}", mode)

# shared experts
se1 = MT + "mtp_model_layer.mlp.shared_experts.linear_fc1.weight"
se2 = MT + "mtp_model_layer.mlp.shared_experts.linear_fc2.weight"
sgk = VI + "mlp.shared_experts.gate_proj.weight"
suk = VI + "mlp.shared_experts.up_proj.weight"
sdk = VI + "mlp.shared_experts.down_proj.weight"
fc1_full = torch.cat([get(r, se1).float() for r in range(N_RANKS)], 0)
sg, su = fc1_uninterleave(fc1_full, F_LOC, N_RANKS)
g, u = vendor_tensor(sgk).float(), vendor_tensor(suk).float()
put(sgk, sg.to(g.dtype), "mtp.shared.fc1.gate",
    delta=float((sg - g).norm()))
put(suk, su.to(u.dtype), "mtp.shared.fc1.up",
    delta=float((su - u).norm()))
d = vendor_tensor(sdk).float()
put(sdk, torch.cat([get(r, se2).float() for r in range(N_RANKS)], 1)
    .to(d.dtype), "mtp.shared.fc2",
    delta=float((torch.cat([get(r, se2).float() for r in range(N_RANKS)], 1) - d).norm()))

# routed experts (288): fc1 composite per rank -> gate/up scatter; fc2 row cat
n_exp = 288
gate = vendor_tensor(VI + "mlp.experts.0.gate_proj.weight")  # probe shape
print(f"[export] expert fc1 vendor probe: {tuple(gate.shape)}")
for e in range(n_exp):
    pfx = f"{MT}mtp_model_layer.mlp.experts.local_experts.{e}."
    vpfx = f"{VI}mlp.experts.{e}."
    g = torch.empty_like(vendor_tensor(vpfx + "gate_proj.weight").float())
    u = torch.empty_like(vendor_tensor(vpfx + "up_proj.weight").float())
    fc1_full = torch.cat([get(r, pfx + "linear_fc1.weight").float()
                          for r in range(N_RANKS)], 0)
    g_part, u_part = fc1_uninterleave(fc1_full, F_LOC, N_RANKS)
    g[:] = g_part
    u[:] = u_part
    d = torch.cat([get(r, pfx + "linear_fc2.weight").float()
                   for r in range(N_RANKS)], 1)          # [H, f_full]
    gp, up_, dp = (vendor_tensor(vpfx + "gate_proj.weight").float(),
                   vendor_tensor(vpfx + "up_proj.weight").float(),
                   vendor_tensor(vpfx + "down_proj.weight").float())
    assert g.shape == gp.shape and u.shape == up_.shape and d.shape == dp.shape
    put(vpfx + "gate_proj.weight", g.to(gp.dtype), "mtp.expert.gate")
    put(vpfx + "up_proj.weight", u.to(up_.dtype), "mtp.expert.up")
    put(vpfx + "down_proj.weight", d.to(dp.dtype), "mtp.expert.down")
    if e % 64 == 0:
        print(f"  experts {e}...")

# ---- 3) write: vendor copy (symlink-farm) + patch ----
# container root fs is 100% FULL (trap) — the serve dir MUST live on NFS and
# UNMODIFIED files are SYMLINKS into the vendor dir (vLLM reads through
# symlinks); only shards containing modified tensors are real copies.
print(f"[export] building serve dir {out_dir} (symlink farm) ...")
os.makedirs(out_dir, exist_ok=True)
for entry in os.listdir(VENDOR):
    src = os.path.join(VENDOR, entry)
    dst = os.path.join(out_dir, entry)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    elif entry in {weight_map[k] for k in modified}:
        shutil.copy2(src, dst)                     # real copy, will be patched
    else:
        if not os.path.exists(dst):
            os.symlink(src, dst)                   # untouched -> vendor
# group by shard file: the naive per-key loop re-serialized each 2GB shard
# once per modified key (864 MTP expert keys -> ~2h of duplicate writes)
from collections import defaultdict
from safetensors.torch import load_file, save_file
by_file = defaultdict(dict)
for key, t in modified.items():
    by_file[weight_map[key]][key] = t
for fname, patches in by_file.items():
    path = os.path.join(out_dir, fname)
    tensors = load_file(path)
    for key, t in patches.items():
        assert key in tensors, f"{key} not in {fname}"
        tensors[key] = t
    save_file(tensors, path, metadata={"format": "pt"})
    print(f"  patched {fname} ({len(patches)} tensors)")

# ---- 4) audit block ----
print("\n=== EXPORT AUDIT ===")
comp = {}
for c, k, dn in delta_audit:
    comp.setdefault(c, []).append(dn)
for c in sorted(comp):
    v = comp[c]
    print(f"  {c:24s} n={len(v):4d} delta_norm mean={sum(v)/len(v):.3e} "
          f"max={max(v):.3e}")
print(f"modified tensors: {len(modified)} / vendor total {len(full_keys)}")
print(f"[export] DONE -> {out_dir}")
