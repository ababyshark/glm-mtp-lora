"""V10 fill — GLM-5.3-Flash-BF16 checkpoint → mcore V10 model (plain copy,
ZERO dequant math; the FP8 weight_scale_inv variant is out of scope for
training per iron rule 2026-11-19).

Coverage ledger (V8 discipline): every mcore param/buffer written exactly
once, every non-visual checkpoint key consumed exactly once. MTP keys
(layers.45.*) are bucketed 'deferred' while mtp=False (Phase 2 wires them).

Composite transforms (per-rank interleave — shard EACH part, then cat):
- self_attention.conv1d_weight = cat(q_conv1d, k_conv1d, v_conv1d) after each
  part's column shard (the forward consumes cat([q,k,v]) of three INDEPENDENT
  ColumnParallel outputs — cat-then-shard would hand rank 0 a pure-q block)
- linear_fc1 = cat(gate_proj, up_proj) after each part's column shard (mcore
  0.18.2 glu() chunks the TP-LOCAL output: torch.chunk(x, 2, dim=-1) — so
  rank r's fc1 must be [gate[r*f/tp:(r+1)*f/tp]; up[...]]; cat-then-shard
  would hand rank 0 a pure-gate block)
- router.expert_bias buffer = e_score_correction_bias (fp32, no-grad)
- dtype policy: assign casts to the TARGET param dtype — bf16 ckpt -> fp32
  params (A_log/dt_bias/hc fn+base+scale/o_norm/norms/ape/conv) is a lossless
  upcast; bf16->bf16 / fp32->fp32 are bit-exact copies.

TP=1: shard_slice is a no-op — both paths degenerate to plain cat.
"""
import json
import os
import struct

import numpy as np
import torch

CKPT = "model.language_model."
CONSUMED_VIA_COMPOSITE = "__composite__"

# safetensors dtype string -> torch dtype (numpy has no bf16; read as raw
# uint8 bytes and reinterpret via torch view)
_TORCH_OF = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16,
             "F64": torch.float64, "I64": torch.int64, "I32": torch.int32,
             "I8": torch.int8, "U8": torch.uint8, "BOOL": torch.bool}
_ITEMSIZE = {"BF16": 2, "F32": 4, "F16": 2, "F64": 8, "I64": 8, "I32": 4,
             "I8": 1, "U8": 1, "BOOL": 1}


class SlicedCheckpoint:
    """Byte-range checkpoint reader: each rank reads ONLY its TP slice of
    every tensor (dim0 'column'/'vocab' slices are contiguous byte ranges;
    'row' slices are strided row groups; 'replicated'/TP=1 read full).

    Cuts the fill IO by TP× at TP16 (~599GB/rank of full-tensor reads ->
    ~37GB/rank). The naive get_tensor-per-key path reads every FULL tensor on
    every rank (measured 20-30MB/s per rank -> 6-8h per gate attempt).
    slice_tensor ≡ shard_slice(full_tensor) bit-exactly (unit-tested); the
    sharding MATH is untouched — only the IO path changed.
    """

    def __init__(self, vendor_dir, weight_map):
        self.vendor_dir = vendor_dir
        self.weight_map = weight_map
        self._mmaps = {}
        self._headers = {}
        self._bases = {}          # data-section base = 8 + header_len

    def _header(self, fname):
        hdr = self._headers.get(fname)
        if hdr is None:
            with open(os.path.join(self.vendor_dir, fname), "rb") as f:
                (n,) = struct.unpack("<Q", f.read(8))
                hdr = json.loads(f.read(n))
            # safetensors spec: data_offsets are RELATIVE TO THE DATA SECTION
            # (after 8-byte len + JSON header) — the mmap slices must add the
            # base or every tensor reads shifted bytes (probe: 3 dtypes all
            # BIT-EXACT False; the units test caught it)
            self._headers[fname] = hdr
            self._bases[fname] = 8 + n
        return hdr

    def _base(self, fname):
        self._header(fname)
        return self._bases[fname]

    def _mmap(self, fname):
        m = self._mmaps.get(fname)
        if m is None:
            m = np.memmap(os.path.join(self.vendor_dir, fname),
                          dtype=np.uint8, mode="r")
            self._mmaps[fname] = m
        return m

    def full(self, key):
        """Full tensor (replicated / TP=1)."""
        fname = self.weight_map[key]
        info = self._header(fname)[key]
        shape, dt = info["shape"], info["dtype"]
        base = self._base(fname)
        s, e = info["data_offsets"]
        t = torch.from_numpy(
            np.ascontiguousarray(self._mmap(fname)[base + s:base + e]))
        return t.view(_TORCH_OF[dt]).reshape(tuple(shape))

    def slice_tensor(self, key, mode, tp_rank, tp_size):
        """Exactly shard_slice(full(key), mode, tp_rank, tp_size), reading only
        the rank's bytes. 'column'/'vocab' = contiguous dim0 byte range; 'row'
        = strided dim1 row group (2-D only)."""
        if tp_size == 1 or mode == "replicated":
            return self.full(key)
        fname = self.weight_map[key]
        info = self._header(fname)[key]
        shape, dt = info["shape"], info["dtype"]
        s, _ = info["data_offsets"]
        s += self._base(fname)
        itemsize = _ITEMSIZE[dt]
        tdt = _TORCH_OF[dt]
        if mode in ("column", "vocab"):
            assert len(shape) >= 1
            per_row = 1
            for d in shape[1:]:
                per_row *= d
            n0 = shape[0] // tp_size
            b0 = s + tp_rank * n0 * per_row * itemsize
            b1 = s + (tp_rank + 1) * n0 * per_row * itemsize
            t = torch.from_numpy(
                np.ascontiguousarray(self._mmap(fname)[b0:b1])).view(tdt)
            return t.reshape(n0, *shape[1:])
        if mode == "row":
            assert len(shape) == 2, f"row slice on non-2D tensor {key} {shape}"
            n1 = shape[1] // tp_size
            row_bytes = shape[1] * itemsize
            raw = self._mmap(fname)[s:s + shape[0] * row_bytes] \
                .reshape(shape[0], row_bytes)[:, tp_rank * n1 * itemsize:
                                              (tp_rank + 1) * n1 * itemsize]
            t = torch.from_numpy(np.ascontiguousarray(raw)).view(tdt)
            return t.reshape(shape[0], n1)
        raise ValueError(mode)

    def load(self, key, mode, tp_rank=0, tp_size=1):
        """Rank-local load dispatch (the ONLY primitive fill uses)."""
        if tp_size == 1 or mode in (None, "replicated"):
            return self.full(key)
        return self.slice_tensor(key, mode, tp_rank, tp_size)


def param_shard_mode(name):
    """TP sharding mode of a V10 mcore param (0.18.2 ETP conventions;
    V8-proven: column/vocab = dim0 contiguous split, row = dim1 split).
    DSA q_a/kv_a + indexer are REPLICATED by construction (lora dims — the
    vLLM MLA TP scheme), as are hc/norms/router/expert_bias."""
    if name.startswith("embedding.word_embeddings") or name == "output_layer.weight":
        return "vocab"
    if (".attn_hc." in name or ".ffn_hc." in name
            or name.endswith("layernorm.weight")
            or name.endswith(("enorm.weight", "hnorm.weight"))
            or ".mlp.router." in name):
        return "replicated"
    if name.endswith("eh_proj.weight"):
        return "column"
    sa = ".self_attention."
    if sa in name:
        if "indexer." in name:
            return "replicated"
        if name.endswith(("q_a_proj.weight", "kv_a_proj_with_mqa.weight",
                          "q_a_layernorm.weight", "kv_a_layernorm.weight",
                          "o_norm.weight")):
            return "replicated"
        if name.endswith(("q_b_proj.weight", "kv_b_proj.weight")):
            return "column"
        if name.endswith("o_proj.weight"):
            return "row"
        # KDA linears
        if name.endswith(("linear_q_proj.weight", "linear_k_proj.weight",
                          "linear_v_proj.weight", "linear_b_proj.weight",
                          "linear_fb_proj.weight", "linear_gb_proj.weight")):
            return "column"
        if name.endswith("linear_o_proj.weight"):
            return "row"
        if name.endswith(("linear_fa_proj.weight", "linear_ga_proj.weight")):
            return "replicated"
        if name.endswith(("A_log", "dt_bias", "conv1d_weight")):
            return "column"  # TP dim0 (per-head / per-channel)
        raise ValueError(f"unknown attention param shard mode: {name}")
    if name.endswith("linear_fc1.weight"):
        return "column"
    if name.endswith("linear_fc2.weight"):
        return "row"
    raise ValueError(f"unknown shard mode for {name}")


def shard_slice(t, mode, tp_rank, tp_size):
    if tp_size == 1 or mode == "replicated":
        return t
    if mode in ("column", "vocab"):
        n0 = t.shape[0] // tp_size
        return t[tp_rank * n0:(tp_rank + 1) * n0]
    if mode == "row":
        n0 = t.shape[1] // tp_size
        return t[:, tp_rank * n0:(tp_rank + 1) * n0]
    raise ValueError(mode)


def composite_fuse(parts, mode, tp_rank, tp_size):
    """Composite fill: shard EACH part per the target's TP mode, THEN cat
    (per-rank interleave). See the module docstring — mcore 0.18.2 glu()
    chunks the TP-LOCAL fc1 output, and the KDA conv consumes cat([q,k,v]) of
    three independent ColumnParallel outputs. TP=1 / replicated: plain cat."""
    if tp_size == 1 or mode == "replicated":
        return torch.cat(parts, dim=0)
    return torch.cat([shard_slice(p, mode, tp_rank, tp_size) for p in parts],
                     dim=0)


def flush_grouped(grouped, li, lp, ckpt, targets, written, tp_rank, tp_size):
    """GLM53_V10_GROUPED=1 flush: build V10GroupedExperts w1/w2 from the
    recorded vendor expert keys. Layout (matches the SequentialMLP fill
    bit-for-bit after transpose):
      w1[e] = cat(shard(gate,column), shard(up,column)).T  -> [H, 2f_loc]
      w2[e] = shard(down,row).T                            -> [f_loc, H]
    The transpose moves the TP dim to the LAST axis, so slicing the vendor
    tensor BEFORE the transpose is bit-identical to transpose-then-slice."""
    e_map = grouped.pop((lp, li), None)
    if not e_map:
        return
    w1_t = targets[lp + "mlp.experts.w1"]
    w2_t = targets[lp + "mlp.experts.w2"]
    n_experts = w1_t.shape[0]
    assert n_experts == w2_t.shape[0] and set(e_map) == set(range(n_experts)), \
        f"grouped experts mismatch at layer {li}: {sorted(e_map)[:5]}..."
    w1 = torch.empty(w1_t.shape, dtype=w1_t.dtype)
    w2 = torch.empty(w2_t.shape, dtype=w2_t.dtype)
    for e in range(n_experts):
        assert set(e_map[e]) == {"gate_proj", "up_proj", "down_proj"}, \
            f"expert {e} keys incomplete: {sorted(e_map[e])}"
        pfx = f"{CKPT}layers.{li}.mlp.experts.{e}."
        g = ckpt.load(pfx + "gate_proj.weight", "column", tp_rank, tp_size)
        u = ckpt.load(pfx + "up_proj.weight", "column", tp_rank, tp_size)
        d = ckpt.load(pfx + "down_proj.weight", "row", tp_rank, tp_size)
        w1[e] = torch.cat([g, u], dim=0).transpose(0, 1).to(w1_t.dtype)
        w2[e] = d.transpose(0, 1).to(w2_t.dtype)
    assert tuple(w1.shape) == tuple(w1_t.shape) and \
        tuple(w2.shape) == tuple(w2_t.shape), \
        (f"grouped w1/w2 shape mismatch: param "
         f"{tuple(w1_t.shape)}/{tuple(w2_t.shape)} vs source "
         f"{tuple(w1.shape)}/{tuple(w2.shape)}")
    w1_t.data.copy_(w1)
    w2_t.data.copy_(w2)
    written[lp + "mlp.experts.w1"] = tuple(w1_t.shape)
    written[lp + "mlp.experts.w2"] = tuple(w2_t.shape)


def _load_index(vendor_dir):
    with open(os.path.join(vendor_dir, "model.safetensors.index.json")) as f:
        return json.load(f)["weight_map"]


def fill_model(model, vendor_dir, vc, mtp=False, verbose=True,
               tp_rank=0, tp_size=1):
    """Fill `model` from the Flash-BF16 checkpoint dir. Every load is
    rank-local: SlicedCheckpoint reads only the tp_rank/tp_size slice of each
    source tensor per param_shard_mode (byte-range, TP× less IO than full
    reads); composites slice each part then cat. Returns the ledger."""
    weight_map = _load_index(vendor_dir)
    ckpt = SlicedCheckpoint(vendor_dir, weight_map)

    targets = {}
    for name, p in model.named_parameters():
        targets[name] = p
    for name, b in model.named_buffers():
        if name.endswith("mlp.router.expert_bias"):
            targets[name] = b

    written, consumed, deferred, skipped = {}, set(), [], []
    # GLM53_V10_GROUPED=1: routed experts fill into V10GroupedExperts w1/w2
    grouped = {} if os.environ.get("GLM53_V10_GROUPED", "0") == "1" else None

    # bucket checkpoint keys
    top = {}      # embed / lm_head / final norm
    layers = {}   # li -> [(tail, full_key)]
    for k in weight_map:
        if k.startswith("model.visual."):
            skipped.append(k)
            continue
        if k == "lm_head.weight":          # bare top-level key (probe 2026-11-19)
            top["lm_head.weight"] = k
            continue
        assert k.startswith(CKPT), f"unexpected top-level key: {k}"
        rest = k[len(CKPT):]
        if rest.startswith("layers."):
            li_str, tail = rest[len("layers."):].split(".", 1)
            li = int(li_str)
            layers.setdefault(li, []).append((tail, k))
        else:
            top[rest] = k

    def assign(name, key=None, t=None):
        """Put one target: from a ckpt KEY (rank-local slice per the target's
        shard mode) or from an already rank-local tensor (composites)."""
        if t is None:
            mode = param_shard_mode(name)
            t = ckpt.load(key, mode, tp_rank, tp_size)
        tgt = targets[name]
        if tuple(t.shape) != tuple(tgt.shape):
            raise ValueError(
                f"fill shape mismatch {name}: param {tuple(tgt.shape)} vs "
                f"source {tuple(t.shape)} (mode={param_shard_mode(name)}, "
                f"tp={tp_rank}/{tp_size})")
        tgt.data.copy_(t.to(tgt.dtype).reshape(tgt.shape))
        written[name] = tuple(tgt.shape)

    # ---- top-level ----
    top_map = {
        "embed_tokens.weight": "embedding.word_embeddings.weight",
        "lm_head.weight": "output_layer.weight",
        "norm.weight": "decoder.final_layernorm.weight",
    }
    for rest, k in top.items():
        assert rest in top_map, f"unhandled top-level key: {k}"
        assign(top_map[rest], key=k)
        consumed.add(k)

    # ---- decoder layers ----
    layer_types = list(vc["layer_types"])
    ffn_kinds = ["dense" if t == "dense" else "moe" for t in vc["mlp_layer_types"]]
    n_main = len(layer_types)
    # truncation-aware: defer keys of layers the MODEL doesn't carry (derived
    # from the build, not vc — truncated smokes build a prefix)
    n_built = 0
    while f"decoder.layers.{n_built}.input_layernorm.weight" in targets:
        n_built += 1

    for li in sorted(layers):
        entries = layers[li]
        if li >= n_built:
            if mtp and li == n_main:
                _fill_mtp_layer(entries, li, ckpt, assign, consumed,
                                targets, written, tp_rank, tp_size,
                                grouped=grouped)
                continue
            deferred.extend(k for _, k in entries)
            continue
        is_kda = layer_types[li] == "linear_attention"
        is_moe = ffn_kinds[li] == "moe"
        lp = f"decoder.layers.{li}."
        ap = f"{lp}self_attention."
        composites = {}     # mcore_target -> list of full ckpt keys (in cat order)

        for tail, k in entries:
            if tail.startswith("self_attn."):
                rule = _kda_rule if is_kda else _dsa_rule
                rule(tail, k, li, ap, composites, assign)
            elif tail.startswith("mlp."):
                _mlp_rule(tail, k, li, lp, is_moe, composites, assign,
                          grouped=grouped)
            elif tail.startswith("hc_attn_"):
                assign(f"{lp}attn_hc.{tail[len('hc_attn_'):]}",
                       key=k)
            elif tail.startswith("hc_ffn_"):
                assign(f"{lp}ffn_hc.{tail[len('hc_ffn_'):]}",
                       key=k)
            elif tail == "input_layernorm.weight":
                assign(f"{lp}input_layernorm.weight", key=k)
            elif tail == "post_attention_layernorm.weight":
                assign(f"{lp}pre_mlp_layernorm.weight", key=k)
            else:
                raise ValueError(f"unhandled layer key: {k}")
            consumed.add(k)

        for target, keys in composites.items():
            mode = param_shard_mode(target)
            parts = [ckpt.load(kk, mode, tp_rank, tp_size) for kk in keys]
            assign(target, t=torch.cat(parts, dim=0))
            consumed.update(keys)
        if grouped:
            flush_grouped(grouped, li, lp, ckpt, targets, written,
                          tp_rank, tp_size)

    # ---- ledger ----
    ledger = dict(n_written=len(written), n_targets=len(targets),
                  n_consumed=len(consumed), n_skipped=len(skipped),
                  n_deferred=len(deferred),
                  unwritten=sorted(set(targets) - set(written)),
                  unconsumed=sorted((set(weight_map) - consumed) -
                                    set(skipped) - set(deferred)))
    if verbose:
        print(f"[v10-fill] written {len(written)}/{len(targets)} targets; "
              f"consumed {len(consumed)} ckpt keys; skipped {len(skipped)} "
              f"(visual); deferred {len(deferred)} (MTP)")
        for flag, items in (("UNWRITTEN targets", ledger["unwritten"]),
                            ("UNCONSUMED keys", ledger["unconsumed"])):
            if items:
                print(f"[v10-fill] {flag} ({len(items)}): {items[:20]}")
    return ledger


# ---------------------------------------------------------------------------
# per-family rules (register composites; members are consumed at flush)
# ---------------------------------------------------------------------------
def swap_eh_halves(w):
    """[out, 2h] with the GLM/DSv3 column order [h | e] -> mcore order [e | h]
    (mcore _concat_embeddings cats enorm(emb) FIRST — multi_token_prediction
    :935-946). Column-space op: TP column slicing is dim0 (rows) — order
    independent, safe to swap BEFORE shard_slice.
    VERIFY-AT-SERVING-ALIGNMENT: mapping §5 marks the GLM order [推断]; a wrong
    swap leaves the main head intact but the draft head garbage."""
    h = w.shape[-1] // 2
    if w.dim() != 2 or w.shape[-1] % 2:
        raise ValueError(f"eh_proj expected 2D [out, 2h], got {tuple(w.shape)}")
    return torch.cat([w[:, h:], w[:, :h]], dim=1).contiguous()


def _fill_mtp_layer(entries, li, ckpt, assign, consumed, targets, written,
                    tp_rank, tp_size, grouped=None):
    """layers.45 (MTP) -> mtp.layers.0.*. Enorm/hnorm/eh_proj/shared_head.norm
    live on the MultiTokenPredictionLayer; the inner DSA+MoE layer (standard
    residual, no mHC) lives under mtp_model_layer. eh_proj halves SWAPPED
    (see swap_eh_halves). The swap acts on dim1 (columns) while the TP slice
    is dim0 (rows) — order-independent, so slicing BEFORE the swap is
    bit-identical to swap-then-slice."""
    lp = "mtp.layers.0.mtp_model_layer."
    ap = f"{lp}self_attention."
    mtp_top = "mtp.layers.0."
    composites = {}

    def load(key, mode):
        return ckpt.load(key, mode, tp_rank, tp_size)

    for tail, k in entries:
        if tail == "enorm.weight":
            assign(f"{mtp_top}enorm.weight", key=k)
        elif tail == "hnorm.weight":
            assign(f"{mtp_top}hnorm.weight", key=k)
        elif tail == "eh_proj.weight":
            tgt_name = f"{mtp_top}eh_proj.weight"
            # eh swap VERDICT (diag v2 cell D, 2026-11-20): the RAW checkpoint
            # order is correct — GLM stores [e | h], already aligned with
            # mcore's cat([enorm(e), hnorm(h)]). Swapping destroyed the draft
            # (CE 12.72 swapped vs 1.37 raw). Default OFF; the env knob kept
            # for A/B reproducibility.
            t = load(k, param_shard_mode(tgt_name))
            if os.environ.get("GLM53_V10_EH_SWAP", "0") == "1":
                t = swap_eh_halves(t)
            assign(tgt_name, t=t)
        elif tail == "shared_head.norm.weight":
            assign(f"{mtp_top}final_layernorm.weight", key=k)
        elif tail.startswith("self_attn."):
            _dsa_rule(tail, k, li, ap, composites, assign)
        elif tail.startswith("mlp."):
            _mlp_rule(tail, k, li, lp, True, composites, assign, grouped)
        elif tail == "input_layernorm.weight":
            assign(f"{lp}input_layernorm.weight", key=k)
        elif tail == "post_attention_layernorm.weight":
            assign(f"{lp}pre_mlp_layernorm.weight", key=k)
        else:
            raise ValueError(f"unhandled MTP key: {k}")
        consumed.add(k)
    for target, keys in composites.items():
        mode = param_shard_mode(target)
        parts = [load(kk, mode) for kk in keys]
        assign(target, t=torch.cat(parts, dim=0))
        consumed.update(keys)
    if grouped:
        flush_grouped(grouped, li, lp, ckpt, targets, written,
                      tp_rank, tp_size)


_KDA_MAP = {
    "q_proj.weight": "linear_q_proj.weight",
    "k_proj.weight": "linear_k_proj.weight",
    "v_proj.weight": "linear_v_proj.weight",
    "b_proj.weight": "linear_b_proj.weight",
    "f_a_proj.weight": "linear_fa_proj.weight",
    "f_b_proj.weight": "linear_fb_proj.weight",
    "g_a_proj.weight": "linear_ga_proj.weight",
    "g_b_proj.weight": "linear_gb_proj.weight",
    "o_norm.weight": "o_norm.weight",
    "o_proj.weight": "linear_o_proj.weight",
    "A_log": "A_log",
    "dt_bias": "dt_bias",
}

_DSA_MAP = {
    "q_a_proj.weight": "q_a_proj.weight",
    "q_a_layernorm.weight": "q_a_layernorm.weight",
    "q_b_proj.weight": "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight": "kv_a_proj_with_mqa.weight",
    "kv_a_layernorm.weight": "kv_a_layernorm.weight",
    "kv_b_proj.weight": "kv_b_proj.weight",
    "o_proj.weight": "o_proj.weight",
    "indexer.wq_b.weight": "indexer.linear_wq_b.weight",
    "indexer.wk.weight": "indexer.linear_wk.weight",
    "indexer.k_norm.weight": "indexer.k_norm.norm.weight",
    "indexer.k_norm.bias": "indexer.k_norm.norm.bias",
    "indexer.weights_proj.weight": "indexer.linear_weights_proj.weight",
    "indexer.index_kpool_compress_gate": "indexer.index_kpool_compress_gate",
    "indexer.index_kpool_compress_ape": "indexer.index_kpool_compress_ape",
}


def _kda_rule(tail, k, li, ap, composites, assign):
    sa = tail[len("self_attn."):]
    if sa == "q_conv1d.weight":
        pfx = f"{CKPT}layers.{li}.self_attn."
        composites[ap + "conv1d_weight"] = [pfx + "q_conv1d.weight",
                                            pfx + "k_conv1d.weight",
                                            pfx + "v_conv1d.weight"]
        return
    if sa in ("k_conv1d.weight", "v_conv1d.weight"):
        return  # consumed via the composite flush
    assert sa in _KDA_MAP, f"unhandled KDA key: self_attn.{sa}"
    assign(ap + _KDA_MAP[sa], key=k)


def _dsa_rule(tail, k, li, ap, composites, assign):
    sa = tail[len("self_attn."):]
    assert sa in _DSA_MAP, f"unhandled DSA key: self_attn.{sa}"
    assign(ap + _DSA_MAP[sa], key=k)


def _mlp_rule(tail, k, li, lp, is_moe, composites, assign, grouped=None):
    m = tail[len("mlp."):]
    if not is_moe:
        if m == "gate_proj.weight":
            pfx = f"{CKPT}layers.{li}.mlp."
            composites[lp + "mlp.linear_fc1.weight"] = [pfx + "gate_proj.weight",
                                                        pfx + "up_proj.weight"]
        elif m == "up_proj.weight":
            pass
        elif m == "down_proj.weight":
            assign(lp + "mlp.linear_fc2.weight", key=k)
        else:
            raise ValueError(f"unhandled dense mlp key: {tail}")
        return
    if m == "gate.weight":
        assign(lp + "mlp.router.weight", key=k)
    elif m == "gate.e_score_correction_bias":
        assign(lp + "mlp.router.expert_bias", key=k)
    elif m.startswith("shared_experts."):
        s = m[len("shared_experts."):]
        if s == "gate_proj.weight":
            pfx = f"{CKPT}layers.{li}.mlp.shared_experts."
            composites[lp + "mlp.shared_experts.linear_fc1.weight"] = [
                pfx + "gate_proj.weight", pfx + "up_proj.weight"]
        elif s == "up_proj.weight":
            pass
        elif s == "down_proj.weight":
            assign(lp + "mlp.shared_experts.linear_fc2.weight", key=k)
        else:
            raise ValueError(f"unhandled shared expert key: {tail}")
    else:
        em = m.split(".")
        assert em[0] == "experts" and len(em) == 4, f"unhandled mlp key: {tail}"
        e = int(em[1])
        proj = em[2]
        if grouped is not None:
            # GLM53_V10_GROUPED=1: routed experts live in V10GroupedExperts
            # (w1 [E,H,2f_loc] / w2 [E,f_loc,H], transposed dense layout) —
            # record the vendor keys; the flush builds w1/w2 in one pass.
            grouped.setdefault((lp, li), {}).setdefault(e, {})[proj] = k
            return
        if proj == "gate_proj":
            pfx = f"{CKPT}layers.{li}.mlp.experts.{e}."
            composites[lp + f"mlp.experts.local_experts.{e}.linear_fc1.weight"] = [
                pfx + "gate_proj.weight", pfx + "up_proj.weight"]
        elif proj == "up_proj":
            pass
        elif proj == "down_proj":
            assign(lp + f"mlp.experts.local_experts.{e}.linear_fc2.weight", key=k)
        else:
            raise ValueError(f"unhandled expert key: {tail}")
