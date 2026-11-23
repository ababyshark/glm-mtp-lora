#!/usr/bin/env python
"""V10 MTP diag probe v2 — the four decisive cells from
docs/V10_MTP_MECHANISM.md §二, one 16-card run, all numbers on the gate-proven
measurement path (v10_diag.gather_full_logits + fp32 softmax):

  A. chunk order     — checksum-vs-decoder for both halves of the mtp concat
                       (max-abs-diff, NOT sum) → concat order verdict
  B. metric          — manual full-vocab CE on chunk[0]/chunk[1] vs labels
                       + mcore-internal native value + teacher anchor
                       (from the parity json) → is 17.09 a metric bug?
  C. draft CE (true) — manual CE on the NON-decoder chunk vs ids[t+1]
  D. eh A/B (true)   — toggle eh halves (verified by max-abs-diff), rerun C

Phase 2 (after freeing phase-1 model): mtp=False rebuild → main CE with
mask=None vs explicit tril → isolates the mask variable.

Run (.40, when freed):
  torchrun --nproc_per_node=16 v10_mtp_diag.py --dump h0-test/eval35/v10_mcore_ce16p_40.json
"""
import argparse
import json
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torch  # noqa: E402

from launch_ce_gate import npu_shim  # noqa: E402
from v10_diag import (full_vocab_ce, gather_full_logits,  # noqa: E402
                      lm_head_ce, max_abs_diff, rolled_targets, sig)


def report(rank, lines):
    if rank == 0:
        for ln in lines:
            print(ln, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="h0-test/eval35/v10_mcore_ce16p_40.json")
    ap.add_argument("--parity", default="h0-test/eval35/v10_mcore_ce16p_40_parity.json")
    ap.add_argument("--text", type=int, default=0)
    args = ap.parse_args()

    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    npu_shim()
    import torch.distributed as dist
    dist.init_process_group(backend="hccl", init_method="env://")
    from megatron.core import parallel_state
    world = int(os.environ["WORLD_SIZE"])
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=world)
    rank = dist.get_rank()

    from v10_config import vendor_config
    from v10_fill import fill_model
    from v10_model import build_v10_model

    vc = vendor_config()
    with open(args.dump) as f:
        ids = torch.tensor([json.load(f)["results"][args.text]["ids"]],
                           dtype=torch.int64, device="npu")
    S = ids.shape[1]
    tgt_main = rolled_targets(ids)                    # [S-1]

    te_anchor = None
    if rank == 0 and os.path.exists(args.parity):
        with open(args.parity) as f:
            par = json.load(f)
        te_anchor = [x["te_ce"] for x in par.get("per_text", [])] or None

    # ================= phase 1: mtp=True =================
    model, cfg = build_v10_model(vc=vc, device="npu", mtp=True,
                                 max_seq_len=2048, tp_size=world)
    ledger = fill_model(model, os.environ.get(
        "V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16"), vc, mtp=True,
        tp_rank=rank, tp_size=world)
    assert not ledger["unwritten"] and not ledger["unconsumed"]
    model.eval()

    # recorder: capture the mtp block's input (decoder out) and output (concat)
    rec = {}
    orig_forward = model.mtp.forward

    def recorded_forward(*a, **kw):
        rec["in"] = kw["hidden_states"].detach().clone()
        out = orig_forward(*a, **kw)
        rec["out"] = out.detach().clone()
        return out

    model.mtp.forward = recorded_forward

    with torch.no_grad():
        native = model(input_ids=ids, position_ids=torch.zeros_like(ids),
                       attention_mask=None, labels=ids)
    native = native[0] if isinstance(native, tuple) else native
    if native.dim() > 0:
        native = native.float().mean()
    native = float(native)

    # cell B2: same-tensor CE comparison INSIDE compute_language_model_loss —
    # if the internal vocab-parallel CE differs from the manual gather-CE on
    # the very same logits tensor, the bug is inside the internal CE (matters:
    # the mtp-term GRADIENTS flow through it).
    ce_probe = {}
    import megatron.core.models.common.language_module.language_module as _lm

    orig_clml = _lm.LanguageModule.compute_language_model_loss

    def probing_clml(self, labels, logits):
        lg = logits[0] if isinstance(logits, tuple) else logits
        lg = lg[:, 0] if lg.dim() == 3 else lg
        full = gather_full_logits(lg, dim=-1)
        ce_probe["manual"] = float(full_vocab_ce(full, labels[0]).mean())
        ce_probe["internal"] = float(orig_clml(self, labels, logits).mean())
        return orig_clml(self, labels, logits)

    _lm.LanguageModule.compute_language_model_loss = probing_clml
    with torch.no_grad():
        _ = model(input_ids=ids, position_ids=torch.zeros_like(ids),
                  attention_mask=None, labels=ids)
    _lm.LanguageModule.compute_language_model_loss = orig_clml
    report(rank, [
        f"[mtp-diag] B2 same-tensor CE: internal = {ce_probe.get('internal', float('nan')):.4f}"
        f"   manual(gather) = {ce_probe.get('manual', float('nan')):.4f}   "
        f"same logits tensor → bug inside vocab_parallel CE"
        f" if these differ",
    ])

    c0, c1 = torch.chunk(rec["out"], 1 + cfg.mtp_num_layers, dim=0)
    d0 = max_abs_diff(c0, rec["in"])
    d1 = max_abs_diff(c1, rec["in"])
    if d0 <= d1:
        order = f"decoder-FIRST (max|c0-in|={d0:.3e}, max|c1-in|={d1:.3e})"
        dec_chunk, mtp_chunk = c0, c1
    else:
        order = f"REVERSED (chunk1==decoder: max|c1-in|={d1:.3e}, max|c0-in|={d0:.3e})"
        dec_chunk, mtp_chunk = c1, c0

    # cell B: manual full-vocab CE on BOTH chunks + the internal value + anchor
    ce_dec = lm_head_ce(model, dec_chunk, tgt_main).mean().item()
    ce_mtp_as_main = lm_head_ce(model, mtp_chunk, tgt_main).mean().item()

    # cell C/D: TRUE draft CE with a FRESH mtp forward per eh convention.
    # (Probe-v1 lesson: measuring on a recorded tensor makes the toggle a
    # no-op — the chunk must be RECOMPUTED after each eh change.)
    def fresh_draft_ce():
        """Mirror GPTModel._postprocess's exact mtp call, then gate-path CE."""
        with torch.no_grad():
            emb = model.embedding(input_ids=ids, position_ids=torch.zeros_like(ids))
            emb = emb[0] if isinstance(emb, tuple) else emb
            dec = model.decoder(hidden_states=emb, attention_mask=None)
            dec = dec[0] if isinstance(dec, tuple) else dec
            mtp_out = model.mtp(input_ids=ids, position_ids=torch.zeros_like(ids),
                                hidden_states=dec, attention_mask=None,
                                inference_params=None, embedding=model.embedding)
            chunks = torch.chunk(mtp_out, 1 + cfg.mtp_num_layers, dim=0)
            lg, _ = model.output_layer(chunks[-1])
            lg = lg[0] if isinstance(lg, tuple) else lg
            if lg.dim() == 3:
                lg = lg[:, 0]
            full = gather_full_logits(lg, dim=-1)
            ce = full_vocab_ce(full, tgt_main)
        return ce.mean().item(), float(ce.max()), chunks[-1]

    draft_ce, draft_max, mtp_chunk = fresh_draft_ce()
    chunk_sig = sig(mtp_chunk)
    eh = model.mtp.layers[0].eh_proj.weight
    saved = eh.detach().clone()
    h2 = eh.shape[1] // 2
    with torch.no_grad():
        eh.data.copy_(torch.cat([eh[:, h2:], eh[:, :h2]], dim=1))
    toggled = max_abs_diff(eh, saved)
    draft_ce_toggled, draft_max_toggled, _ = fresh_draft_ce()
    with torch.no_grad():
        eh.data.copy_(saved)
    restored = max_abs_diff(eh, saved)

    report(rank, [
        f"[mtp-diag] text{args.text} S={S}  sig(decoder)={sig(rec['in'])}",
        f"[mtp-diag] A concat order: {order}",
        f"[mtp-diag] B native(mcore-internal CE) = {native:.4f}",
        f"[mtp-diag] B manual full-vocab CE: chunk-as-decoder = {ce_dec:.4f}"
        f"   other-chunk-vs-main-labels = {ce_mtp_as_main:.4f}",
        f"[mtp-diag] B teacher anchor for text{args.text} = "
        f"{te_anchor[args.text] if te_anchor else 'n/a'}",
        f"[mtp-diag] C TRUE draft CE (fresh forward, manual path) = {draft_ce:.4f}"
        f"   max-pos = {draft_max:.4f}   chunk sig={chunk_sig}",
        f"[mtp-diag] D eh toggle max-diff={toggled:.3e} (changed={toggled > 0})"
        f"  draft CE toggled = {draft_ce_toggled:.4f} (max {draft_max_toggled:.4f})"
        f"  restore diff={restored:.3e}",
        f"[mtp-diag] D VERDICT: "
        f"{'AS-FILLED keeps swap' if draft_ce < draft_ce_toggled else 'TOGGLED (drop swap)'}"
        f"  (|d|={abs(draft_ce - draft_ce_toggled):.4f}; "
        f"d==0 => eh NOT in draft graph = wiring bug)",
    ])

    # ================= cell E: mtp input-convention bisect =================
    # E1 h=decoder-out(pre-norm)+inner   (= current 12.72)
    # E2 h=decoder-out POST main-final-norm + inner (mapping gap #3 实测)
    # E3 inner-skip: same concat, skip the inner transformer (贡献隔离)
    # E4 roll-direction: e(t) instead of e(t+1)
    layer0 = model.mtp.layers[0]

    def mtp_stage_ce(h, use_inner=True, roll_e=True):
        with torch.no_grad():
            ids_r = torch.roll(ids, shifts=-1, dims=-1) if roll_e else ids
            e = model.embedding(input_ids=ids_r,
                                position_ids=torch.zeros_like(ids_r))
            e = e[0] if isinstance(e, tuple) else e
            x = layer0._concat_embeddings(h, e)      # enorm|cat|eh_proj|TP-gather
            if use_inner:
                y = layer0.mtp_model_layer(hidden_states=x, attention_mask=None)
                y = y[0] if isinstance(y, tuple) else y
            else:
                y = x
            out = layer0._postprocess(y)             # final_layernorm
            lg, _ = model.output_layer(out)
            lg = lg[0] if isinstance(lg, tuple) else lg
            if lg.dim() == 3:
                lg = lg[:, 0]
            full = gather_full_logits(lg, dim=-1)
            return full_vocab_ce(full, tgt_main).mean().item()

    h_pre = dec_chunk                                # decoder out (pre-norm)
    with torch.no_grad():
        h_post = model.decoder.final_layernorm(h_pre).clone()
    e1 = mtp_stage_ce(h_pre, use_inner=True, roll_e=True)
    e2 = mtp_stage_ce(h_post, use_inner=True, roll_e=True)
    e3 = mtp_stage_ce(h_pre, use_inner=False, roll_e=True)
    e3b = mtp_stage_ce(h_post, use_inner=False, roll_e=True)
    e4 = mtp_stage_ce(h_pre, use_inner=True, roll_e=False)
    report(rank, [
        f"[mtp-diag] E bisect: E1 pre-norm+inner = {e1:.4f}   "
        f"E2 post-norm+inner = {e2:.4f}",
        f"[mtp-diag] E inner-skip: pre-norm = {e3:.4f}   post-norm = {e3b:.4f}"
        f"   (inner 贡献 = E1 - skip)",
        f"[mtp-diag] E roll-check: e(t+1) = {e1:.4f}   e(t) = {e4:.4f}",
    ])

    del model, rec
    torch.npu.empty_cache()

    # ================= phase 2: mtp=False (mask isolation) =================
    dist.barrier()
    model2, _ = build_v10_model(vc=vc, device="npu", mtp=False,
                                max_seq_len=2048, tp_size=world)
    ledger2 = fill_model(model2, os.environ.get(
        "V10_VENDOR_DIR", "/mnt/weight2/GLM-5.3-Flash-BF16"), vc, mtp=False,
        tp_rank=rank, tp_size=world, verbose=False)
    assert not ledger2["unwritten"] and not ledger2["unconsumed"], \
        f"phase-2 fill ledger not clean: {ledger2['unwritten'][:3]}"
    model2.eval()
    tril = torch.ones(1, 1, S, S, dtype=torch.bool, device="npu").tril()
    with torch.no_grad():
        lg_none = model2(input_ids=ids, position_ids=torch.zeros_like(ids),
                         attention_mask=None)
        lg_mask = model2(input_ids=ids, position_ids=torch.zeros_like(ids),
                         attention_mask=tril)
    lg_none = (lg_none[0] if isinstance(lg_none, tuple) else lg_none)[0]
    lg_mask = (lg_mask[0] if isinstance(lg_mask, tuple) else lg_mask)[0]
    ce_none = full_vocab_ce(gather_full_logits(lg_none), tgt_main).mean().item()
    ce_mask = full_vocab_ce(gather_full_logits(lg_mask), tgt_main).mean().item()
    report(rank, [
        f"[mtp-diag] C main CE (mtp=False, manual): mask=None = {ce_none:.4f}"
        f"   mask=tril = {ce_mask:.4f}",
        "[mtp-diag] DONE",
    ])
    dist.barrier()


if __name__ == "__main__":
    main()
