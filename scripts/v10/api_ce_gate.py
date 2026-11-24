#!/usr/bin/env python3
"""DEPRECATED (2026-11-23) — disabled. Use scripts/v10/ce_gate.py instead.

History: the v10_distill/ce_gate.log fake green (2026-11-19, "mean CE =
0.0000 -> gate: PASS" with tokens=0 on every line) was produced by an
UNCOMMITTED variant of this HTTP-API gate whose per-token exception was
swallowed downstream, zeroing every sample. The committed code does not
reproduce that exact defect (no undefined-name at any tracked revision —
the forensic line attribution was unverifiable), so the whole HTTP path is
retired rather than trusted: main() hard-exits instead of gate-verdicting.
The live gate is ce_gate.py (offline vLLM prompt_logprobs, verified on .40
with real CE 5.4462/4.9808). extract_prompt_ce is kept ONLY as the tested
parsing reference (test_v10_units.py) — importing this module is silent;
executing it is fatal.
"""
import sys


def extract_prompt_ce(plp, ids):
    """Mean next-token CE over one prompt's vllm prompt_logprobs.

    Position 0 and None positions are skipped. Handles the HTTP-JSON form
    ({token_key: {"logprob": float}}) and the offline form
    ({token_key: Logprob(logprob=...)}); keys may be the token id (str or
    int, depending on the server serialization).
    """
    ce_toks = []
    for p in range(1, len(ids)):
        d = plp[p]
        if d is None:
            continue
        key = str(ids[p]) if str(ids[p]) in d else ids[p]
        if key not in d:
            continue
        entry = d[key]
        lp = entry["logprob"] if isinstance(entry, dict) \
            else getattr(entry, "logprob", None)
        if lp is None:
            continue
        ce_toks.append(-float(lp))
    if not ce_toks:
        raise ValueError("no scoring positions in prompt_logprobs response "
                         "(empty CE must not silently read as PASS)")
    return sum(ce_toks) / len(ce_toks)


def main():
    sys.stderr.write(
        "[FATAL] api_ce_gate.py is deprecated and DISABLED (fake-PASS "
        "incident 2026-11-19, see v10_distill/ce_gate.log). "
        "Use scripts/v10/ce_gate.py for the live CE gate.\n")
    sys.exit(2)


if __name__ == "__main__":
    main()
