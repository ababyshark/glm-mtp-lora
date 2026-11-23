#!/usr/bin/env python3
"""V10 T1 semantic CE gate via the vllm teacher API (prompt_logprobs).

Gate: mean next-token CE on 16 real cmb_eagle_30k texts ~2-4 (<< ln V=11.95).
Log: v10_distill/ce_gate.log
"""
import argparse
import json
import math
import time
import urllib.request

PREFIX = "[gMASK]<sop><|system|>Reasoning Effort: Max"
DATA = "/mnt/share_space/l00840580/mtp_git/lf_glm53_qlora_mtp/data/lf_data/cmb_eagle_30k.jsonl"


def post(url, payload, tries=3):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for t in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                return json.loads(r.read())
        except Exception:
            if t == tries - 1:
                raise
            time.sleep(10)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8400/v1")
    ap.add_argument("--model", default="glm53-flash-teacher")
    ap.add_argument("--tokenizer", default="/mnt/weight2/GLM-5.3-Flash")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--log", default="/mnt/share_space/l00840580/mtp_git/v10_distill/ce_gate.log")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    seqs = []
    with open(DATA) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            conv = json.loads(line)["conversations"]
            human = next((t["value"] for t in conv if t["from"] == "human"), "")
            ids = tok(PREFIX + "<|user|>" + human + "<|assistant|><think>",
                      add_special_tokens=False)["input_ids"][:2048]
            seqs.append((i, ids))

    lines, ces = [], []
    for i, ids in seqs:
        r = post(args.server.rstrip("/") + "/completions", {
            "model": args.model, "prompt": ids, "max_tokens": 1,
            "prompt_logprobs": 0, "temperature": 0.0})
        plp = r["choices"][0]["prompt_logprobs"]
        ce = extract_prompt_ce(plp, ids)
        ces.append(ce)
        lines.append(f"sample idx={i} len={len(ids)} CE={ce:.4f}")
        print(lines[-1], flush=True)
    mean = sum(ces) / len(ces)
    verdict = "PASS" if mean <= 4.5 else "FAIL"
    with open(args.log, "w") as f:
        f.write("\n".join(lines) + f"\n\nmean CE = {mean:.4f}\n"
                f"ln(vocab 154880) = {math.log(154880):.4f}\ngate: {verdict}\n")
    print(f"MEAN CE = {mean:.4f} -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
