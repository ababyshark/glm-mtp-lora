#!/usr/bin/env python3
"""V10 T1: semantic CE gate for the rollout generator (iron rules §4 step ZERO).

Loads GLM-5.3-Flash BF16 via vllm-ascend (this stack is the rollout path;
HF load is impossible here: transformers 5.14.1 has no glm5_next and the
checkpoint ships no remote code) and measures main-head next-token CE on real
cmb_eagle_30k texts with vllm prompt_logprobs. Gate: mean CE ~2-4 (<< ln V=11.95).
"""
import argparse
import json
import os

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

PREFIX = "[gMASK]<sop><|system|>Reasoning Effort: Max"
DATA = "/mnt/share_space/l00840580/mtp_git/lf_glm53_qlora_mtp/data/lf_data/cmb_eagle_30k.jsonl"


def build_full_ids(tok, conv, cutoff=2048):
    human = next((t["value"] for t in conv if t["from"] == "human"), "")
    gpt = next((t["value"] for t in conv if t["from"] == "gpt"), "")
    prompt = PREFIX + "<|user|>" + human + "<|assistant|><think>"
    if "</think>" in gpt:
        reasoning = gpt.split("</think>")[0].split("<think>")[-1]
        answer = gpt.split("</think>")[-1].strip()
    else:
        reasoning, answer = "", gpt.strip()
    assistant = reasoning + "</think>" + answer
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tok(assistant, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return (ids + a_ids)[:cutoff]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/weight2/GLM-5.3-Flash")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--log", default="/mnt/share_space/l00840580/mtp_git/v10_distill/ce_gate.log")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    seqs = []
    with open(DATA) as f:
        for i, line in enumerate(f):
            if i >= args.n:
                break
            conv = json.loads(line)["conversations"]
            seqs.append((i, build_full_ids(tok, conv)))
    print(f"gate: {len(seqs)} real-text samples, lens "
          f"{[len(s) for _, s in seqs]}", flush=True)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=2048, gpu_memory_utilization=0.90,
              trust_remote_code=True, enforce_eager=True)
    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    outs = llm.generate([TokensPrompt(prompt_token_ids=s) for _, s in seqs],
                        sp)

    lines, ces = [], []
    for (i, s), out in zip(seqs, outs):
        plp = out.prompt_logprobs
        ce_toks = []
        for pos in range(1, len(s)):
            d = plp[pos]
            lp = d[s[pos]].logprob if d is not None and s[pos] in d else None
            if lp is not None:
                ce_toks.append(-lp)
        ce = sum(ce_toks) / len(ce_toks)
        ces.append(ce)
        lines.append(f"sample idx={i} len={len(s)} tokens_scored={len(ce_toks)} CE={ce:.4f}")
        print(lines[-1], flush=True)
    mean = sum(ces) / len(ces)
    verdict = "PASS" if mean <= 4.5 else "FAIL"
    with open(args.log, "w") as f:
        f.write("\n".join(lines) + f"\n\nmean CE = {mean:.4f}\n"
                f"ln(vocab) = {__import__('math').log(154880):.4f}\n"
                f"gate (mean CE <= 4.5): {verdict}\n")
    print(f"MEAN CE = {mean:.4f}  -> {verdict}", flush=True)


if __name__ == "__main__":
    main()
