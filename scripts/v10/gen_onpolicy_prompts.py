#!/usr/bin/env python3
"""V10 T2.1 (refreshed): five-domain prompt extraction, quota-balanced.

Domains (doc): 语文/数学/代码/英语/科学推理. Source: cmb_eagle_30k
(the sft-old-data megatron corpora decode to garbage with the Flash tokenizer
-> fallback to doc's cmb candidates; reported honestly in T5).
Each row: {"idx", "domain", "n_tokens", "prompt_token_ids"}, prompt <=1024.
"""
import argparse
import json
import os
import re

CODE_PAT = re.compile(r"```|^\s*(def |class |import |from \S+ import|function |#!/|<\?php|public static)|\b(?:java|python|javascript|c\+\+|rust|golang)\b", re.I)
MATH_PAT = re.compile(r"\$[^$]+\${1,2}|\\frac|\\int|\\sum|\\sqrt|\\begin\{|\d+\s*[\+\-\*/=]\s*\d+|[0-9]+%", re.I)
SCI_PAT = re.compile(r"物理|化学|生物|量子|分子|原子|基因|细胞|能量|热力学|遗传|生态|天体|行星|实验|假设|变量|磁场|光速|physics|chemistry|biology|molecule|quantum|experiment|hypothesis|enzyme|orbit|thermodynamic", re.I)
REASON_PAT = re.compile(r"首先|其次|然后|因此|所以|由此|综上|推导|证明|第一步|第二步|Step \d|therefore|thus|hence|because", re.I)
THINK_PAT = re.compile(r"<think>|</think>")


def classify(conv):
    human = next(t["value"] for t in conv if t["from"] == "human")
    if CODE_PAT.search(human):
        return "代码", human
    if SCI_PAT.search(human):
        return "科学推理", human
    if MATH_PAT.search(human):
        return "数学", human
    n_cn = len(re.findall(r"[\u4e00-\u9fff]", human))
    if n_cn / max(len(human), 1) > 0.3:
        return "语文", human
    return "英语", human


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/share_space/l00840580/mtp_git/lf_glm53_qlora_mtp/data/lf_data/cmb_eagle_30k.jsonl")
    ap.add_argument("--tokenizer", default="/mnt/weight2/GLM-5.3-Flash")
    ap.add_argument("--out", default="/mnt/share_space/l00840580/mtp_git/v10_distill/onpolicy_prompts.jsonl")
    ap.add_argument("--quota", type=int, default=6000, help="per-domain quota")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    PREFIX = "[gMASK]<sop><|system|>Reasoning Effort: Max"

    dom_rows = {}
    with open(args.data) as f:
        for i, line in enumerate(f):
            dom, human = classify(json.loads(line)["conversations"])
            dom_rows.setdefault(dom, []).append((i, human))
    print("domain counts:", {k: len(v) for k, v in dom_rows.items()})

    stats = {}
    fills = {"英语", "语文"}
    with open(args.out, "w") as out:
        for dom, rows in dom_rows.items():
            pick = rows[: args.quota]
            stats[dom] = len(pick)
        shortfall = args.quota * 5 - sum(stats.values())
        for dom in fills:
            if shortfall <= 0:
                break
            extra = dom_rows[dom][stats[dom]: stats[dom] + shortfall]
            dom_rows[dom] = dom_rows[dom][: args.quota] + extra
            stats[dom] += len(extra)
            shortfall -= len(extra)
        for dom, rows in dom_rows.items():
            for idx, human in rows:
                ids = tok(PREFIX + "<|user|>" + human + "<|assistant|><think>",
                          add_special_tokens=False)["input_ids"]
                if len(ids) > 1024:
                    ids = ids[:1024]
                out.write(json.dumps({"idx": idx, "domain": dom,
                                      "n_tokens": len(ids),
                                      "prompt_token_ids": ids}) + "\n")
    print("picked per domain:", stats, "total:", sum(stats.values()))
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
