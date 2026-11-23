#!/usr/bin/env python3
"""V10 T2.3: merge rollout shards -> per-domain flash_onpolicy_<domain>.jsonl.

conversations: human = original cmb prompt, gpt = Flash self-generated text.
Output goes to the repo root dir (user requirement) and lf_data/.
"""
import argparse
import json
import os

DATA = "/mnt/share_space/l00840580/mtp_git/lf_glm53_qlora_mtp/data/lf_data/cmb_eagle_30k.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True)
    ap.add_argument("--out-dir", default="/mnt/share_space/l00840580/mtp_git")
    ap.add_argument("--also-lf-data", action="store_true")
    args = ap.parse_args()

    orig = {}
    with open(DATA) as f:
        for i, line in enumerate(f):
            conv = json.loads(line)["conversations"]
            orig[i] = next(t["value"] for t in conv if t["from"] == "human")

    per_dom = {}
    n = 0
    for shard in args.shards:
        with open(shard) as f:
            for line in f:
                r = json.loads(line)
                dom = DOM_BY_IDX.get(r["idx"])
                if dom is None:
                    continue
                text = r["gen_text"].strip()
                if not text or len(text) < 8:
                    continue
                per_dom.setdefault(dom, []).append({
                    "conversations": [
                        {"from": "human", "value": orig[r["idx"]]},
                        {"from": "gpt", "value": text},
                    ],
                    "domain": dom,
                })
                n += 1

    os.makedirs(args.out_dir, exist_ok=True)
    for dom, rows in per_dom.items():
        path = os.path.join(args.out_dir, f"flash_onpolicy_{dom}.jsonl")
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{dom}: {len(rows)} -> {path}")
    print(f"total {n}")
    if args.also_lf_data:
        lf = "/mnt/share_space/l00840580/mtp_git/lf_glm53_qlora_mtp/data/lf_data"
        for dom in per_dom:
            src = os.path.join(args.out_dir, f"flash_onpolicy_{dom}.jsonl")
            dst = os.path.join(lf, f"flash_onpolicy_{dom}.jsonl")
            os.replace(src, dst) if os.path.exists(src) else None
            print("moved ->", dst)


if __name__ == "__main__":
    # domain lookup from the prompts file (single source of truth)
    with open("/mnt/share_space/l00840580/mtp_git/v10_distill/onpolicy_prompts.jsonl") as f:
        DOM_BY_IDX = {}
        for line in f:
            r = json.loads(line)
            DOM_BY_IDX[r["idx"]] = r["domain"]
    main()
