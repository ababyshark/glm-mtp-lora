#!/usr/bin/env python3
"""382-case groups -> vLLM bench prompts (bench_acceptance.py format).

Per-group files (bench_acceptance takes counter DELTAS per run -> run it
once per group file for per-group K). The prompt = the HUMAN turn of the
case (the serving input); group/class ride in the id string.
"""
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "reports", "v10_eval382")
os.makedirs(OUT, exist_ok=True)

for grp in ("group1_unseen", "group2_cp250", "group3_late"):
    lines = []
    for line in open(os.path.join(ROOT, "data", "eval_groups",
                                  f"{grp}.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        human = next((t["value"] for t in r["conversations"]
                      if t["from"] == "human"), "")
        lines.append(json.dumps({
            "id": f"{grp}#{r['ds_idx']}|{r['class']}",
            "prompt": human,
        }, ensure_ascii=False))
    path = os.path.join(OUT, f"prompts_{grp}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{grp}: {len(lines)} prompts -> {path}")
