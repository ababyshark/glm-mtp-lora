#!/usr/bin/env python
"""V10 eval-set taxonomy + deterministic group sampler (2026-11-21 user spec).

Taxonomy: 2 lang x 4 task x 2 turns x 2 length = 32 classes.
  lang   {zh, en}        zh iff human turn OR assistant head(500 chars) has CJK
  task   {math, code, yuwen, general}
                         priority on the HUMAN turn: yuwen (zh + art regex)
                         > math > code > general; math beats code (math+code
                         -> math) to keep the axis at 4 — measured on the
                         consumed-1000: math 199(132+67), code 113 -> math+code
                         folded into math per the 32-class spec
  turns  {single, multi} conversations length <=2 vs >2
  length {mid, long}     total assistant chars <8000 vs >=8000 (cut between
                         the consumed-1000 median 5015 and P90 21258)

Sampler windows (the batcher is randperm(30000, seed=42+epoch), batch=1,
no-within-epoch replacement — every consumed sample is trained EXACTLY ONCE):
  position p of the permutation = the step (0-based) that consumes ds[p].
  run-1 (steps 1-1000)   consumed positions [0, 1000)
  run-2 (resume->2000)   consumes positions [1000, 2000) (skip_batches=1000)
  => Group 1 (UNSEEN at every CP up to 2000) draws from the complement of
     [0, 2000); Group 2 (trained-by-CP250) draws from positions [0, 250).
"""
import json
import os
import re

import torch

POOL = "data/lf_data/cmb_eagle_30k.jsonl"
SEED = 42
ZH = re.compile(r"[\u4e00-\u9fff]")
MATH = re.compile(r"(计算|求解|值域|方程|几何|概率|数列|函数|整数|余数|\\frac|\\int"
                  r"|\\overrightarrow|\bprove\b|how many|probability|sum of"
                  r"|largest|smallest|integer|digit)")
CODE = re.compile(r"(def |class |import |function|method|代码|编程|算法实现"
                  r"|python|Java|C\+\+|API|bug)")
YUWEN = re.compile(r"(阅读|文言文|作文|诗句|古诗|赏析|翻译下面|理解下面的|主旨|作者意图)")
LONG_CUT = 8000                       # assistant chars


def classify(row):
    """-> (lang, task, turns, length) — the 32-class label tuple."""
    convs = row["conversations"]
    human = next((t["value"] for t in convs if t["from"] == "human"), "")
    asst = "".join(t["value"] for t in convs if t["from"] == "assistant")
    lang = "zh" if (ZH.search(human) or ZH.search(asst[:500])) else "en"
    if lang == "zh" and YUWEN.search(human):
        task = "yuwen"
    elif MATH.search(human):
        task = "math"
    elif CODE.search(human):
        task = "code"
    else:
        task = "general"
    turns = "multi" if len(convs) > 2 else "single"
    length = "long" if len(asst) >= LONG_CUT else "mid"
    return (lang, task, turns, length)


def consumption_positions(n_pool=30000, epoch=0):
    """order[p] = ds index consumed at step p (0-based). Deterministic."""
    g = torch.Generator().manual_seed(SEED + epoch)
    return torch.randperm(n_pool, generator=g).tolist()


def label_str(cls):
    return "|".join(cls)


def build_groups(pool_path=POOL, per_class=10, out_dir="data/eval_groups",
                 group1_exclude=2000, group2_window=250, seed=20261121):
    rows = [json.loads(l) for l in open(pool_path, encoding="utf-8")]
    order = consumption_positions(len(rows))
    labels = [classify(r) for r in rows]

    by_label_pool = {}
    for i, lb in enumerate(labels):
        by_label_pool.setdefault(label_str(lb), []).append(i)

    pos_of = {ds_idx: p for p, ds_idx in enumerate(order)}
    g1_exclude_pos = set(range(group1_exclude))
    g1_excluded_ds = {order[p] for p in g1_exclude_pos}

    g = torch.Generator().manual_seed(seed)

    def sample_from(cands, k):
        cands = sorted(cands)
        if len(cands) <= k:
            return cands, len(cands)
        pick = [cands[i] for i in
                torch.randperm(len(cands), generator=g)[:k].tolist()]
        return pick, len(cands)

    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for grp, cand_fn in (
            ("group1_unseen", lambda lb: [i for i in by_label_pool[lb]
                                          if i not in g1_excluded_ds]),
            ("group2_cp250", lambda lb: [i for i in by_label_pool[lb]
                                         if pos_of.get(i, 10**9) < group2_window]),
            # group3_late: consumed at steps 751-1000 — UNSEEN at CP250/500/750
            # (positions >=750 are outside every earlier window), trained once
            # by CP1000 only. vs group2 (trained by CP250) = the recency read.
            ("group3_late", lambda lb: [i for i in by_label_pool[lb]
                                        if 750 <= pos_of.get(i, 10**9) < 1000])):
        lines, short = [], []
        for lb in sorted(by_label_pool):
            cands = cand_fn(lb)
            picked, avail = sample_from(cands, per_class)
            if len(picked) < per_class:
                short.append((lb, avail))
            for i in picked:
                lines.append(json.dumps({
                    "group": grp, "ds_idx": i, "class": lb,
                    "step_consumed": pos_of.get(i),
                    "conversations": rows[i]["conversations"],
                }, ensure_ascii=False))
        path = os.path.join(out_dir, f"{grp}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        report[grp] = {"path": path, "cases": len(lines),
                       "classes": len(set(json.loads(l)["class"] for l in lines)),
                       "shortfall": short}
    return report


if __name__ == "__main__":
    rep = build_groups()
    for k, v in rep.items():
        print(f"[{k}] {v['cases']} cases / {v['classes']} classes -> {v['path']}")
        if v["shortfall"]:
            print(f"  shortfall (<{10} available): {v['shortfall']}")
