#!/usr/bin/env python3
"""V10 T2.2 (API edition): on-policy greedy rollout against the vllm teacher.

Teacher: http://127.0.0.1:8400/v1 (glm53-flash-teacher, GLM-5.3-Flash BF16 TP=16).
Prompt = glm5_next_text template prefix + user + <|assistant|><think> (ids from
gen_onpolicy_prompts.jsonl). Greedy, total length <= 4096. Resume = one part
file per row; a merged output jsonl is written at --out.
"""
import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CUTOFF = 4096
STOP_IDS = None  # filled after tokenizer load


def call(server, payload, tries=5):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(server.rstrip("/") + "/completions", data=data,
                                 headers={"Content-Type": "application/json"})
    for t in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=1800) as r:
                return json.loads(r.read())
        except Exception as e:
            if t == tries - 1:
                raise
            time.sleep(5 * (t + 1))


def one(row, args):
    outp = os.path.join(args.parts_dir, f"r{row['row']}.json")
    if os.path.exists(outp):
        return None
    ids = row["prompt_token_ids"]
    payload = {
        "model": args.model,
        "prompt": ids,
        "max_tokens": max(1, CUTOFF - len(ids)),
        "temperature": 0.0,
        "stop_token_ids": STOP_IDS,
    }
    resp = call(args.server, payload)
    ch = resp["choices"][0]
    rec = {"row": row["row"], "idx": row["idx"], "domain": row["domain"],
           "prompt_len": len(ids), "gen_text": ch.get("text", ""),
           "finish_reason": ch.get("finish_reason"),
           "gen_token_ids": ch.get("token_ids", [])}
    tmp = outp + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, ensure_ascii=False)
    os.replace(tmp, outp)
    return rec


def main():
    global STOP_IDS
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8400/v1")
    ap.add_argument("--model", default="glm53-flash-teacher")
    ap.add_argument("--prompts", default="/mnt/share_space/l00840580/mtp_git/v10_distill/onpolicy_prompts.jsonl")
    ap.add_argument("--parts-dir", default="/mnt/share_space/l00840580/mtp_git/v10_distill/parts")
    ap.add_argument("--out", default="/mnt/share_space/l00840580/mtp_git/v10_distill/rollout_all.jsonl")
    ap.add_argument("--conc", type=int, default=96)
    ap.add_argument("--tokenizer", default="/mnt/weight2/GLM-5.3-Flash")
    args = ap.parse_args()
    os.makedirs(args.parts_dir, exist_ok=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    stop = []
    for s in ("<|user|>", "<|observation|>"):
        e = tok.convert_tokens_to_ids(s)
        if e is not None and e >= 0:
            stop.append(e)
    if tok.eos_token_id is not None:
        stop.append(tok.eos_token_id)
    STOP_IDS = stop

    rows = [json.loads(l) for l in open(args.prompts)]
    for i, r in enumerate(rows):
        r["row"] = i
    todo = [r for r in rows if not os.path.exists(os.path.join(args.parts_dir, f"r{r['row']}.json"))]
    print(f"total {len(rows)}, done {len(rows)-len(todo)}, todo {len(todo)}", flush=True)

    t0 = time.time()
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.conc) as ex:
        for rec in ex.map(lambda r: one(r, args), todo):
            if rec is not None:
                n_done += 1
            if n_done and n_done % 200 == 0:
                dt = time.time() - t0
                print(f"{n_done}/{len(todo)} rows, {n_done/dt:.2f} rows/s, "
                      f"elapsed {dt/60:.1f} min", flush=True)

    with open(args.out, "w") as f:
        for r in rows:
            p = os.path.join(args.parts_dir, f"r{r['row']}.json")
            if os.path.exists(p):
                f.write(open(p).read().strip() + "\n")
    print(f"merged -> {args.out}; COMPLETE", flush=True)


if __name__ == "__main__":
    main()
