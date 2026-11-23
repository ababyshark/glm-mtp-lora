#!/usr/bin/env python3
"""A/B acceptance benchmark for MTP speculative decoding (baseline vs fine-tuned CP).

Sends a fixed prompt set (reports/test_prompts_200.jsonl) to a vllm-ascend server
with greedy decoding (temperature=0 => deterministic acceptance), then extracts
speculative-decode acceptance counters from /metrics (counter deltas are taken
against a pre-run scrape so only THIS run's tokens are counted).

    python scripts/bench_acceptance.py --tag baseline --port 8455
    python scripts/bench_acceptance.py --tag cp50b    --port 8455
"""
import argparse
import json
import os
import re
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROMPTS = os.environ.get(
    "V10_BENCH_PROMPTS",
    os.path.join(ROOT, "reports", "test_prompts_200.jsonl"))
OUT_DIR = os.path.join(ROOT, "reports", "ab_results")

N_PROMPTS = 200
CONCURRENCY = 8
MAX_TOKENS = 512


def http(url, payload=None, token=None, timeout=600):
    h = {"Content-Type": "application/json"}
    data = None
    if payload is not None:
        data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=h)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(req, timeout=timeout).read()


def get_metrics(port):
    return http(f"http://localhost:{port}/metrics").decode()


def parse_spec_counters(text):
    out = {}
    for line in text.splitlines():
        if not line.startswith("vllm:spec_decode") or line.startswith("#"):
            continue
        try:
            key, val = line.rsplit(" ", 1)
            out[key] = out.get(key, 0.0) + float(val)
        except ValueError:
            pass
    return out


def run_bench(port, n, prompts_file=None):
    prompts = [json.loads(l) for l in open(prompts_file or PROMPTS)][:n]
    before = parse_spec_counters(get_metrics(port))
    t0 = time.time()
    results, errs = [], 0
    sem_key = [0]

    import concurrent.futures as cf

    def one(case):
        payload = {
            "model": "glm53",
            "messages": [{"role": "user", "content": case["prompt"]}],
            "temperature": 0, "max_tokens": MAX_TOKENS,
        }
        t1 = time.time()
        r = json.loads(http(f"http://localhost:{port}/v1/chat/completions", payload))
        dt = time.time() - t1
        u = r["usage"]
        return {"id": case["id"], "s": round(dt, 2),
                "completion_tokens": u["completion_tokens"],
                "text_head": r["choices"][0]["message"]["content"][:80]}

    with cf.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        for res in ex.map(one, prompts):
            results.append(res)
    wall = time.time() - t0

    after = parse_spec_counters(get_metrics(port))
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in after}
    return {"wall_s": round(wall, 1), "n": len(results), "results": results,
            "spec_counters_delta": delta}


def summarize(tag, res):
    # vllm 0.27.1 counter format (position is a LABEL, 0-based draft depth):
    #   vllm:spec_decode_num_drafts_total{...}
    #   vllm:spec_decode_num_draft_tokens_total{...}          (= K * draft steps)
    #   vllm:spec_decode_num_accepted_tokens_per_pos_total{...,position="0"..K-1}
    d = res["spec_counters_delta"]
    K = 5
    drafts = sum(v for k, v in d.items()
                 if "num_drafts_total" in k and "_created" not in k)
    draft_toks = sum(v for k, v in d.items()
                     if "num_draft_tokens_total" in k and "_created" not in k)
    steps = drafts if drafts else draft_toks / K
    accepted = {}
    for k, v in d.items():
        if "num_accepted_tokens_per_pos_total" in k and "_created" not in k:
            m = re.search(r'position="(\d+)"', k)
            if m:
                accepted[int(m.group(1)) + 1] = accepted.get(int(m.group(1)) + 1, 0) + v
    per_depth = {i: round(accepted.get(i, 0) / steps, 4) if steps else None
                 for i in range(1, 6)}
    acc_total = sum(accepted.values())
    mean_acc = (acc_total / steps) if steps else None
    toks = sum(r["completion_tokens"] for r in res["results"])
    return {"tag": tag, "draft_steps": int(steps) if steps else 0,
            "draft_tokens": int(draft_toks), "accepted_total": int(acc_total),
            "mean_accept_len": round(mean_acc, 3) if mean_acc else None,
            "accept_rate": round(acc_total / draft_toks, 4) if draft_toks else None,
            "per_depth_accept": per_depth,
            "throughput_tok_s": round(toks / res["wall_s"], 1),
            "wall_s": res["wall_s"], "n": res["n"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--port", type=int, default=8455)
    ap.add_argument("--n", type=int, default=N_PROMPTS)
    ap.add_argument("--prompts", default=None, help="prompt jsonl (default reports/test_prompts_200.jsonl)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    res = run_bench(args.port, args.n, args.prompts)
    summ = summarize(args.tag, res)
    out = os.path.join(OUT_DIR, f"{args.tag}.json")
    json.dump({"summary": summ, "raw": res}, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print("saved ->", out)


if __name__ == "__main__":
    main()
