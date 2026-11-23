#!/usr/bin/env python3
"""V10 T2.2: on-policy greedy rollout on GLM-5.3-Flash BF16 via vllm-ascend.

16-card layout: DP=2 x TP=8 (306GB BF16 needs >=8 cards per group), one process
per TP group (ASCEND_RT_VISIBLE_DEVICES 0-7 / 8-15). Greedy, total length <= 4096
stop at <|user|>/<|observation|>/EOS. Per-shard jsonl with append-style resume.
"""
import argparse
import json
import os
import time

CUTOFF = 4096


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/mnt/weight2/GLM-5.3-Flash")
    ap.add_argument("--prompts", default="/mnt/share_space/l00840580/mtp_git/v10_distill/onpolicy_prompts.jsonl")
    ap.add_argument("--out", required=True, help="shard output jsonl")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=2)
    ap.add_argument("--tp", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="only first N prompts of this shard (speed test)")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    stop_ids = set()
    for s in ("<|user|>", "<|observation|>"):
        e = tok.convert_tokens_to_ids(s)
        if e is not None and e >= 0:
            stop_ids.add(e)
    if tok.eos_token_id is not None:
        stop_ids.add(tok.eos_token_id)

    rows = []
    with open(args.prompts) as f:
        for line in f:
            rows.append(json.loads(line))
    rows = [r for r in rows if r["idx"] % args.num_shards == args.shard]
    if args.limit:
        rows = rows[: args.limit]

    done = set()
    if os.path.exists(args.out):
        with open(args.out) as f:
            for line in f:
                done.add(json.loads(line)["idx"])
    todo = [r for r in rows if r["idx"] not in done]
    print(f"shard {args.shard}/{args.num_shards}: {len(rows)} prompts, "
          f"{len(done)} done, {len(todo)} to go", flush=True)

    llm = LLM(model=args.model, tensor_parallel_size=args.tp,
              max_model_len=CUTOFF, gpu_memory_utilization=0.90,
              trust_remote_code=True)

    batch = []
    for r in todo:
        n = r["n_tokens"]
        max_new = min(CUTOFF - n, CUTOFF - n)  # total <= cutoff
        batch.append((r["idx"], r["prompt_token_ids"], max_new))

    sp_kwargs = dict(temperature=0, stop_token_ids=list(stop_ids))
    t0 = time.time()
    CHUNK = 512
    n_gen = 0
    for s in range(0, len(batch), CHUNK):
        part = batch[s: s + CHUNK]
        sps = [SamplingParams(max_tokens=m, **sp_kwargs) for _, _, m in part]
        outs = llm.generate([TokensPrompt(prompt_token_ids=ids) for _, ids, _ in part], sps)
        with open(args.out, "a") as f:
            for (idx, ids, m), o in zip(part, outs):
                f.write(json.dumps({
                    "idx": idx,
                    "prompt_len": len(ids),
                    "gen_token_ids": list(o.token_ids),
                    "gen_text": o.text,
                    "finish_reason": str(o.finish_reason),
                }) + "\n")
        n_gen += len(part)
        dt = time.time() - t0
        print(f"shard {args.shard}: {n_gen}/{len(batch)} gen, "
              f"{n_gen * 1.0 / dt:.1f} seqs/s, elapsed {dt / 60:.1f} min", flush=True)
    print(f"shard {args.shard} complete", flush=True)


if __name__ == "__main__":
    main()
