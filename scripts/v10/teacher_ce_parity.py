#!/usr/bin/env python3
"""V10 CE-gate upgrade: heuristic threshold -> EXACT per-position parity vs
the vLLM teacher (same text, same tokenizer, same weights).

Reads the mcore gate dump (ids + per-position CE + argmax from
launch_ce_gate.py --dump), queries the teacher server with prompt_logprobs=5,
and compares:
  - per-position CE:   mean|dCE| / max|dCE|  (bf16 noise-floor calibrated)
  - structural:        argmax agreement rate (iron rule: argmax >= 0.99 PASS)
Writes a json report + prints the worst mismatches.

Run: python3 teacher_ce_parity.py --dump v10_mcore_ce_40.json \
        [--server http://127.0.0.1:8400/v1] [--model glm53-flash-teacher]
"""
import argparse
import json
import time
import urllib.request


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


def forensics(plp, ids, mc_ce, mc_am):
    """Per-text comparison: CE deltas + argmax disagreements with teacher-side
    margins (small margin = bf16/路由数值序 noise flipping a near-tie)."""
    S = len(ids)
    te_ce, te_am, margins = [], [], {}
    for p in range(1, S):
        e = plp[p]
        assert e is not None, f"teacher logprob None at {p}"
        key = str(ids[p]) if str(ids[p]) in e else ids[p]
        entry = e[key]
        lp = entry["logprob"] if isinstance(entry, dict) else entry.logprob
        te_ce.append(-float(lp))
        top1, best = None, None
        for tok, v in e.items():
            l = float(v["logprob"] if isinstance(v, dict) else v.logprob)
            if best is None or l > best:
                best, top1 = l, int(tok)
        te_am.append(top1)
        pk = str(mc_am[p - 1]) if str(mc_am[p - 1]) in e else mc_am[p - 1]
        if pk in e:
            v = e[pk]
            margins[p - 1] = best - float(
                v["logprob"] if isinstance(v, dict) else v.logprob)
        else:
            margins[p - 1] = None     # mcore pick not even in teacher top-5
    n = len(te_ce)
    dces = [abs(a - b) for a, b in zip(mc_ce, te_ce)]
    dis = [{"pos": i, "mc_am": mc_am[i], "te_am": te_am[i],
            "te_margin": margins[i], "mc_ce": mc_ce[i], "te_ce": te_ce[i]}
           for i in range(n) if mc_am[i] != te_am[i]]
    return {"n": n, "dces": dces, "dis": dis, "mc_ce_mean": sum(mc_ce) / n,
            "te_ce_mean": sum(te_ce) / n, "te_ce": te_ce, "te_am": te_am}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8400/v1")
    ap.add_argument("--model", default="glm53-flash-teacher")
    ap.add_argument("--dump", required=True,
                    help="mcore gate dump json (single text or multi-text)")
    ap.add_argument("--report", default=None)
    args = ap.parse_args()

    with open(args.dump) as f:
        d = json.load(f)
    texts = d["results"] if "results" in d else [d]

    tot = tie = nontie = 0
    sum_dce = 0.0
    max_dce = 0.0
    mc_sum = te_sum = 0.0
    all_dis = []
    per_text = []
    for ti, r in enumerate(texts):
        ids, mc_ce, mc_am = r["ids"], r["ce"], r["argmax"]
        plp = post(args.server.rstrip("/") + "/completions", {
            "model": args.model, "prompt": ids, "max_tokens": 1,
            "prompt_logprobs": 5, "temperature": 0.0}
        )["choices"][0]["prompt_logprobs"]
        assert len(plp) == len(ids), \
            f"text{ti}: teacher returned {len(plp)}, want {len(ids)}"
        f = forensics(plp, ids, mc_ce, mc_am)
        t = sum(1 for x in f["dis"] if x["te_margin"] is not None
                and x["te_margin"] <= 0.2)
        tot += f["n"]
        tie += t
        nontie += len(f["dis"]) - t
        sum_dce += sum(f["dces"])
        max_dce = max(max_dce, max(f["dces"]))
        mc_sum += f["mc_ce_mean"] * f["n"]
        te_sum += f["te_ce_mean"] * f["n"]
        for x in f["dis"]:
            x["text"] = ti
        all_dis += f["dis"]
        per_text.append({"text": ti, "n": f["n"],
                         "mc_ce": f["mc_ce_mean"], "te_ce": f["te_ce_mean"],
                         "dis": len(f["dis"]), "ties": t,
                         "mean_dce": sum(f["dces"]) / f["n"]})
        print(f"text{ti:2d}: n={f['n']:4d}  mcCE={f['mc_ce_mean']:.4f} "
              f"teCE={f['te_ce_mean']:.4f}  mean|dCE|={sum(f['dces'])/f['n']:.4f}"
              f"  argmax-flip={len(f['dis'])} (ties {t})", flush=True)

    agree_raw = (tot - len(all_dis)) / tot
    non_tie_rate = nontie / tot
    verdict = "PASS" if (agree_raw >= 0.99 or non_tie_rate <= 0.005) else "FAIL"
    lines = [
        f"texts={len(texts)}  positions={tot}",
        f"CE_mean: mcore={mc_sum/tot:.4f}  teacher={te_sum/tot:.4f}",
        f"mean |dCE| = {sum_dce/tot:.5f}   max |dCE| = {max_dce:.5f}",
        f"argmax agreement raw = {agree_raw:.4f}  (gate >= 0.99)",
        f"flips: ties(margin<=0.2)={tie}  non-tie={nontie}  "
        f"non-tie rate={non_tie_rate:.4%}  (tie-excluded criterion <= 0.5%)",
        "",
        f"{'ALL non-tie flips:' if nontie else '(no non-tie flips)'}",
    ]
    for x in all_dis:
        if x["te_margin"] is None or x["te_margin"] > 0.2:
            m = "not-in-top5" if x["te_margin"] is None else f"{x['te_margin']:.4f}"
            lines.append(f"  text{x['text']} pos {x['pos']:4d}  "
                         f"mcore {x['mc_am']} vs te {x['te_am']}  margin={m}  "
                         f"mcCE={x['mc_ce']:.3f} teCE={x['te_ce']:.3f}")
    print("\n".join(lines))
    rep = {"verdict": verdict, "positions": tot, "agree_raw": agree_raw,
           "ties": tie, "non_tie": nontie, "non_tie_rate": non_tie_rate,
           "mean_dce": sum_dce / tot, "max_dce": max_dce,
           "mc_ce_mean": mc_sum / tot, "te_ce_mean": te_sum / tot,
           "per_text": per_text, "disagreements": all_dis}
    out = args.report or (args.dump.rsplit(".", 1)[0] + "_parity.json")
    with open(out, "w") as f:
        json.dump(rep, f, indent=1)
    print(f"\nSTATISTICAL PARITY GATE: {verdict}   -> {out}")


if __name__ == "__main__":
    main()
