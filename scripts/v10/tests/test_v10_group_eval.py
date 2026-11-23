#!/usr/bin/env python
"""eval_groups_40.batch_metrics kernel test (CPU, no mcore/NPU).

Fake model (echo contract):
  embedding(ids[B,S])            -> ids.float().unsqueeze(-1)        [B,S,1]
  decoder(emb)                   -> emb.clone()                       (h^0)
  mtp(input_ids=ids_k, hidden)   -> cat([zeros, ids_k.float()], dim=0)
  output_layer(chunk[-1])        -> logits [S,B,V] one-hot(ids_k[t+1,b])

With ids_k = roll(ids, -(k-1)): ids_k[t+1] == ids[t+k] == the kernel's
depth-k target => accept == 1.0 and CE ~ 0 on valid rows EXACTLY:
  - roll/target alignment (any misalignment breaks the echo)
  - true_len masking (poisoned pad rows must not leak into the metrics)
  - the chain (mtp depth k must receive depth k-1's d_k)
gather is monkeypatched to identity (TP1 semantics).
"""
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
V10 = os.path.dirname(HERE)
for p in (V10, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import eval_groups_40 as eg  # noqa: E402

V = 512


def test_kernel_alignment_mask_chain():
    eg._gather_logits = lambda x: x                    # TP1 identity
    torch.manual_seed(0)
    B, S, D = 3, 12, 5
    true_lens = [10, 7, 12]                            # row1 short, row2 full
    ids = torch.randint(1, V - 1, (B, S))
    for b, L in enumerate(true_lens):                  # poison pad region
        ids[b, L:] = 404
    batches = [{"input_ids": ids, "true_lens": true_lens,
                "case_ids": [f"c{b}" for b in range(B)]}]

    calls = {"rolls": [], "hiddens": []}

    class FakeM:
        def embedding(self, input_ids, position_ids):
            return input_ids.float().unsqueeze(-1)                 # [B,S,1]

        def decoder(self, hidden_states, attention_mask=None):
            return hidden_states.clone()

        def mtp(self, input_ids, position_ids, hidden_states,
                attention_mask=None, inference_params=None, embedding=None):
            calls["rolls"].append(input_ids.clone())
            calls["hiddens"].append(hidden_states.clone())
            e = torch.zeros_like(hidden_states)                    # [B,S,1]
            h = input_ids.float().unsqueeze(-1)                    # [B,S,1]
            return torch.cat([e, h], dim=0)                        # [2B,S,1]

        def output_layer(self, h):
            ids_k = h.squeeze(-1).long()                           # [B,S]
            Sx = ids_k.shape[1]
            logits = torch.full((Sx, ids_k.shape[0], V), -10.0)
            for t in range(Sx - 1):
                for b in range(ids_k.shape[0]):
                    logits[t, b, int(ids_k[b, t + 1].clamp(max=V - 1))] = 10.0
            return logits, None

    res = eg.batch_metrics(FakeM(), batches, mtp_num_layers=1, depths=D)
    assert set(res) == {f"c{b}" for b in range(B)}, res.keys()
    for b in range(B):
        d = res[f"c{b}"]
        L = true_lens[b]
        assert len(d["accept"]) == D and len(d["ce"]) == D, (b, d)
        for k in range(1, D + 1):
            if L - k <= 0:
                continue
            # echo alignment => perfect accept / zero CE on valid rows only
            assert abs(d["accept"][k - 1] - 1.0) < 1e-6, (b, k, d)
            assert d["ce"][k - 1] < 1e-3, (b, k, d)
    # chain: depth k's hidden input == depth k-1's draft hidden (= rolled ids)
    for k in range(1, D):
        want = calls["rolls"][k - 1].float().unsqueeze(-1)
        assert torch.equal(calls["hiddens"][k], want), f"chain broken at {k}"
    # roll alignment: depth k input == roll(ids, -(k-1))
    for k in range(1, D + 1):
        assert torch.equal(calls["rolls"][k - 1],
                           torch.roll(ids, shifts=-(k - 1), dims=1)), k


if __name__ == "__main__":
    test_kernel_alignment_mask_chain()
    print("kernel test OK")


def test_aggregate_means_and_pooling():
    """per_class = plain mean over its cases; per_group = plain mean over ALL
    cases (NOT class-weighted — unequal class sizes must not reweight)."""
    from v10_eval_taxonomy import classify  # noqa: F401 (import sanity)
    meta = {"a1": {"group": "g1", "cls": "en|math|single|mid"},
            "a2": {"group": "g1", "cls": "en|math|single|mid"},
            "b1": {"group": "g1", "cls": "zh|code|single|long"}}
    results = {"base": {"a1": {"accept": [0.2, 0.4], "ce": [2.0, 1.0]},
                        "a2": {"accept": [0.6, 0.8], "ce": [1.0, 0.0]},
                        "b1": {"accept": [0.9, 0.9], "ce": [0.5, 0.5]}}}
    agg = eg.aggregate(results, meta, depths=2)
    pc = agg["per_class"]
    assert abs(pc["base|g1|en|math|single|mid|D1"]["accept"] - 0.4) < 1e-9
    assert pc["base|g1|en|math|single|mid|D1"]["n"] == 2
    pg = agg["per_group"]
    # pooled over 3 cases: (0.2+0.6+0.9)/3 = 0.5667 (NOT class-weighted 0.65)
    assert abs(pg["base|g1|D1"]["accept"] - (0.2 + 0.6 + 0.9) / 3) < 1e-9
    assert pg["base|g1|D1"]["n"] == 3


def test_classify_edges():
    from v10_eval_taxonomy import classify, LONG_CUT
    def row(h, a, n=2):
        return {"conversations": ([{"from": "human", "value": h},
                                   {"from": "assistant", "value": a}] +
                                  [{"from": "human", "value": "x"},
                                   {"from": "assistant", "value": "y"}] * ((n - 2) // 2))}
    assert classify(row("求值域: y=2^x", "<think>..." + "a" * 10))[:2] == ("zh", "math")
    # lang rule = ANY CJK in human-or-assistant-head -> zh (deployed口径,
    # 组按此生成; 混合语言代码题判 zh 是设计决定不是 bug)
    assert classify(row("def foo(): pass 请解释", "<think>x</think>ok"))[:2] == ("zh", "code")
    assert classify(row("def foo(): pass", "<think>x</think>ok"))[:2] == ("en", "code")
    assert classify(row("阅读下面的文言文，完成题目", "答案"))[:2] == ("zh", "yuwen")
    assert classify(row("def f(): 导入import 后 计算方程", "a"))[1] == "math"   # math beats code
    assert classify(row("hello", "a" * (LONG_CUT - 1)))[3] == "mid"
    assert classify(row("hello", "a" * LONG_CUT))[3] == "long"
    assert classify(row("hello", "a", n=4))[2] == "multi"
    assert classify(row("hello", "a", n=2))[2] == "single"
