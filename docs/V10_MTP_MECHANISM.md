# V10 MTP native 训练机制 — 源码审读纪要 + 17.09 矛盾分析

> 2026-11-20。依据：运行时真实源码 `/tmp/opencode/mc0182`（worktree @571370c82
> = venv_v8 安装版本）+ V10 接线代码。所有引用行号均出自该 worktree。
> mc0182 patch 只改 dot_product_attention / schedules / moe_layer，不碰 mtp（已验证）。

---

## 一、mcore 0.18.2 native MTP 训练的完整数据流（逐行验证）

前提：PP=1 / TP=16 / CP=1，`mtp_num_layers=1`，B=1，`parallel_output=True`。
形状约定：GPTModel.forward 收 **[B,S]** input_ids/labels；embedding 内部转置，
decoder 走 seq-first [S,B,H]；logits 输出 [S,B,V/tp]（vocab 分片）。

```
GPTModel.forward (gpt_model.py:499)
  _preprocess → decoder_input [S,B,H]
  decoder (TransformerBlock ×45 Glm5NextLayer) → hidden [S,B,H]
  _postprocess (gpt_model.py:599):
    if mtp_in_postprocess:                                  # = mtp_process ✓
      hidden_states = self.mtp(                             # multi_token_prediction.py:1620
        input_ids, position_ids, hidden_states, embedding=self.embedding)
      # Block.forward (:1603):
      #   chunk(hidden, 1+offset=1) → [decoder]             # get_mtp_layer_offset=0 (:539, PP=1)
      #   layers[0](hidden_states=decoder, input_ids, ...)   # → (layer_out, rolled_ids, ...)
      #   hidden_states_list.append(layer_out)               # [decoder, layer_out]
      #   return cat(dim=0) = [2S, B, H]                     # decoder 在前 ✓
    if self.config.mtp_num_layers:                          # =1 ✓
      hidden_states = process_mtp_loss(hidden_states, labels, ...)   # :674
  logits, _ = self.output_layer(hidden_states, ...)          # :728  ← hidden=chunk[0]
  loss = compute_language_model_loss(labels, logits)         # :765  ← 返回值
  return loss
```

### process_mtp_loss（:621-714）
```
chunk(hidden [2S,B,H], 2, dim=0) → chunk[0]=decoder, chunk[1]=layer_out
if labels is None: return chunk[0]                          # 推理/无标签路径只回主头
for depth in range(1):
    mtp_logits = output_layer(chunk[1])                     # [S,B,V/tp]
    mtp_labels = roll(labels, -1, dim=-1)；尾部填 0         # 预测 t+1
    mtp_loss = vocab CE(mtp_logits, mtp_labels^T)           # 见下，CE 对齐正确
    hidden_states = MTPLossAutoScaler.apply(chunk[0], scale*mtp_loss)
return chunk[0]  # forward 恒等；mtp 梯度只在 backward 注入（:581 AutoScaler）
```
**关键结论 1：返回的 loss 数值 = 主头 CE（labels, lm_head(decoder)）——
mtp 项只存在于梯度里，不影响打印值。**

### MultiTokenPredictionLayer（:722 起）
```
_get_embeddings(:1048 区域)：input_ids roll(-1) → e = embedding(ids_{t+1})
_concat_embeddings(:935)：enorm(e) | hnorm(h) → cat = **[e | h]** → eh_proj
  → gather_from_tensor_model_parallel_region（TP gather 回全维）
_proj_and_transformer_layer(:962)：inner = mtp_model_layer（标准残差
  TransformerLayer：input_ln → DSA → bda → pre_mlp_ln → MoE → bda）
layer._postprocess(:1022)：final_layernorm（= checkpoint shared_head.norm）
layer 返回 (out, rolled_ids, rolled_pos, pad_mask)
```

### compute_language_model_loss（language_module.py:144）
```
labels [B,S] → transpose → [S,B]；logits [S,B,V/tp]（seq-first）
vocab_parallel_cross_entropy(logits, labels_T)  # mcore 内部跨 TP CE，数学正确
```

### V10 侧接线（v10_model.py:141-176 / v10_mtp.py:37-75）
- build(mtp=True) → GPTModel(mtp_block_spec=...) → model.mtp = MultiTokenPredictionBlock ✓
- inner spec = mcore TransformerLayer（标准残差）+ V10DSAAttention(causal) + MoE ✓
- fill: mtp.layers.0.{enorm,hnorm,eh_proj,final_layernorm,mtp_model_layer.*}，
  eh_proj 按列半交换（[h|e]→[e|h] 假设，[推断]）✓ 台账干净
- batcher: [B,S] 右对齐 padding，labels=input_ids（含 pad=0，全 LM loss ——
  已知未迁 -100 掩码，audit 在案）

---

## 二、经验矛盾（.40 实测四元组）

同一 text0（prefixed，S=882，ids 完全一致——dump 来自门）：

| 测量 | 值 | 路径 |
|---|---|---|
| 门（mtp=False，显式 mask，**手动 all_gather + fp32 CE**） | **0.8864** | 自算 CE |
| 探针 labels=None（mtp=True，mask=None，logits[0,:-1] 直接 softmax） | 4.9712 | **V/16 分片 logits 上算的** |
| 探针 native（labels=ids，mcore 内部 vocab CE） | 17.0942 | mcore 内部 CE |
| 探针 DIRECT（手动 mtp call + lm_head） | 18.2358 | 自算 |

eh_proj 置零/切换：native 值逐位不变 —— **与"返回值=主头 CE"完全一致**
（eh_proj 是 mtp 专属，不在主头路径 ✓ 这条矛盾已解）。

**未解矛盾：门 0.886 vs mcore 内部 CE 17.09。** 按源码，两者测同一量
（主头 CE）。候选解释（按可能性排序）：

1. **mcore 内部 CE 在我们的 TP/形状组合下算错**（最可能）——17.09 ≈
   "对错位/错分片的 softmax 求 CE"的量级。可能的机制：
   `compute_language_model_loss` 的 labels 转置假定 logits 是 [S,B,V]，
   而我们的 output_layer 若在某路径返回了 gather 后的 [B,S,V]（或相反），
   CE 会在错误轴上配对。B=1 时 flatten 顺序本应相同——需实测排除。
2. concat 顺序在运行时为 [layer_out; decoder]（与源码相反）——则 17.09 =
   CE(lm_head(layer_out), labels)（draft 预测当前 token ≈ 乱），且
   labels=None 的 4.97 = CE(lm_head(chunk[0]=layer_out), 未移位 labels) 的
   部分词表版本。**与源码矛盾，但与数值自洽。**
3. 探针 bug（未排除：4.97 是在 V/16 分片上 log_softmax 的值——分片 CE
   对低 id 文本可算出 4-5 的假值；gather 未崩是因为 text0 token 多为低 id）。

**判定性实验（已实现，.40 空闲后一次 16 卡 run 全部出数）**：
`scripts/v10/v10_mtp_diag.py`（探针 v2，全部数字走 v10_diag.py 的
gate-proven 路径：all_gather + fp32 softmax）：
a) cell A：hook 记录 mtp 块入参/出参，两个 chunk 分别与 decoder 输出做
   max-abs-diff → 一锤定音 concat 顺序（假设 2）
b) cell B：chunk[0]/chunk[1] 各自手动全词表 CE vs labels + mcore 内部值 +
   teacher 锚点（parity json 逐文本真值）→ 假设 1 裁决
c) cell C：TRUE draft CE（手动路径）——修正第一轮探针的假值（18.24 是
   V/16 分片 softmax 的伪值）
d) cell D：eh A/B 用正确度量重测（带 max-abs-diff 变更验证 + 回写校验）
e) phase 2：mtp=False 重建 → mask=None vs 显式 tril 的主头 CE → 分离掩码变量
基座：`scripts/v10/v10_diag.py`（gather_full_logits / full_vocab_ce /
lm_head_ce / max_abs_diff），单测 `tests/test_v10_diag.py`（含 2 进程 gloo
gather 端到端）✅ 65+4 全绿（2026-11-20）。

---

## 三、对 T4 的含义

1. **训练暂停是对的**：在 (1)/(2) 裁决前，native loss 的语义未知——
   "能收敛"不构成开工依据（loss 若是度量 bug，收敛的是噪声）。
2. eh 交换问题**降级**：zero/切换不影响 native 值说明 eh 不在主头路径，
   这只证明"返回值=主头 CE"（机制 1 ✓），**不能**证明 draft 路径正确。
   DIRECT draft CE 18.2 的归因同样悬在 (1)/(2) 上。
3. 若 (b) 证实模型无恙（= 假设 1）：修复 = trainer/probe 统一用手动
   all_gather CE（复刻门的实现）做 loss 度量；梯度仍由 mcore AutoScaler
   注入（不受影响），训练可继续。
4. 若 (a) 证实 concat 反序：修 = V10 侧包一层（调换 block 返回的 chunk 序），
   或改用 mcore 的 hybrid mtp 路径；同时 eh 交换需重判。

## 四、本轮方法教训（写入即避免重蹈）

1. **只读运行时 worktree**（venv 版本号 = commit 短哈希，`pip show megatron-core`
   或 venv 装路径反查），tag 版本与 worktree 并行存在时严禁混读——本轮
   17.09 分析一度被 tag/worktree 差异污染。
2. 每个经验数字标注测量路径（内部 CE / 手动 gather CE / 直接 lm_head），
   不同路径的数字不可直接对比。
3. probe 的 checksum 设计要先做不变性分析（求和对半区交换天然不敏感——
   第一轮 eh 探针的 toggle 检验是无效检验）。
