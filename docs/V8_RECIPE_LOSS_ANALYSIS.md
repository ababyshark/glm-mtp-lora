# V8 Loss 构成 + 外部分析评估 + 主线 bisect 计划

> 记录时间：2026-11-19。目的：把「serving_chain 配方」的事实、对外部「Loss/收敛优化」建议的取舍、
> 以及当前真正卡点（h0 前向 bug）的定位路径固化下来，避免后续方向跑偏。

---

## 一、当前根因（主线，优先级最高）

**主干 78 层 decoder 的 h0 是坏的**：`h0-quality diag: main-head CE=12.85 ≈ ln(V)=11.95`（近随机）。

- 表现：MTP draft 头 CE ≈ ln V，ladder `[13.8 → 10.9]`（depth-1 最差、越深略好）——这是「h0 喂垃圾」的签名，不是「深层 mode collapse」。
- 已排除：环境/栈/OOM/meta 泄漏、TP16 机制、全部权重值级 EXACT（native + slim + MTP head/embed/rot）、recipe 数学 1:1 移植、label 对齐、depth-1 注意力（k=1 平凡）。
- 结论：**前向 bug，在 decoder 内部（attention rope/mask、MoE 路由、norm/residual 三者其一）。与 loss 配方无关。**

**主线动作（唯一解卡路径）**：逐层对拍 V8 vs branch-1（`_mlab`），定位第一个 rel 跳变层，再钻那一层。

---

## 二、Loss 构成事实（serving_chain recipe，`v8_recipe.py`）

最终 `loss = CE + 2.33·KL + 0.5·agree + 0.3·margin`（V7.0 判决 env `launch_v70_38.sh:23-30`）：

| 项 | 公式 | 权重 | 标签 |
|---|---|---|---|
| CE | 对 gold 下一 token 交叉熵 | 1.0 | 硬 |
| KL 蒸馏 | `pv·(log pv − log pd)` = **KL(verifier‖draft)** | 2.33 | 软（主干分布）|
| agree-CE | draft logits 追 verifier argmax | 0.5 | 硬伪 |
| margin | `relu(m − (logit_tgt − max_other))` | 0.3, m=0.5 | 硬 |

关键语义：
1. **KL 方向是 Forward KL**（verifier 作教师，draft 作学生）。branch-1 docstring 的 "KL(draft||v)" 是过时命名，代码权威反着。
2. **verifier 全程 detach**：`vlp`/`vargmax` 从 `h0 → output_layer` 算一次后 `.detach()`，梯度只进 draft 头，不回传主干。

---

## 三、外部建议评估（「可借鉴 / 否 / 已证伪」）

### 可借鉴（阶段在后，非当前）
1. **On-policy 自生成数据**（Phase 5）：cmb_eagle_30k 是静态人类语料，verifier 在其上的 vlp 是 off-policy 分布，是「接受率上不去」的经典根因。做法：冻结 Base 自生成 response 再训。**救不了当前 h0。**
2. **步数衰减权重**（Phase 4/5 消融）：`γ^(k-1)` 衰减硬标签项（CE/agree/margin），k≥3 只留 KL。针对单层共享权重的多步梯度冲突。**前提是 h0 先对、ladder 恢复递增。**
3. **MTP 步数 2~3 步是端到端吞吐甜点**（部署账本）。

### 不可借鉴 / 已证伪
1. **Scheduled Sampling / self-feed**：已证伪。`launch_v70_38.sh:24` `GLM53_CHAIN_SELF_FEED=0 # self-feed falsified (V5.2/V5.3)`。V8 的 `serving_chain_recipe_loss(self_feed=False)` 是残留 stub，别重做。
2. **立刻调 loss 权重**：h0 没修之前调了白调（拟合垃圾输入）。

---

## 四、两个诊断问题的代码事实（外部分析所问）

1. **GLM MTP 是单层共享还是每步独立权重？**
   → **单层共享**（DeepSeek-V3 式）。`v8_mtp.py` `_v8_mtp_block_forward`: `layer_idx = 0 if self.mtp_use_repeated_layer else iteration`；recipe 5 个深度全复用 `mtp.layers[0]`（`v8_recipe.py:190` `tl = layer.mtp_model_layer`）。

2. **第 2~5 步传的是上一步 MTP 输出，还是重新从主干截断？**
   → **上一步 MTP 输出（链式 hidden）+ Gold token embedding（teacher forcing）**。
   `v8_recipe.py`：`prev_norm = layer.final_layernorm(out_k)` 作为下一深度输入（chained）；`emb = embed(ids_flat[c0+k : c0+k+s_k])`（gold）。
   即「hidden 是 roll 的、token 是 gold 的」——推理时 token 会是 draft 自己的 argmax，这处训练/推理失配是真实存在的，但其修法（self-feed）已被否。

---

## 五、主线 bisect 计划（当前唯一解卡路径）

1. **h0 对拍**：固定输入，V8 decoder 输出 h0 vs branch-1 base 输出，比 rel。
   - `rel < 1e-2` → decoder 正常，问题在 MTP 层内。
   - `rel 大` → decoder 前向 bug，逐层 bisect。
2. **逐层 bisect（§4.3）**：emb → norm → attn → norm → mlp，逐层 rel，第一个非零层命名 culprit。
   - 重点怀疑顺序：attention rope/mask > MoE 路由 > norm/residual 接线。
3. **MTP 层内前向对拍**：同一 h0 + 同一 ids，V8 `chain_step` vs branch-1 `chain_step`，按 `rot→hnorm→eh_proj→input_layernorm→attn→mlp→final_layernorm→head` 逐项比中间量。

> 已完成的证据链：valueprobe（mtp_head/embed/rot EXACT）+ slimvalue（eh_proj/q_up/专家 W4W8 EXACT）→ 权重全对；问题在激活层，不在权重层。
