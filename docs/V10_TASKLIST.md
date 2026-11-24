# V10 任务清单 — GLM-5.3-Flash MTP 训练（LF + Megatron-Core）

> 前置：docs/V10_IRON_RULES.md（铁律）+ docs/V10_DESIGN.md（架构设计）
> 参考：V1-V8 全部历史方案与 15 轮审计教训

---

## 一、LF + Megatron 适配度分析

### 1.1 Flash 架构对 mcore 0.18.2 的适配度

| Flash 特性 | mcore 0.18.2 原生支持 | 适配方案 | 风险 |
|---|---|---|---|
| **MoE（288 experts, topk 8）** | ✅ SequentialMLP + router | 直接复用 V8 模式（config 锚定） | 低 |
| **MLA（q_lora 1536, nope 256, 无 rope）** | ✅ MLATransformerConfig | 维度改 Flash 的值；rope 相关全部移除 | 低 |
| **DSA indexer（32H/128D/topk 2048）** | ⚠️ TE-coupled（V8 已解决：V8CoreAttention wrapper） | 复用 V8 wrapper + 修维度 | 低 |
| **Gated DeltaNet linear attention（34 层）** | ❌ 无原生支持 | 三选一：A. 从 HF 移植 / B. mcore MambaLayer 适配 / C. 先占位跑通 pipeline | **高** |
| **hyper-connections (mhc)** | ❌ 无此概念 | 自定义 layer forward wrapper（hc_apply + sinkhorn） | **高** |
| **hybrid layer types（LA + DSA 交替）** | ✅ TransformerBlockSubmodules per-layer spec | V8 已验证 | 低 |
| **weight_scale_inv 量化** | ❌ 非 ModelSlim 格式 | 需新的反量化/直通方案 | 中 |
| **MTP（layers.45）** | ✅ MultiTokenPredictionBlock | 复用 V8 模式 + 维度改 | 低 |
| **嵌入/输出（154880 vocab）** | ✅ VocabParallelEmbedding | 直接复用 | 低 |

### 1.2 LF 对 V10 的适配度

| LF 功能 | V8 使用情况 | Flash 适配 | 变化量 |
|---|---|---|---|
| **get_dataset（数据管线）** | ✅ cmb_eagle_30k + glm5_next_text | tokenizer 需确认是否同 vocab 154880 | 可能为零 |
| **get_template_and_fix_tokenizer** | ✅ glm5_next_text | Flash 用的 template 可能不同 → 确认 | 低 |
| **load_tokenizer** | ✅ | tokenizer 路径改为 Flash 的 | 低 |
| **训练循环（Trainer）** | ❌ V8 绕过了 LF 的 Trainer（自定义 fwdbwd） | 同 V8：LF 仅做数据，训练循环自定义 | 零变化 |
| **yaml 配置** | ✅ 数据键（dataset/template/cutoff_len） | 新建 Flash yaml | 低 |

### 1.3 集成点

```
LF 职责：get_dataset → tokenized samples → McoreBatcher → [B,S] batches
Megatron 职责：GPTModel(hybrid) → forward → backward → optimizer → CP
胶水：v10_trainer.py（同 V8 v8_trainer.py 模式）
```

**与 V8 的差异**：数据管线基本复用，训练循环的模型构建完全重写（hybrid + hc + 新量化），优化器可能从 LoRA-only 改为 LoRA + 部分全参（Flash 是 bf16，无 quant 限制）。

### 1.4 内存预算（TP16 估算）

Flash 总参数 ≈ 45 层 × (attention ~100M + MoE 288×37.7M ≈ 10.9B/层 × 42 + dense 3×226M) ≈ 45×10.9B ≈ 490B params

但 Flash 是 bf16 标准 HF（非 W4A8 packed）：
- bf16 全量 = 980GB >> 61G/卡
- **必须 TP16 分片**：980G / 16 = 61.25G/rank — **超过 61.27G 限制！**

Flash bf16 TP16 每卡 = 61.25G — 几乎刚好 = 61.27G 上限 — 没有激活空间！

**结论：Flash bf16 TP16 装不下训练（无激活/优化器空间）。**

方案：
- a. **训练侧量化**：把 Flash 权重转 W4A8 packed → 23.4G/rank → 剩 38G 给激活+优化器 ✓
- b. **Flash 瘦身**：Flash 可能有小版本（check config）
- c. **混合精度**：权重 bf16 + 梯度/优化器 fp32 → 更大
- d. **Flash 实际参数量重算**：45 层 × (attention 165M + MoE 288×37.7M) ≈ 45×11B ≈ 495B — bf16 = 990GB — /16 = 62G/rank — **确认装不下**

**结论：Flash 也需要量化后才能 TP16 训练。Flash 的 weight_scale_inv 可能本身就是量化格式。**

需要 Phase 0 探明：weight_scale_inv 是什么量化（W4A8? W8A8? block-wise 的精度？）。

### 1.5 V8 可复用资产清单

| V8 资产 | Flash 可复用 | 适配量 |
|---|---|---|
| v8_trainer.py 训练循环骨架 | ✅ 模型构建部分重写 | 中 |
| v8_data.py LF 数据管线 | ✅ tokenizer 路径改 | 低 |
| v8_recipe.py serving_chain 配方 | ⚠️ 维度全变 | 中 |
| v8_slim.py packed residency | ❌ Flash 无 W4A8 packed | N/A（除非 Flash 也量化） |
| v8_mtp.py MTP contract | ⚠️ 结构同但维度变 | 中 |
| v8_model.py 模型构建 | ❌ Flash 是不同架构 | 高 |
| fill_vendor.py fill 骨架 | ⚠️ 反量化数学要改（新 quant 格式） | 中 |
| mc0182 patch | ✅ mcore 版本相同 | 零 |
| v8_data.py LF get_dataset | ✅ | 零 |
| setup env | ✅ venv_v7 复用 | 零 |
| 测试基础设施 | ✅ 模式复用 | 低 |

---

## 二、完整任务清单

### Phase 0：理解与环境（估时 1-2 天）

| # | 任务 | 依赖 | 产出 | 验证 |
|---|---|---|---|---|
| 0.1 | Flash config/weight 格式理解：weight_scale_inv 的 dtype/shape/反量化数学 | 无 | 探针报告 | 反量化值与 HF 前向输出对拍 |
| 0.2 | Flash tokenizer/template 确认 | 0.1 | tokenizer 路径 | tokenize 一个样本 → 与 W4A8 比对 |
| 0.3 | vendor HF Glm5Next 实现阅读：linear_attention + mhc + MTP 的精确语义 | 0.1 | 设计笔记 | 每个算子的输入/输出/公式记录 |
| 0.4 | mcore 0.18.2 linear_attention 支持调研 | 0.3 | 选型决策（A/B/C） | 一行代码验证 mcore MambaLayer 是否可用 |
| 0.5 | Flash 参数量/内存预算重算（含量化） | 0.1 | 内存账本 | TP16 每卡 < 55G |

### Phase 1：模型构建 + 语义门（估时 3-7 天）

| # | 任务 | 依赖 | 产出 | 验证 |
|---|---|---|---|---|
| 1.1 | Flash mcore 骨架：hybrid layer spec（LA + DSA 交替）+ MoE + MTP | 0.3/0.4 | v10_model.py | 参数数与 vendor 对拍 |
| 1.2 | linear_attention 实现 | 0.4 | GatedDeltaNet module 或适配 | 单测：与 HF 输出对拍 |
| 1.3 | hyper-connections 实现 | 0.3 | hc wrapper module | 单测：hc 输出与 HF 对拍 |
| 1.4 | weight_scale_inv 反量化 | 0.1 | 反量化函数 | 与 HF 权重加载对拍 |
| 1.5 | vendor→mcore mapping | 1.1 | mapping json | 全覆盖 0 missing |
| 1.6 | fill | 1.5 | fill_v10 | 权重范数对拍 |
| **1.7** | **语义 CE 门（第零步）** | 1.6 | 主头 CE on 真实文本 | **CE ≈ 2-4 = PASS；≈11 = FAIL（不过不继续）** |
| 1.8 | 逐算子 bisect（如果 1.7 不过） | 1.7 | 第一分叉算子定位 | per-op rel 对拍 |

### Phase 2：训练循环（估时 2-3 天）

| # | 任务 | 依赖 | 产出 | 验证 |
|---|---|---|---|---|
| 2.1 | v10_trainer.py 骨架 | 1.7 | fwdbwd + optimizer + CP | 1 卡 2 步 + determinism |
| 2.2 | optimizer 策略 | 2.1 | **已裁决（2026-11-20 用户令）：LoRA on BF16 基座 + MTP 全训**——基座本体冻结、无量化（QLoRA 名沿用项目名，实际不量化） | 冻结泄漏=0 |
| 2.3 | recipe serving_chain 迁移 | 2.1 | v10_recipe.py | per-depth ladder 对拍 |
| 2.4 | pytest 全绿 | 2.1-2.3 | test 日志提交 | 全部通过 |

### Phase 3：部署验证（估时 1 天）

| # | 任务 | 依赖 | 产出 | 验证 |
|---|---|---|---|---|
| 3.1 | 16 卡 fill + 首步 | 2.4 | fill 16/16 + step 1 loss | determinism 逐位 |
| 3.2 | CP round-trip | 3.1 | restore 后 loss 一致 | 逐位 |
| 3.3 | 内存验证 | 3.1 | alloc/resv < 55G | |
| 3.4 | ladder 门 2b | 3.1 | full-depth ladder 形状 | 递增 ~2.5→7 |

### Phase 4：配方迁移 + 判决对照（估时 2-3 天）

| # | 任务 | 依赖 | 产出 | 验证 |
|---|---|---|---|---|
| 4.1 | V5.4a 配方迁移（CE+KL+agree+margin） | 3.4 | v10_recipe.py 完整版 | per-depth 对拍 |
| 4.2 | LF -100 掩码迁移 | 4.1 | 监督行掩码 | |
| 4.3 | 与 V7.0 判决对照 | 4.1 | 同数据栈差归因 | |

---

## 三、LF + Megatron 集成的三个关键决策点

### 决策 1：linear_attention 实现方案

| 方案 | 描述 | 优点 | 缺点 | 工作量 |
|---|---|---|---|---|
| A. HF 移植 | 把 transformers 的 Gated DeltaNet 逐行移植到 mcore ModuleSpec | 最忠实 | 大量代码 | 3-5 天 |
| B. mcore MambaLayer 适配 | 用 mcore 0.18.2 的 MambaLayer 如果支持 | 最少代码 | 需验证是否兼容 | 1-2 天（如果不兼容=0） |
| C. 占位→逐层替换 | 先用恒等/零占位跑通 pipeline → 逐层替换 | 不阻塞 | 语义不对 | 0.5 天 + 后续 |

**建议：C 先行跑通 pipeline（不阻塞），同时调研 B，B 不行则 A。**

### 决策 2：mhc 实现

| 方案 | 描述 | 优点 | 缺点 |
|---|---|---|---|
| A. layer wrapper | 自定义 LayerWrapper 包装 mcore TransformerLayer，在 forward 中做 hc | 最干净 | 需理解 mhc 精确语义 |
| B. pre/post hook | forward hook 做 hc | 最快 | hook 内不能改 residual 流 |

**建议：A（wrapper）——hc 是 Flash 架构的核心创新，需要精确实现。**

### 决策 3：Flash 权重加载策略

| 方案 | 描述 | 适用 | 
|---|---|---|
| A. bf16 直填 | Flash 的 bf16 权重直接填入 mcore bf16 参数 | 如果 Flash 权重就是 bf16 |
| B. 保持 weight_scale_inv 量化 | 在线反量化（同 V8 模式但新数学） | 如果 Flash 权重是量化格式且需要保持量化训练 |

**取决于 Phase 0.1 的探针结果。**

---

## 四、跨线提醒

1. **V7.0 判决训练（W4A8）**：已跑到 step 100+ 后死亡（CP50/CP100 已存），结果待分析
2. **V6.1 线**：工作树有未提交的 V6.1 改动（modeling_glm53_w4a8_pipe_mtp.py + test_v61_contract_kl.py）——**不混入 V10 commit**
3. **V8 线**：已冻结（45d2115），代码/测试/铁律保留供未来复用
4. **.38 卡**：空闲可用

---

## 五、V10.3 → On-policy 转段 TODO(2026-11-24 用户裁决)

**裁决**:当前 Arm A(off-policy,cmb_eagle_30k)保留跑完 100 步 → 382 评测 →
过门后启动 **on-policy 蒸馏 + 基于 on-policy 数据的新训练(B1)**。
假设:**数据侧 on-policy(前缀轨迹 = 主模型自己的)效果会更好**。

### 5.1 分析定论(为什么换 on-policy)

| 层 | Arm A(off-policy,当前) | Arm B(on-policy) |
|---|---|---|
| 数据层:语料"下一个词"谁写 | R1 蒸馏文风 + 人类文本(外部静态语料,cmb_eagle_30k) | **Flash 主模型自己 greedy rollout** |
| 目标层:草稿损失拟合谁 | main(已 on-policy 化:KL(main‖draft) 全词表 0.8 + agree-CE 0.2,DRAFT_W=0) | 同左 |
| serving 层:谁写草稿 | MTP 模块 | MTP 模块 |
| 残余缺口 | **前缀 x≤t 是语料轨迹,不是 main 的**(暴露偏差留在输入侧,目标侧修不掉) | 无(与 serving 轨道同源) |

实锤:cmb_eagle_30k 样本 assistant 以 "Okay, let's try to figure out..." 开头
(R1 思考文风);dacc 归因显示 R1 文风样本正是低接受率重灾区(step 16/18/21/24 zh)。
目标侧修正(V10.2/V10.3 的 KL/agree)已被 382 bench 证明不够(V10.1 K̄ 3.52→2.34)。

### 5.2 转段任务表

| # | 任务 | 依赖 | 产出 | 验证 |
|---|---|---|---|---|
| T1 | Arm A 跑完 100 步 → CP100 | 训练中(.40,~04:10) | /tmp/v103_saves/cp100 | R8 分桶/EVAL-FIXED 趋势 |
| T2 | CP100 导出 serve dir | T1 | symlink farm + delta 台账 | 冻结键跳过台账核对 |
| T3 | **382 bench 终审(发令枪)** | T2 | K̄ | ≥2.34 及格 / 3.15-3.28 目标 |
| T4 | 发令枪判定 → 若过门,启动 on-policy 蒸馏 | T3 | rollout 放 **.38**(16 卡空,不占 .40) | 语义 CE 门先过(垃圾生成器不 rollout) |
| T5 | rollout:30000 prompts(onpolicy_prompts.jsonl 已备),greedy,CUTOFF 4096 | T4 | 分片 jsonl(DP2×TP8,断点续跑) | 抽样人工核查文风=Flash |
| T6 | pack:flash_onpolicy_<domain>.jsonl + merged all.jsonl | T5 | LF 无感替换数据源 | 域分布计数 |
| T7 | **B1 训练**:.40,配方=V10.3 逐字不变,**只换数据**(单变量) | T6 | CP + 382 bench | 与 T3 同口径对照 |
| T7b | **B1b 消融**:`--lr 0` 冻结主模型(LoRA 组 LR=0,纯草稿蒸馏;on-policy 下 main_ce 退化为自我锐化,锚静止=draft 匹配的即 serving 那个 main) | T7 | B1b CP + 382 bench | B1b vs B1:K̄ + EVAL-FIXED agree 爬速(锚静止,预判 B1b 收敛更快);若优,后续 on-policy 轮默认冻结主模型只训草稿 |
| T8 | 判决:on-policy vs off-policy 同 prompt 域内对照 | T3+T7 | per-domain 接受率 diff | 唯一变量=response 分布 |

T7b 工程注记:LR 分组 LoRA 组 lr=args.lr(v10_trainer.py:366),置 0 即冻结;GRAD AUDIT
`lora-grads=0` 属预期(打印不强制),R8 分桶只看 mtp 不受影响;export 时 LoRA 增量为零、
serve main = 纯 vendor base,CP 台账记录。

### 5.3 工程优化点随 B1 一并落地

Sequence Packing(+block-diagonal causal mask)或 grad_accum=8;cutoff 2048
对齐 serving 长 `<think>` 链(显存余 ~10G,先 OOM 探针)。详见
`docs/V10.3_TRAIN40_OPTIMIZATIONS.md`。

### 5.4 纪律

- B1 单独回答"on-policy 数据是否有效";B2(软标签)不混入。
- rollout 产物 = Flash 自生成,**禁混外部语料**;同 prompt 域内对照保单变量。
- Arm A 的 off-policy 基线数字(T3)必须先落档,B1 才有对照锚。
