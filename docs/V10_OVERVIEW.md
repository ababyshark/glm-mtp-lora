# V10 训练说明（通俗版）——与“M6 serving 抽测解释”同风格的端到端介绍

> 2026-11-20。给非深度参与的听众：V10 训练是什么、怎么做、怎么知道有效。
> 术语细节回查附录 A（V11_PLAN）与 V10_TRAIN.md；本文用与 M6 解释相同的
> “原理 → 为什么是裁决 → 步骤 → 悬案 → 前置”五段式。

---

## 1. V10 训练是什么（原理）

**一句话**：给 GLM-5.3-Flash 主模型旁挂一个“草稿层”（MTP），训练它模仿
主模型的下一 token 预测——草稿越像主模型，推理时的投机解码接受率越高。

```
主模型（45 层，LoRA 微调）:
  tokens → embedding → 45 层 decoder → hidden h(t)
    ├─ 主头: lm_head(h(t)) → 主模型对 t+1 的预测        ← main CE 训它
    └─ draft 入口: h(t) + e(t)（当前 token 的 embedding）
                        ↓
草稿层（MTP，全参微调）:
  融合(e(t), h(t)) → 单层 transformer → 共享 lm_head → draft 对 t+1 的预测
                                                          ← draft CE 训它
loss = main CE + draft CE     ← 两项共享目标（t+1），分别训主模型行为与草稿
```

**为什么两项 loss 联合**：主项让主模型行为保持/变好（LoRA）；draft 项让
草稿逼近“移动中的主头”——联合训练消除两者各自演化的 co-adaptation 缺口。

## 2. 为什么这能提升接受率（机制链）

```
draft 逼近主头分布（draft CE ↓）
  → argmax-agree 上升（draft 的 top-1 = 主头的 top-1 更频繁）
    → 投机解码接受率上升（draft 提案被主模型采纳）
      → 推理加速倍数上升（一次前向验证多个 token）
```

**实证链（已闭环）**：CP50 评测 main CE 0.89→0.53、draft CE 1.37→0.71、
argmax-agree 0.755→0.896（+14 点/50 步）；长跑持续上行（0.9141@600）。
serving 级实测（M6 式抽测）是最后一环。

## 3. 训练的五个步骤

| 步骤 | 内容 | 命令/位置 |
|---|---|---|
| 1 启动 | 16 卡 TP16，CP50 续训（fp32 master+动量恢复，步数接续）| launch_v10_train_40.sh 1000 --load-cp ...cp50_rank{rank}.pt |
| 2 监控 | 每步：loss/S/gnorm/fwdbwd/opt/相位/时间戳；每 10 步：双轨 diag | h0-test/eval35/v10_train_40.log |
| 3 诊断 | 固定 eval batch：main/draft CE + argmax-agree（趋势）；train batch（信号）| diag step N 行（非致命——失败只告警）|
| 4 CP | 250 步一存（fp32 master+动量，16 rank 分片，~92GB）| saves/v10_t4_long/cp{N}_rank{r}.pt |
| 5 效果评测 | 任一 CP：fresh-fill 基线 vs 加载后，同 token 同路径 | scripts/v10/eval_cp50_40.py（改 CP 路径）|

## 4. 顺带裁决的悬案（训练本体之外的三个未知数）

| 悬案 | 状态 | 裁决方式 |
|---|---|---|
| eh_proj 列序（swap/no-swap） | **已终审**：checkpoint 原序正确（12.72 vs 1.37 探针），fill 默认不交换 | 已闭环（无悬案）——若 serving 出问题可复查 GLM53_V10_EH_SWAP |
| draft 输入 e(t) vs e(t+1) | **已终审**：e(t) 正确（no-roll patch，E4 实测 11.73 vs 12.72）| 已闭环 |
| V5.4a 配方迁移（KL/agree/margin + 深度展开） | **待 A/B**：native 纯 CE 基线跑完后切 RECIPE=1（单变量）| M5 后 |
| pad/掩码策略（全 LM vs assistant-only） | **待 A/B**：全 LM 现状 vs 掩码 | 配方迁移后 |

## 5. 前置条件与已知边界

| 项 | 状态 |
|---|---|
| 机器 | .40 单机 16 卡（TP16 = 61.28G 卡的显存下限，不可减）|
| 显存 | B=1 + recompute=full ≈ 47.7G/卡 ✓；B=4 待重测（fp32 物化 hog 已修）|
| recompute | 只能 full（off 激活账 72G 不可行）；torch_npu 的非 reentrant 释放机制失效（已知）|
| 数据 | cmb_eagle_30k（1000 步仅消耗 3.3% ✓）；330k 备用 |
| 框架缺口 | KDA/mHC/MTP 配对——框架没有的实现（V10 自有 4654 行）；MTP 的 2 个 mcore patch |
| 崩溃恢复 | CP 续训已验证 ✓；诊断/CP 已非致命化 ✓ |
| 悬置 | MindSpeed-MM 混装 import（待清理）；多深度展开（V11 配方迁移）|

## 6. 当前状态与判据

- **长跑**：1000 步进行中（step 600+，gnorm 平台 2-3，无 nan，显存稳定）
- **判据**：EVAL-FIXED main/draft CE 与 argmax-agree 的斜率（step 600 读数：
  0.9946/0.9957 + agree 0.9141 train-batch ✓）
- **下一步**：step 1000 完成 → CP1000 评测 → serving 抽测（M6 式，
  含 eh 双份 A/B）→ 依据接受率决定规模化配方

## 7. serving 抽测（V10 版 M6 协议）

| 步骤 | 内容 |
|---|---|
| 1 导出 | CP1000 masters → bf16 回写 → vllm-ascend Glm5Next 格式（含 draft）|
| 2 服务 | vllm-ascend 起服务 + 投机解码配置（draft = 我们的 MTP）|
| 3 测量 | 固定 prompt 集 × 投机解码：接受位图 + 接受率（k 步分别）|
| 4 对照 | fresh fill 同协议——Δacceptance = 训练净效果 |
| 5 判据 | 接受率 ≥ fresh 基线 + 5 点 = 训练有效性的 serving 级确认 |

前置探明项：vllm 0.30 + vllm-ascend 对 glm5_next 的投机解码支持形态
（speculative_config 挂 draft 的方式）——launch 前需一次 probe。


---

## 8. V10 体外实现 vs V11 框架内实现——逐组件对比分析

> 核心判断（用户方向确认）：**两者在数学与功能上本质等价**——V11 的
> "迁入 LF"是**代码住所与所有权的重构**，不是数学重写。逐组件对比如下
> （V11 目标形态详见 V11_PLAN §二）。每组件附迁移验收判据。

### 8.1 逐组件对比表

| 组件 | V10（体外实现）| V11（LF 框架内）| 功能等价？ | 差异本质 |
|---|---|---|---|---|
| 训练循环 | v10_trainer.py 独立 torchrun 脚本（batcher/loss/优化器/CP/diag 单文件闭环）| llamafactory/train/mcore_bridge 模块，同一循环经 LF plugin 表面暴露 | **✅ 数学逐位等价**（同 seed 同输入 → 同 loss 轨迹）| 住所与所有权：体外=独立快迭代；LF=配置/审计/日志集成 |
| 数据管线 | load_lf_data（LF dataset）+ McoreBatcher（自研，纯随机）| 同 batcher + 长度分组/打包（batcher 内实现，LF DataArguments 修复 group_by_length）| ✅ 数据面等价（打包改变布局但 full-LM 语义不变）| 打包是 V11 的**真改进**（不是纯搬家）：步时方差消失+pad 回收 |
| loss | native_loss（chunked 可微 gather CE，main+draft 双项）| 同一 loss 类注册进 LF loss 面 | ✅ 逐位等价 | 无数学变化；LF 面获得可配置性 |
| 优化器 | fp32 master + AdamW + p.grad=None + clip/warmup | 同一实现经 LF optimizer 插件 | ✅ | 同上 |
| LoRA | v10_lora hook 注入（TP 正确、fill 键名零破坏）| **同一实现保留**——peft 不认识 mcore TP 线性层，hook 式在两个版本里都是正确做法 | ✅ | 无（hook 是 TP 场景的正确形态，不是临时方案）|
| CP | rank 分片 fp32 master torch.save | 同一实现经 LF checkpoint 扩展 | ✅ | LF 面获得统一命名/保留策略 |
| 诊断/审计 | 双轨 diag + GRAD AUDIT + nan-guard + 相位计时 | 同一实现 → TrainerCallback（on_step_end）| ✅ | LF 面获得统一日志面板 |
| 模型构建 | build_v10_model（meta→fill→to_empty）| 同一构建经 LF 的 model 注册面调用（model_name_or_path 解析到 mcore 构建）| ✅ | 调用入口变化，构建代码不变 |
| mcore patch | 运行时 monkeypatch（v10_mtp.py：no-roll + padding_mask）| **mc0182 patch 文件扩展**（先例机制：环境的一部分，可复现可审计）| ✅ | patch 化后 monkeypatch 依赖消失（import 顺序风险消除）|
| 分组 MoE | V10GroupedExperts（pad+bmm，已写未接线）| 同一模块接线进 spec（mcore 内核保持）| ✅ | 接线即生效（V11 吞吐改造）|

### 8.2 “本质作用一样”的精确含义

1. **数学不变**：迁入是代码住所重构——forward/backward 的每一条计算路径
   保持逐位一致（迁移验收 = 同 seed 同数据 → loss/梯度轨迹逐位对比）。
2. **所有权重语义不变**：LoRA hook、MTP 全参、基座冻结——两个版本完全
   相同（唯一的功能差异是 V11 的打包/分组 MoE，那是**改进**不是搬家）。
3. **改变的是三件事**：① 所有权（代码从独立脚本迁到 LF fork，维护/审计
   走 LF 表面）；② 可复现性（monkeypatch → patch 文件，环境即代码）；
   ③ 操作面（配置/日志/检查点走 LF 约定，团队统一操作）。

### 8.3 迁移的验收判据（每组件一道闸）

| 组件 | 迁移验收 |
|---|---|
| 训练循环 | 同 seed 同数据 3 步：loss/gnorm/grad 逐位对比（迁前 vs 迁后）|
| 打包数据 | 打包 vs pad 的 EVAL-FIXED 斜率对比（打包是新改进，单独 A/B）|
| loss/优化器 | 同上（逐位）|
| CP | cp 迁移后 resume 的 loss 连续性 |
| 模型构建 | fill 台账 unwritten=∅ + 参数数 parity |
| patch 化 | 无 monkeypatch 生效日志 + 同 seed 3 步逐位 |
| 全链 | 迁移后 EVAL-FIXED agree 轨迹与迁前重合（同 CP 续训对照）|

### 8.4 为什么现在做（V11 时机）与不做的风险

- **现在做的收益**：① 配置/审计/日志统一（团队操作面）；② monkeypatch
  消除（import 顺序风险）；③ group_by_length/打包修复进 fork（ trap #8
  根治）；④ 双 agent 编辑权收敛（v10_trainer 收缩为薄 glue——本次
  start_step 编辑被覆盖的教训）
- **不做的风险**：体外栈与框架演化的分叉持续扩大；操作面分裂（LF 的
  配置/审计/团队约定不覆盖我们的训练）；关键修复散落在 monkeypatch
  （不可审计、可被覆盖——本次竞态的根因）
- **迁移成本**：~2 周（V11_PLAN §二.1 的六项）；逐组件验收闸保证
  迁移零数学回归


---

## 9. V10 与 V11 的关系——解耦质量决定长效性

> 用户确认：**V10 如果解耦好，也是一个很好的、可长效的方案**——
> V10（体外）与 V11（框架内）不是二选一的对错题，而是同一套已验证数学的
> 两种"住所"，**选择标准 = 解耦质量与长期维护面**。

### 9.1 两条路的共同前提：解耦纪律

V10 的长效性取决于它与框架（mcore/LF/torch_npu）的**接触面是否干净**。
当前违反项（全部有修复路径，且与 V11 的 M2 动作重合）：

| 违反项 | 危害 | 修复（M2，两条路共享）|
|---|---|---|
| MindSpeed-MM 混装 import | 版本解析顺序是隐形变量 | 容器 PYTHONPATH 清理 |
| 运行时 monkeypatch（MTP ×2）| 修改不可审计、可被 import 顺序绕过 | patch 文件化（mc0182 机制先例 ✓）|
| v10_trainer 双 agent 共写 | 编辑被覆盖（本次 start_step 的教训）| 迁入 LF fork 或编辑权收敛 |

**M2 完成后：V10（体外加固）与 V11（迁入）在解耦前提上汇合——分叉点只剩
"代码住所"。**

### 9.2 两条路的长效性对比

| 维度 | V10（体外，解耦加固后）| V11（迁入 LF fork）|
|---|---|---|
| 数学/修复 | 同一套（§8 已证功能等价）| 同一套 |
| 迭代速度 | **快**（无框架 API 约束，单文件闭环）| 慢（框架接口约束）|
| 团队操作面 | 独立脚本（需自文档）| **LF 统一表面**（配置/审计/日志）|
| 框架演化跟随 | 手工重新对齐（每次 LF/mcore 升级）| **自动**（框架内代码随框架升级）|
| LF 能力完备性 | 不贡献 | **贡献通用 mcore 训练桥（所有 mcore 模型受益）**|
| 上游路径 | 无自然路径 | fork → upstream（长期维护卸载）|
| 双 agent 竞态 | 收敛（trainer 薄 glue）| 收敛（结构化模块）|

### 9.3 决策

- **短期**：两条路共享 M2（patch 化 + 环境清理）——先跑完当前 1000 步
  长跑（CP1000），期间完成 M2
- **分叉点**：M3-M4（分组 MoE 接线 + LF 迁入）——技术上 M3 两条路都需要，
  M4 是 V11 特有
- **判断**：若 V10 解耦加固完成（M2 + 编辑权收敛）且长跑指标持续改善，
  **V10 体外是可长效的方案**（独立快迭代 + 干净接触面）；V11 的增量价值
  在 LF 能力完备性与团队操作面——**由团队按运维需求定夺，技术上都成立**
- 本文档与 V11_PLAN 的对比表（§8）即决策依据的完整记录
