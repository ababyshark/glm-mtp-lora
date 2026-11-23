# V10 差距分析 — 为什么原生 LF+Megatron 不够，以及融入 LF 的完整路径

> 2026-11-20。基于 19 项错误台账、19 个文件的实现、.40 实测证据。
> 结论先行（2026-11-20 修订，用户方向）：**缺口 = 框架尚未支持的 Flash 机制
> ——正确的动作不是绕开框架自建平行栈，而是在框架上改进**（项目已有两个
> 成熟机制：LF fork 直接修改 + mcore patch 文件）。本分析的"不支持"均应
> 读作"尚未支持，改进路径如下节"。自研代码的归宿 = 迁入框架，而非平行存在。

---

## 一、原生 LF + Megatron 为什么不足（按层拆解）

### 1.1 模型层缺口（根本原因——架构太新）

| Flash 机制 | transformers 5.14/5.16 | Megatron-Core 0.18.2 | 我们补的 |
|---|---|---|---|
| KDA（Kimi Delta Attention） | ✗ 无此架构概念（GDN2 数学不等价：f/z/b 形状、o_norm 门控、g 公式全不同） | ✗ 同样无（`gate_lower_bound` 全仓库零命中） | GlmKDAAttention：chunk+Householder 三角修正、per-channel 衰减、conv1d 融合（299 行） |
| DSA（DeepSeek Sparse Attention + indexer） | ✗（glm_moe_dsa 有但 Flash 的 indexer 语义不同：kpool=4、ape、无 fp8/rotate） | ✗ 有骨架但 TE-coupled（NPU 非 TE 不可用）且 kpool 语义不同 | V10DSAAttention + Indexer：kpool 压缩、topk 展开、因果掩码语义（380 行） |
| mHC（hyper-connections） | ✗ 无此概念 | ✗（0.20 的 mHC 语义不同：learned-head 收缩 vs 无条件 mean） | GlmHyperConnectionModule：双门（attn/ffn 各一套）、sinkhorn 20 次、4 流架构 |
| MTP（layers.45 draft） | ✗ **HF 加载时直接丢弃 layers.45**（官方 `_keys_to_ignore` 明文） | △ 有 MultiTokenPredictionBlock 但：label 配对（self-pairing）、内部 CE（.40 栈值错 10×）、e(t+1) roll 约定、padding_mask 签名 bug 全部与 vendor 约定冲突 | native_loss v2（main+draft 双项 CE、vendor 配对、可微 gather）+ 3 个 mcore patch |
| **判定结论** | **训练目标在 HF 里不存在** | **训练目标在 mcore 里语义错误** | 必须实现 |

### 1.2 数值层缺口（错误台账 #1-19 的分布证明）

19 项错误台账全部落在我们自己写或自己粘合的层——**没有一项是框架核心 bug**
（除了 #10 内部 CE op 的 .40 值错与 #18 的 mcore gather 签名缺口——都已绕开/上报）。
这证明：**框架的"成熟"覆盖不到 Flash 的领域数学**——KDA 的 decay-mask 上三角
exp(+320)=inf（#nan 根因）、mHC 的 fp32 flat 显存、MTP 的 e(t) 配对约定——
这些正确性责任在实现者，且必须由实现者的回归测试守护。

### 1.3 NPU 训练级缺口

| 能力 | HF Trainer + accelerate | mcore 0.18.2 + torch_npu | 现状 |
|---|---|---|---|
| 训练级 TP16 | **零证据**（device_map 是推理式分层） | **V7/V8/V10 三代实证**（600+ 步 × 多 run 零故障） | 只能 mcore |
| fused AdamW dtype | — | 拒绝混合 dtype（CheckDtypeValid）→ fp32 master 必需 | 已解决 |
| 集合通信 | HCCL 经 accelerate 包装（未验证） | HCCL 原生（V8 实证） | 只能 mcore |

### 1.4 数据层缺口

- LF 的 `group_by_length`：**fork 不识别**（V6.2 陷阱 #8：HfArgumentParser 拒绝）；
  且 LF 的 sampler 被绕过（V10 数据顺序归 McoreBatcher）——长度分组只能
  落在 McoreBatcher（§V10_TRAIN 十.4）
- 预训练式固定长度打包：LF 是 SFT 风格（逐样本截断/pad）——pad 浪费 ~25%
  + 步时方差——需要 McoreBatcher 改造（已分析 §十）

### 1.5 loss 层缺口

| 项 | mcore 原生 | LF 原生 | 我们的实现 |
|---|---|---|---|
| vocab CE | .40 栈值错 10×（#10）| 依赖 HF 模型 | chunked 可微 gather CE（门证明路径）|
| MTP 配对 | self-pairing（labels 未预移位）+ chained target | 无 MTP 概念 | main+draft 双项、vendor 配对（h[t],e[t]→t+1）|
| 梯度完整性 | gather Function 无 backward（x.grad=None）| — | 自封 Function / pad+bmm / torch 原生 all_gather |
| 主项+draft 项加权 | 无（固定结构） | 无 | GLM53_V10_DRAFT_W 可调 |

---

## 二、融入 LF 的完整路径（两层含义 + 行动清单）

### 2.1 第一层：改进 LF fork——让 LF 的流程原生支持 mcore 模型训练

LF fork 是我们可修改的（/mnt/share_space/l00840580/LlamaFactory）。改进清单：

| 动作 | 工作量 | 说明 |
|---|---|---|
| 1. loss 作为自定义 loss 类 | ~3 天 | chunked 可微 gather CE 类化（V10 自有 loss 移入标准接口：接受 (model, inputs) 返回 loss）|
| 2. fp32 master 优化器插件 | ~2 天 | LF 的 optimizer 覆盖点（或 TrainerCallback 的 on_optimizer_setup）|
| 3. p.grad=None / nan-guard / warmup → TrainerCallback | ~2 天 | 标准回调 API（on_step_end / on_log），LF 继承 HF 全部回调机制 ✓ |
| 4. 双轨 diag + 相位计时 + GRAD AUDIT → callback 日志 | ~1 天 | 纯日志搬家 |
| 5. CP 分片存取 → LF checkpoint 格式扩展 | ~3 天 | 或保留 custom callback（rank 分片 + fp32 master 语义 LF 没有）|
| 6. McoreBatcher → LF dataset/sampler 桥 | ~3 天 | 长度分组/打包在此层实现（绕过 fork 的 group_by_length 缺陷）|

小计 ~2 周。**效果：训练循环的启动/调度/日志/检查点全部走 LF 表面，
我们的数学以 LF 模块的形式存在（不是平行栈）。**

### 2.1b 改进 mcore——patch 机制扩展（先例：mc0182_dpa_k_channels.patch）

我们的三个 mcore 修改目前是运行时 monkeypatch（v10_mtp.py），应**正式化
为 patch 文件**（setup_v7_env.sh 的 patch 机制，先例已验证）：

| mcore 修改 | 现状 | patch 化动作 |
|---|---|---|
| MTP checkpoint padding_mask 签名 bug | v10_mtp.py monkeypatch | 移入 mc0182 patch 文件（+padding_mask 参数）|
| MTP 输入 e(t) 约定（no-roll） | v10_mtp.py monkeypatch | 同上（或 upstream 讨论：DSv3 原文 e(t_{i+k-1}) 语义确认）|
| （待议）内部 vocab CE 的 .40 值错 | 绕开（自有 CE）| 向 CANN/mcore 上报（需最小复现）|

patch 化收益：**修改成为环境的一部分**（可复现、可审计），不再依赖
运行时 import 顺序。

### 2.2 改进 transformers（本地）——glm5_next MTP 层

transformers 5.16 原生 Glm5Next 已可加载（.40 glm53flash-train 验证 ✓），
缺 MTP 层（layers.45 被丢弃）。本地改进路径：在 glm53flash-train 的
transformers 里补 Glm5NextMTP 层实现（我们的 MTP 块迁移，~800 行），
使 HF 原生模型包含 draft——**LF 全栈直接训练 draft 的前提**。
NPU 训练级 TP 证据仍需验证（2.1b 不受影响，两路线并行）。

### 2.2 第二层：把我们的实现"贡献进"框架（代码集成，长期维护卸载）

| 贡献目标 | 内容 | 前置 | 预期收益 |
|---|---|---|---|
| **transformers（HF）** | Glm5Next MTP 层实现（draft layer + e(t) 约定 + draft head 共享） | 5.16 原生模型已存在 ✓ 只缺 MTP | LF 全栈原生训练 draft ✓ 消除自研 trainer |
| **mcore（0.20 hybrid）** | KDA/DSA/mHC 作为 mcore 一等模块（参考项目已验证 0.20 语义 ✓） | 0.20 API 适配（我们的模块是 0.18.2 接口） | V10 模型层收缩为纯 spec 配置 |
| **vllm-ascend** | grouped_matmul 的训练侧 autograd Function（dx 同款 op + dw pad+bmm，已验证 ✓） | op 的 backward 无注册（已探明） | MoE 吞吐 + 训练/推理同栈 |
| **torch_npu** | decay-mask 类数值陷阱的 op 修复建议（checkpoint-recompute 毒化、CE 值错） | CANN 团队通道 | 生态级修复 |

### 2.3 决策树（修订版：框架改进为主线）

```
现在：V10 自有实现（工作系统 ✓ 1000 步长跑中）
  │
  ├─ 近期：长跑完成 → CP250/500 评测 → serving 抽测
  │
  ├─ 中期（框架改进三件套，2-3 周并行）：
  │    ① mcore patch 化：MTP 修复移入 mc0182 patch（机制先例 ✓）
  │    ② LF fork：mcore-model 训练桥 + group_by_length 修复
  │       （V6.2 陷阱 #8 的根治）+ loss/CP 插件面
  │    ③ transformers 本地：glm5_next MTP 层补齐（glm53flash-train venv）
  │    效果：V10 的训练循环/数学全部迁入框架 = "复用成熟框架"的初衷达成
  │
  ├─ 中期（并行）：serving 接受率抽测（agree ~0.91 已到门槛）
  │
  └─ 长期：上游贡献（HF MTP / mcore Flash 模块）
       触发条件：本地验证成熟 + 团队决策
       收益：维护卸载，本地 patch 清零

### 2.4 诚实的约束说明（修订版）

1. **自研不是对复用的否定**：LF/mcore 的成熟覆盖 = "它们认识的模型"的微调；
   Flash 四机制不在其中 → 自研 = 缺口期的必然，且已按**可迁入框架的形态**
   实现（接口对齐、测试随行、机制先例在位）。
2. **框架改进不改变数学**：迁入 LF/patch 化只动代码的"住所"，flash 数学/
   约定/修复全部保留——这正是它们的不可替代价值（LF/mcore 短期内不会有
   这些实现）。
3. **路线 β 的最大风险不是工程而是证据**：NPU 上 HF-native 训练级 TP 零先例
   ——mcore 线的三代实证是我们选 mcore 内核的原因，换线前需要同等量级验证。
4. **双 agent 同文件写入竞态**（本次 start_step 编辑被覆盖的教训）：框架改进
   落地后，v10_trainer.py 的编辑权应收敛到单一 session（或迁入 LF fork 后
   v10_trainer 只剩薄 glue）。

---

## 附：证据索引

- 架构缺口：ARCHITECTURE_MAPPING.md（§1 KDA≠GDN2、§3 DSA、§5 MTP 约定）
- .40 op 实测：probe_gmm10（npu_grouped_matmul 无 backward）、
  probe_gmm11（dx/dw 路径）、probe_gmm7（group_list 约束原文）
- 错误台账 #1-19：V10_TRAIN.md（本文件 §五引用）
- NPU 训练级证据：V7/V8/V10 三代 run 日志（h0-test/eval35/v10_train_40.log 等）
