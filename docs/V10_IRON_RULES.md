# V10 铁律（2026-11-19 立法，用户令）

> **用户决策**：V8 暂停（W4A8 full-depth 调试不可收敛），切换到 **GLM-5.3-Flash**
> 作为 V10 的训练目标。Flash 是全新架构，需要从 Phase 1 重做。
> V8 的所有产出冻结（代码/测试/铁律），后续可复用。
> 本文档是 V10 的不可违反规则。
>
> **2026-11-19 追加决策（用户令）**：
> 1. V10 线 = **LF + mcore 0.18.2 描述微调**（判定训练，同 V7.0 模式）。
>    预训练式路线（mcore 0.20 原生 megatron）由用户另行独立实现，不在 V10 范围。
> 2. `/mnt/share_space/l00840580/AIfromScratch/glm53flash-mtp/`（另一 session 遗产，
>    mcore 0.20 + MindSpeed 栈）**仅作数学参考**：其 GlmKDA/GlmMHC 模块、
>    docs/ARCHITECTURE_MAPPING.md 语义表、tiny/medium parity 结论（rel≤1.7e-06）可信；
>    其栈、训练入口、mcore 0.20 组件**不进入 V10**。0.18.2 缺失的 DSA indexer
>    需从 0.20 源码移植（移植属 mcore 内扩展，合规）。

---

## 第 0 条：栈锁定，禁止静默替换

**V10 的栈 = LlamaFactory + Megatron-Core，训练目标 = GLM-5.3-Flash。**

1. 组件锁定表每一项都是合同。任何更换/替换/绕过必须显式提案 + 用户批准。
2. 代码里出现锁定表之外的框架 import = 违规。
3. 每次交付必须列组件清单与锁定表逐项对齐。

## 第 1 条：组件锁定表

| 层 | 组件 | 版本/形态 | 备注 |
|---|---|---|---|
| 训练目标 | **GLM-5.3-Flash** | **基座 = /mnt/weight2/GLM-5.3-Flash-BF16（BF16 直存，120 shards，2026-11-19 用户裁决）**；FP8 版（62 shards）不用于训练 | 全新架构：45 层混合 attention |
| 编排/数据 | **LlamaFactory** | git tree | 数据管线待适配 Flash tokenizer |
| 训练核 | **Megatron-Core 0.18.2** | venv_v7 source install + mc0182 patch | 同 V8 |
| NPU 运行时 | torch 2.10.0 + torch_npu 2.10.0.post4 | venv_v7 | 红线不变 |
| numpy | 2.4.6 | 已装 | |
| 权重格式 | **BF16 直存，零反量化** | FP8 weight_scale_inv 块量化仅在导出 serving 用途时涉及 | fill = plain copy |
| MoE | 288 experts / topk 8 / gate fp32 | mcore SequentialMLP + router | |
| MTP | layers.45（最后层） | num_nextn_predict_layers = 1 | |

**禁止**：
- megatron-bridge、mindspeed 直用（V8 已否决）
- 将 SlimW4A8Linear 的 W4A8 反量化数学（per-channel scale/offset）套用于 Flash 权重——Flash 是 block-wise weight_scale_inv，数学不同，盲用会静默错权
- 在未跑语义门的情况下声称"green"

**非禁用、仅不涉及**：
- SlimW4A8Linear 组件本身对 Flash **本来就不涉及**（不是禁用对象）。若 Phase 0 探明 Flash weight 为 packed 格式，其 packed-residency 骨架（packed 存储 + forward 时反量化）改数学后复用；若 bf16 直存则 plain copy，同样不涉及。

## 第 2 条：Flash 架构关键差异（vs W4A8）

Flash **不是**小一号的 W4A8——是全新架构：

| 维度 | W4A8 | Flash | V10 影响 |
|---|---|---|---|
| 层类型 | 全部 MLA/DSA | **混合： 34 linear_attention + 11 DSA** | mcore 需支持 linear_attention——这是 Phase 1 最大的技术风险 |
| qk_rope_head_dim | 64 | **0（无 rope 维度）** | MLA 无 rope 部分 → rope 相关的 NPU bug 全部消失 |
| qk_nope_head_dim | 192 | 256 | head dim 变化 |
| q_lora_rank | 2048 | 1536 | 投影维度变化 |
| hidden_size | 6144 | 4096 | 所有线性层维度变化 |
| experts | 256 | 288 | expert 数量变化 |
| MoE 层 | 75 | 42 | 层分布变化 |
| **quant 格式** | ModelSlim per-channel scale/offset | **weight_scale_inv block-wise 128×128 fp32** | **Phase 1 必须先理解此格式的反量化数学** |
| **weight path prefix** | model.layers.N | **model.language_model.layers.N** | mapping 的 key 前缀不同 |
| **mhc** | 无 | **True（multi-head comprehension + sinkhorn）** | 全新机制——需要理解并实现 |
| **hc_mult / sinkhorn** | 无 | **4 / 20** | 同上 |
| index_kpool | 无 | 4 + compress | indexer 新特性 |
| MTP layer | layers.78 | layers.45 | mapping 层号不同 |
| weight format | 62 shards + quant index | 标准 HF 62 shards + model.safetensors.index.json | 标准 HF 格式，无 ModelSlim packed |
| head_dim | 192 | 0 | |

**Flash 的三个全新机制（Phase 1 必须理解和实现）**：
1. **linear_attention（34/45 层）**：mcore 0.18.2 是否原生支持 GLM 的 linear_attention 是最大技术风险。若不支持，需要从 MindSpeed-LLM/transformers 移植。
2. **mhc（multi-head comprehension）**：hc_mult=4 + sinkhorn_iters=20——需要理解其数学并在 mcore 中实现。
3. **weight_scale_inv 反量化**：block-wise [32,16] fp32 scale——与 ModelSlim 的 per-channel 不同，需要新的反量化公式。

## 第 3 条：从 V7/V8 继承的教训（全部不可违反）

### 数值铁律
1. **LAYOUT TRUTH**：mcore 0.18.2 端到端——**GPTModel.forward/embedding 收 [B,S]**，embedding 内部 transpose → decoder/attention **seq-first [S,B,H]** → logits [B,S,V]。本轮在 Flash 上重新验证（不同 config 可能行为不同）。
2. **语义 CE 门（新增，最高优先）**：build 之后、任何训练之前，主头在**真实文本**上的 next-token CE 必须落在 vendor 量级。这一个检验替代所有探针迭代。**V8 教训：跳过此门导致 15+ 轮无效调试。**
3. **norm eps 按名字分配**：q/kv_layernorm eps=1e-6（transformers 默认），主 norm eps=rms_norm_eps。
4. **activation = SiLU**（不是 GELU）。
5. **mtp=False ⇔ config.mtp_num_layers=None**。
6. **rope 公式**：mcore [evens|odds]+rotate_half ≡ vendor interleave（已数学证明）。但 Flash 的 qk_rope_head_dim=0——MLA 无 rope 维度，需确认 linear_attention 层的 rope 处理。
7. **bf16 噪声地板**：单 attention 1.85e-3；4 层 logits 复合 ~12%。语义判定必须用 fp32 或 argmax。
8. **NPU FusedScaleMaskSoftmax 不支持 attn_mask_type=causal 的内部 causal 掩码**——必须传显式 bool causal mask。
9. **masked_fill_ 在 NPU 要求 BOOL mask**（float/bf16 mask 会崩）。
10. **mcore 线性层显式 device=torch.cuda.current_device()** 绕过 ambient meta 上下文——build 路由必须在 builder 内部（try/finally）。

### 环境铁律
1. PATH venv 优先。
2. 装包后必查 numpy≥2.0。
3. TORCHDYNAMO_DISABLE=1。
4. 资源不抢占。

### 工作方式铁律
1. 单变量纪律。
2. **语义门先行**：build → 主头 CE → 然后才是其他任何验证。
3. 两线分离。
4. **有限 loss ≠ 语义正确**（V8 链注意力轴向互换的教训）。
5. **语义差异申报**。
6. **报告字段 detach**。
7. **每次启动前审视 diff**。
8. **逐算子中间量视图优先于隔模块推理**。

## 第 4 条：验证阶梯

1. **单元**：pytest 全绿
2. **语义 CE 门（第零步）**：build → 真实文本 → 主头 CE ≈ vendor 量级。**不过此门不得继续。**
3. **逐算子 bisect**：emb → norms → attention → MoE → 逐层 hidden 对拍
4. **1 卡 smoke**：fill 全覆盖 + forward/backward + determinism + frozen-leak
5. **TP 部署验证**：TP sharded fill + 首步 + CP round-trip

## 第 5 条：Phase 计划

- **Phase 0**：Flash config/tokenizer/weight 格式理解（weight_scale_inv 反量化数学）
- **Phase 1**：mcore Flash 模型骨架（hybrid layer types + MoE + MTP）→ 语义 CE 门
  - 最大风险：linear_attention 在 mcore 0.18.2 的支持
  - mhc 机制的实现
  - weight_scale_inv 反量化
- **Phase 2**：vendor→mcore mapping + fill + 训练循环
- **Phase 3**：TP 部署验证
- **Phase 4**：V5.4a 配方迁移

## 第 6 条：V8 可复用资产

- fill_vendor.py 的 fill 骨架（fill_model/fill_router_bias 的遍历逻辑）— 但反量化数学要改
- SlimW4A8Linear 的 packed-residency 骨架（packed 存储 + forward 时反量化）— 仅当 Phase 0 探明 Flash weight 为 packed 格式时改数学复用；bf16 直存则不涉及
- v8_trainer.py 的训练循环骨架（fwdbwd + optimizer + CP）— 数据源改为 Flash
- v8_data.py 的 LF 数据管线 — tokenizer/template 可能相同
- mc0182 patch — 同 venv 同 mcore 版本
- 测试基础设施（pytest 模式、dist fixture）
- 环境脚本（setup_v7_env.sh → 改名 setup_v10_env.sh）

## 第 7 条：评测 = vllm serve 方式（2026-11-22 用户裁决）

**评测就是推理，推理就用 serving 引擎**——禁止再手搓 mcore 前向链做评测
（那是训练栈的病：TP16 顺序专家 launch-bound、AI Core 0%、OOM 边缘）。

- 基础设施 = V5.2-V5.4 已验证流程：`build_trained_cp` 式导出 →
  `vllm serve --speculative-config deepseek_mtp` → `bench_acceptance.py`
  （/metrics 原生前缀接受计数 = 精确 K̄，非边际估算）
- 评测用例 = 当前 382 条三组用例（data/eval_groups/，逐组跑一遍即分组 K̄）
- A/B 两臂：vendor 纯目录（base）vs 导出目录（base+LoRA合并+MTP替换）
- mcore 链式探针（eval_d15/eval_groups）降级为**诊断工具**：只在链式语义
  或 fill 正确性存疑时使用，不作为交付评测
- 导出器 = v10_export_serving.py（fill 逆映射 + param_shard_mode 权威
  TP 模式），其数字必须过 verify_parity_accept.py 式对拍

## 第 8 条：方案先行 + pre-flight 清单（2026-11-22 用户裁决）

**任何新代码必须先有书面方案，再有代码；方案即验收标准。**

- 长任务启动前必须过 pre-flight 清单（本会话 4 次连环撞坑的直接教训）：
  1. 目标盘 `df`（挂载点是谁？剩余多少？是否容器可写层？）
  2. TMPDIR 指向（NFS = Triton TemporaryDirectory Errno 39 雷；/dev/shm = noexec 雷）
  3. /dev/shm 大小（vLLM TP 需 >160MB，容器默认 64MB）
  4. exec 权限（写 .so 并 dlopen 的目录不可 noexec）
  5. NPU 占用基线 + 孤儿进程（host 侧 VLLM::Worker 会逃逸容器 pkill）
  6. 既有工具优先：先查项目里是否已有验证过的同类流程（V5.x serving 流程）
- 产线单实例纪律：长流水线一次启动不碰它；清理必须覆盖宿主逃逸进程
- 估算纪律：报时间前先实测一个单元（1 批/1 条/1 分片）的耗时
