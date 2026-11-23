# RFC-20261113-001 伴侣文档：LlamaFactory 架构与功能结构分析（SFT+MTP 视角）

> 状态：v1（2026-11-13）| 非规范性伴侣文档（RFC 附录 A.3 挂载）
> 目的：①盘点 LF 架构与 SFT 单目标假设；②MTP 生态现状与成功案例调查；
> ③按 RFC 要素定位 GAP；④多机专项；⑤大颗粒特性设计（LF-MTP-F*）。
> 事实来源：LF 本地副本 `83788da4`（上游 `4759a8de`+patch）、transformers
> 5.14.1（26 容器 site-packages 实测）、本会话实测记录。

---

## 1. 结论摘要（TL;DR）

1. **LF/HF 生态不存在 SFT+MTP 联训先例**：HF transformers 的 deepseek_v3 在
   config 保留 `num_nextn_predict_layers`/`num_mtp_layers` 字段但 modeling 中
   **MTP 实现为零**（MTP 权重不进 HF 格式是生态惯例）；glm_moe_dsa 原生同样
   无 MTP。LF 全量源码检索无任何 MTP 训练逻辑。
2. **我们的 branch-1 + trust_remote_code 模型包 = 生态首个 LF-based
   SFT+MTP 联训实现**（V1→V3 演进，V2 2000 步运行 + V3 探针运行中 = 可用性
   实证）。没有可抄的成功案例 → 大颗粒特性设计（§7）必须自研，但 LF 的
   "loss 由模型侧返回"架构给了实现自由度。
3. **GAP 定位**：LF 的架构性约束是**单目标 CE 假设**（labels→lm_head CE，
   trainer 与模型解耦）+ **单机假设的工程细节**（sampler/checkpoint/日志）。
   MTP 所需的链式几何、多目标、draft 参数组、draft 权重存取全部落在模型侧
   补丁——短期可行（现状），长期需要 LF 特性化（§7 提案）。

## 2. LF 总体架构与 SFT 数据流

```
llamafactory.cli → launcher (FORCE_TORCHRUN → torchrun --nnodes/--node_rank/
                 --master_addr/--master_port；NPROC_PER_NODE=get_device_count)
  └─ run_exp → _training_function → run_sft
       ├─ data: dataset_info.json → load_dataset(num_proc) → template 对齐
       │        → tokenize (preprocessing_num_workers) → collator
       ├─ model: loader → patcher (attention/moe/quantization/rope util 族)
       │        → trust_remote_code 包 或 native 类 → accelerator 准备
       ├─ optimizer: create_optimizer（branch-1 时 fp32 分组补丁生效）
       └─ trainer: transformers Trainer 基类
            ├─ 默认: compute_loss → model(**inputs).loss → accelerator.backward
            └─ branch-1 (supports_multi_microbatch): training_step 直接调
               model.multi_microbatch_loss(**inputs)——**backward 在模型内**，
               返回 loss.detach()；模型可完全接管 loss 构造与梯度路径
```

**架构性事实（对 MTP 的意义）**：
- LF 的 loss 由**模型 forward 返回**，trainer 只负责 backward 与循环 →
  多目标/链式几何可以完全在模型侧实现而不动 LF 核心（我们当前的路线）。
- 数据/模板/optimizer/checkpoint 四个环节 LF 有明确扩展点，但均按**单目标
  SFT** 设计。

## 3. LF 的单目标假设与四类耦合点

| 环节 | LF 原生行为 | 对 MTP 的约束 |
|---|---|---|
| labels | collator 产出单列 labels（supervised mask） | MTP 需要逐深度目标（可由 labels 派生，无需多列——recursive/契约 C 均从单列派生 ✓） |
| loss | 模型返回标量 loss | 自由度：模型内部可组装任意多项（我们已用） |
| optimizer | create_optimizer 按 param 分组（我们补丁注入 fp32+双速 lr） | draft 参数组/冻结调度需模型侧声明（补丁已接） |
| checkpoint | Trainer.save_model + optimizer/scheduler | draft 权重需独立语义（我们的 _save 补丁 + build 合并管线） |
| eval | 评测=EVAL loss，无接受率概念 | 接受率探针在 LF 外部（run32_v* 管线） |

## 4. MTP 生态现状盘点（成功案例调查）

| 层 | 对象 | MTP 处理 | 证据 |
|---|---|---|---|
| HF transformers 5.14.1 | deepseek_v3 | config 含 `num_nextn_predict_layers`(=`num_mtp_layers`) 字段；**modeling 零 MTP 实现**（无 MTP 层类；HF 格式不含 MTP 权重） | 26 容器 site-packages 实测（grep=0/类列表） |
| HF transformers 5.14.1 | glm_moe_dsa | 无 MTP（vendor 家族的 MTP block 只存在于 ModelSlim checkpoint + 我们的 trust_remote_code 包） | 同上 |
| LF 模型注册表 | 无 MTP-bearing 模型 | LF 按"无 MTP 的 SFT 目标"对待这类模型 | 全量源码检索 |
| LF src "mtp" 命中 | 仅我们的补丁（sft/trainer.py branch-1/_save、collator h0 列注释）+ ppo/trainer.py 无关注释 + **陈旧 pyc**（model/__pycache__/glm53_mtp.cpython-311.pyc——历史上存在过已删除的 LF 内嵌模块，属考古痕迹） | — | 检索记录 |
| 业界 | DeepSeek 官方仓库 / speculators 等自带自定义 trainer | 均为 LF 外的自研循环 | 公开资料（非 LF） |

**结论：无成功案例可借鉴。** 现有可用实证 = 本项目 V2（2000 步）+ V3（探针
运行中）——即"LF-based SFT+MTP"的首个落地是我们自己。这既意味着 GAP 定位
只能靠 RFC 要素逐项对账（§5），也意味着我们的模型侧实现有成为生态参考实现
的潜力（§8 落地路线）。

## 5. GAP 矩阵（RFC 要素 × LF 能力 × 状态）

| RFC 要素 | LF 原生 | 我们的补丁（现状） | 剩余 GAP |
|---|---|---|---|
| 链式几何（REQ-LOSS-2） | ✗（无 MTP 概念） | 契约 C 在模型侧实现；路由 env（V5-S1） | 契约 C 在 branch-1 的 canary（M3） |
| 多目标 loss（REQ-LOSS-3） | ✗（单 CE） | ACC margin 已入两契约 | 契约 C CE 深度加权（OPEN-3） |
| 参数参与/分组（REQ-ARCH-2/3, OPT-1/2） | create_optimizer 单一分组 | fp32+双速 lr 补丁 + freeze 钩子 | 声明式参与矩阵（§7 F3） |
| 精度链（REQ-RT-5/6） | autocast bf16 | fp32 master + CPU AdamW 补丁 | 契约 C 精度链实测（M3） |
| 数据列（REQ-DATA-3） | 单 labels 列 | h0 列 collator 补丁（MTP-only 轨道） | 多目标列标准化（§7 F4） |
| draft 权重存取（REQ-SRV-*） | save_model 全量语义 | _save 补丁（mtp_ft+adapter）+ build 管线 | 合并管线契约化（DFX-2 已守卫） |
| 评测（§15） | LF eval=loss | run32_v* 外部管线 | 接受率钩子内置（§7 F6） |
| 多机（REQ-DIST-*） | torchrun launcher+sampler | HCCL 配方+NO_MANUAL_ALLREDUCE+SaveOpt | §6 专项 |

## 6. 多机专项（用户点名维度）

### 6.1 LF 多机机制现状

- **launcher**：`FORCE_TORCHRUN=1` + `NNODES/NODE_RANK/MASTER_ADDR/MASTER_PORT`
  → torchrun 静态组网（launcher.py 实测）；`NPROC_PER_NODE` 缺省=可见卡数
- **数据**：DistributedSampler(seed=42, num_replicas=world)——**2 机 32 卡改变
  分片数**（16→32 副本）：消费序列与单机 16 卡不再逐位一致（V2_HANDOFF 已
  文档化；8 卡 GA2 的配对技巧=保序对照的既有方案）
- **梯度同步**：DDP hook all-reduce（GLM53_NO_MANUAL_ALLREDUCE=1，禁手动双
  重同步）；branch-1 模型内另有手动 allreduce 分支（旧路径门控保留）
- **checkpoint**：rank0 存模型；GLM53_SAVE_OPT 落 optimizer/scheduler
  （per-rank 语义需在多机下核验：checkpoint-30 的 optimizer_state.pt 单文件
  实测——多机时 2×rank 写同名文件的存在性待验证）
- **组网防线**：HCCL_SOCKET_IFNAME=enp162s0f0 + **HCCL_IF_IP per-node**
  （EI0015 实测）+ rendezvous 双节点日志同口核对（E-8）

### 6.2 契约 C × 多机

契约 C 是 per-item 逐条循环（B=1/rank），跨 rank 无共享状态 → **多机正交**：
每 rank 独立 item + DDP 梯度平均即可。差异点仅剩：
1. sampler 分片（数据口径变化，需 32 卡版阶梯核对）
2. SAVE_OPT 多 rank 文件语义（核验项）
3. 双机 loss=203（OPEN-5，V4(已 DROP) LF-native 遗留——我们的 branch-1 集成面不同，
   canary 先行）

### 6.3 多机 GAP 清单

| GAP | 影响 | 动作 |
|---|---|---|
| SAVE_OPT 多 rank 语义 | 续训完整性（G-A4） | 2 机时先 1 步 smoke 核验文件布局 |
| sampler 分片口径 | 对照基线（跨拓扑不可比） | 32 卡臂 MUST 自建基线（ADR-12 口径不变） |
| 双机 contract C 内存 | 未见差异（per-rank 独立） | M3 canary 覆盖 |
| rendezvous 防线 | 启动失败类 | E-8 已固化 |

## 7. 大颗粒特性设计（LF-MTP-F*，把 SFT+MTP 特性化进 LF）

| ID | 特性 | 动机（REQ 映射） | 接口草案（伪代码） | 工作量 | 风险 |
|---|---|---|---|---|---|
| **F1** | **MTP 训练契约抽象**：模型侧声明 contract（recursive/serving_c/插件），LF 只透传 | LOSS-2 | `class MTPContract(Protocol): def depth_logits(model,h,ids,k)->logits`；模型注册表 `model.mtp_contracts={...}`；trainer 按 env 选择 | 中 | 契约间数值不等价（需阶梯探针门控） |
| **F2** | **多目标损失引擎**：weighted terms（CE + Σ depth CE + 对齐项插件），深度权重表 | LOSS-3/4 | `MultiTermLoss(terms=[Term("ce_main",w), Term("mtp_ce",w,depth_ws=...), Term("acc_margin",w,margin=...)])`；sum/count 口径统一 | 中 | 项间梯度冲突（w 起步小值） |
| **F3** | **draft 参数组管理**：声明式参与矩阵（组件→参与方式/lr/freeze 步）→ 自动分组 | ARCH-2/3, OPT-1/2 | `param_matrix=[{"match":"layers.78.*","mode":"fp32_master","lr":1e-5}, {"match":"*.lora_*","mode":"lora","lr":1e-4,"freeze_at":50}]`；create_optimizer 消费 | 中 | 命名匹配跨 DDP 包装 |
| **F4** | **多目标 collator/数据列**：per-depth 目标派生规格化（shifted labels 即可）、h0 列、supervised mask 契约 | DATA-3 | collator 输出 `labels`（单列）+ 模型侧派生；可选 `aux_columns` 声明 | 小 | 与 packing 兼容 |
| **F5** | **draft 权重存取语义**：checkpoint 内 draft 权重独立文件 + 合并管线钩子 + no-overwrite | SRV-1/2 | `_save` 补丁泛化：`draft_files=["mtp_ft","adapter"]`；merge 钩子注册表 | 小 | 已有雏形（现补丁即 F5 子集） |
| **F6** | **接受率评测钩子**：训练中周期性 draft 链探测（CE 阶梯/hit 率），把成熟度探针自动化 | EVAL-1 | `probe_every=N` → 调 `_mtp_serving_chain_metrics`（现成）→ 落 jsonl | 小 | 探测占步时（限 rank0+低频） |

**设计原则**：全部特性走 **opt-in**（缺省关闭 = 纯 SFT 行为不变），与我们的
env 门控纪律一致；特性间正交可单变量验证。

## 8. 落地路线（三选/组合；绿地原生设计见姊妹文档 RFC-20261113-002-LF-MTP-NATIVE-DESIGN.md）

| 路线 | 内容 | 适用 | 时点 |
|---|---|---|---|
| **短期（现状）** | 模型侧补丁 = branch-1 + trust_remote_code 包 + env 门控（S1~S3 已实现） | 探针/对照期 | 现在 |
| **中期** | LF fork 特性分支：把 F1-F6 固化为 warpack patch v2（patch 面从 3 钩子扩到 6 钩子） | 多环境复制、降低模型包复杂度 | V5 出数后 |
| **长期** | upstream PR：GLM/DeepSeek 通用 MTP 接口抽象（F1/F2 通用化）进 LF 社区 | 生态化 | 有余力时 |

## 9. 与 RFC 的映射与开放问题

- 本文档挂载于 RFC 附录 A.3；GAP 矩阵 §5 的"剩余 GAP"列即 RFC OPEN 项的
  LF 侧视图（OPEN-5 loss=203、OPEN-6 DSA、OPEN-7 泛化评测）。
- 多机差异（sampler 分片口径）已回写 ADR-12 注记。
- 新增开放问题：OPEN-9 SAVE_OPT 多 rank 文件语义核验；OPEN-10 glm53_mtp pyc
  考古（确认无残留依赖后清理）。
