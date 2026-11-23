# RFC-20261113-002 伴侣文档：LF 原生支撑 SFT+MTP 的绿地改造设计

> 状态：v1 草案（2026-11-13）| 非规范性设计提案（RFC 附录 A.3 挂载）
> 视角：**假设 V1~V5 从未发生**，仅凭 LF 架构本身与 SFT+MTP 原理（RFC-20261113-001
> §2），重新设计"LF 如何原生支撑 SFT+MTP"。
> 姊妹文档：`RFC-20261113-001-LF-ARCH-ANALYSIS.md`（现状结构分析，本文的输入）

---

## 0. 设计输入（LF 的三个架构事实 + MTP 的四个需求族）

### 0.1 LF 架构事实（绿地设计必须尊重的边界）

| # | 事实 | 设计含义 |
|---|---|---|
| A | **loss 由模型 forward 返回**，trainer（transformers Trainer）只负责 backward/循环/日志 | 几何与目标计算天然属于**模型侧**；LF 侧做编排 |
| B | 数据面是**单目标假设**：collator 产单列 labels；模板/数据集注册/缓存按单目标设计 | 多深度目标可由单列 labels 派生（无需多列）——数据面几乎零改动 |
| C | 参数面入口 = `create_optimizer` 的分组 + Trainer 的 lr scheduler；checkpoint = `save_model` + `_save` 扩展点 | 参数参与/精度链/权重存取做成**声明式配置 → 统一消费** |
| D | 多机 = torchrun 静态组网 + DistributedSampler + DDP hook all-reduce | MTP 状态 per-rank 局部即可；无跨 rank 新需求 |

### 0.2 SFT+MTP 需求族（RFC-001 §2 性质 P1-P12 归纳）

| 需求族 | 内容 | RFC REQ |
|---|---|---|
| N1 几何 | 链式（GLM 原理=链式；vllm serving=链式）；模型族各异 | LOSS-2, FWD-2 |
| N2 多目标 | CE_main + 逐深度 MTP-CE（深度加权）+ top-1 对齐项（margin） | LOSS-3/4/5 |
| N3 参数参与 | 全景矩阵：fp32 master 全参 ∥ LoRA ∥ 冻结；双速 lr；条件冻结 | ARCH-2/3, OPT-1/2 |
| N4 精度链 | 训练域 fp32 master 累积；交付域 cast/int8/FLOAT 表达 | RT-5/6 |
| N5 权重存取 | draft 权重独立文件 + 合并管线 + no-overwrite | SRV-1/2 |
| N6 观测/评测 | 逐分量日志、生效标记、周期性链探针、双臂对照 | RT-8, EVAL-1 |
| N7 多机 | per-rank 局部状态 + DDP 平均；sampler 口径文档化 | DIST-1/2/3 |

## 1. 核心设计判断（绿地三原则）

**GD-1（几何下沉模型，编排上浮 LF）**：MTP 数学（链式几何、topk、深度循环）是
模型族私有的 → MUST 留在 trust_remote_code 模型内；LF 只消费模型产出的
**结构化损失分量**。LF 不实现任何 MTP 数学。

**GD-2（标量 loss → 结构化分量）**：现行的"模型返回一个标量、权重在模型内
写死"是 LF 集成的临时形态。绿地形态：模型返回
`dict[term_name → (sum, cnt) | scalar]`，**LF 侧 ObjectiveComposer 按声明加权
聚合并记账**。这样深度加权/对齐项权重/新增项全部变成 LF 配置，不改模型。

**GD-3（一切参与声明式）**：参数全景、精度模式、冻结调度、draft 文件——全部
从"代码里的 if/环境变量"改为 **yaml 声明 → LF 统一消费**。LF 的数据面几乎
零改动（B 事实）。

## 2. 目标架构

```
┌─ LF（新增 mtp 子系统，全部 opt-in）────────────────────────────────┐
│ yaml 新节: mtp:                                                     │
│   contract: serving_c              # 模型声明的几何键（透传）       │
│   objectives:                      # [GD-2] 损失分量声明             │
│     - {name: ce_main,  weight: 1.0}                                 │
│     - {name: mtp_ce,   weight: 1.0, depth_weights: [1,1,1,1,1]}     │
│     - {name: acc_margin, weight: 0.1, margin: 0.5,                  │
│        depth_weights: [1,1,1,2,2]}                                  │
│   trainables:                      # [GD-3] 参数参与矩阵             │
│     - {match: "layers.78.(eh_proj|.*norm.*|gate\\.weight|kv_b_proj|│
│            shared_head.*)$", mode: fp32_master, lr: 1.0e-5}         │
│     - {match: "layers.78.*.(q_b_proj|o_proj)$.*lora_",              │
│        mode: lora, lr: 1.0e-4, freeze_at: null}                     │
│     - {match: "layers.60..77.*lora_", mode: lora, lr: 1.0e-4,       │
│        freeze_at: 50}                                               │
│   precision: fp32_master_on_cpu / adam_zero1                       │
│   probe: {every: 0, kind: chain_metrics}                           │
│   draft_files: [mtp_ft_weights, adapter]                           │
│                                                                     │
│ llamafactory/train/mtp/（新模块，仿 model_utils 模式）:             │
│   composer.py   ObjectiveComposer: 按 objectives 聚合模型分量       │
│   matrix.py     TrainableMatrix: 正则匹配→参数组/fp32 cast/freeze    │
│   ckpt.py       draft_files 保存钩子 + merge 注册表                 │
│   probe.py      周期性链探针（rank0, 低频）                         │
│                                                                     │
│ trainer.training_step（泛化 branch-1）:                             │
│   if model 声明 mtp_objectives: terms = model.mtp_objectives(...)   │
│       loss = composer(terms)  # LF 记账+加权                        │
│   backward 按 precision 模式（accelerator ∥ model-side）           │
└────────────────────────────────────────────────────────────────────┘
┌─ 模型侧（trust_remote_code，只实现几何）───────────────────────────┐
│ mtp_objectives(ids, labels, **inputs) ->                            │
│   {"ce_main": (sum,cnt), "mtp_ce": [(sum_k,cnt_k) per depth],       │
│    "acc_margin": [(sum_k,cnt_k) per depth]}   # 纯数值，无权重       │
│ 几何: recursive ∥ serving_c（契约 C，vllm 源码对齐）——模型私有      │
│ 参数: 按 trainables 声明 cast fp32 / 注入 lora / 冻结               │
└────────────────────────────────────────────────────────────────────┘
```

**与现状的对照（关键反转）**：现行 V3/V5 把"权重×分量"的乘法写在模型里、
用 env 传参；绿地把它搬到 LF 的 `ObjectiveComposer`，模型只报**未加权的
sum/count 分量**——深度加权、margin 权重、新增项全部成为 LF yaml，模型几何
代码不再随配方改动。

## 3. 组件规格

### 3.1 ObjectiveComposer（LF 侧）

| 项 | 规格 |
|---|---|
| 输入 | 模型分量 dict + yaml objectives 声明 |
| 聚合 | 每项：(sum,cnt) 型 → sum/count；逐深度型 → 先深度加权（归一化）再聚合；标量型直接乘权 |
| 校验 | 模型分量名 ∉ 声明 → raise（拼错即停）；声明项缺分量 → raise；**模型声明的 mtp_depths 与 depth_weights 长度不一致 → raise**（K 口径漂移防线） |
| 记账 | 每项 (加权后均值) 进 trainer 日志（LOSS_DECOMP 的 LF 原生化） |
| 输出 | 标量 loss（backward 用）+ 分量账本（日志用） |

**工作示例**：分量 `mtp_ce = [(1.11,1000),(2.71,1000),(4.15,1000),(5.00,1000),(5.44,1000)]`，
depth_weights=[1,1,1,2,2]（归一化 1/7..2/7）→ 加权 mtp_ce =
(1.11+2.71+4.15)/7 + 2*(5.00+5.44)/7 = 0.797+2.983 = 3.78（非加权均值 3.68；
深位权重抬升 0.10——方向正确）。

### 3.2 TrainableMatrix（LF 侧）

| 项 | 规格 |
|---|---|
| 匹配 | 逐条 `match` 正则按序首匹配（先长后短），跨 DDP 包装（子串匹配） |
| mode | `fp32_master`（cast fp32 + AdamW CPU 状态）/ `lora`（参数已由模型注入，仅分组+lr）/ **`lora_inject`（LF 按声明注入：targets/rank/alpha 传入模型 fill 钩子，注入完成数与声明数 MUST 一致，否则 raise）** / `frozen` |
| freeze_at | step ≥ N 时组内 requires_grad→False（one-shot 日志 `[mtp] frozen`） |
| 落点 | create_optimizer 消费矩阵生成参数组；training_step 顶部消费 freeze |
| 冲突 | 同参数命中多条 → 取首条并告警（确定性） |
| 生效标记 | 每条 mode 的实际生效数 MUST 落日志（fp32 cast 数 / lora 注入数 / frozen 数）——REQ-RT-2 的矩阵化对应 |

### 3.3 draft 权重存取（LF 侧）

- `draft_files` 声明 → _save 时额外写独立文件（现 mtp_ft_weights/adapter 形态）
- merge 注册表：`model.mtp_merge(out_dir, vendor)` 由 build 管线调用（现
  build_trained_cp 的模型侧入口泛化）
- **resume/加载语义**：断点续训时按 `draft_files` 清单反向加载（模型提供
  `mtp_load_extras(dir)`），加载后 MUST 校验文件齐全 + 键集合一致
  （G-A4 的矩阵化对应）；缺失 → raise，禁止静默跳过
- no-overwrite / 验证块为管线纪律（不变）

### 3.4 精度链（LF 侧）

- `precision: fp32_master_on_cpu` → 矩阵 mode=fp32_master 的参数 cast fp32、
  AdamW 状态驻 CPU（ZeRO-1 分片可选）
- backward 模式开关：`accelerator`（标准）∥ `model_side`（W4A8 branch-1）——
  现有 branch-1 泛化为 precision 声明的一部分

### 3.5 探针（LF 侧）

`probe.every=N>0` → training_step 每 N 步 rank0 调 `model.mtp_probe()`（链式
CE 阶梯/hit 率，现 `_mtp_serving_chain_metrics` 泛化）→ 追加 jsonl。缺省关。

## 4. 模型侧接口（唯一契约，模型族可替换）

```python
class MTPObjectiveProvider(Protocol):
    def mtp_objectives(self, ids, labels, **inputs) -> dict:
        """term_name -> (sum, cnt) | [per-depth (sum, cnt)] | scalar
        纯数值、未加权；几何自包含（模型私有）。"""

    def mtp_param_hints(self) -> dict:    # 可选：模型建议的矩阵缺省值
    def mtp_probe(self) -> dict: ...      # 可选
    def mtp_merge(self, out_dir, vendor): ...  # 可选
```

GLM/DeepSeek 各自实现几何；LF 不感知链式细节（GD-1）。新模型族接入 =
实现 provider + vendor 权重包，**零 LF 改动**——这是绿地设计相对"逐版本
env 门控"的本质收益。

## 5. 设计决策（ADR-G*）

| ID | 决策 | 备选/权衡 | 状态 |
|---|---|---|---|
| GD-1 | 几何在模型、编排上浮 LF | LF 实现 MTP 数学 ✗（模型族私有，LE 每模型重写） | 提案 |
| GD-2 | 结构化分量替代标量 loss | 标量+模型内加权（现状）：LF 侧不可记账、配方改动动模型 | 提案（M4 后迁移） |
| GD-3 | yaml 声明式（objectives/trainables/probe） | env 门控（现状）：不可版本化、难交接 | 提案 |
| GD-4 | fp32 master 成为 first-class mode | ad-hoc 补丁：矩阵外参数无保障 | 提案 |
| GD-5 | draft 文件 first-class | 全量 save：格式破坏（HF 无 MTP 权重位） | 提案 |
| GD-6 | backward 位置声明式（accelerator/model_side） | 写死 branch-1：非 W4A8 模型无法复用 | 提案 |

## 6. 迁移映射（绿地设计 ↔ V5 现状 → delta）

| 绿地组件 | V5 现状等价物 | delta |
|---|---|---|
| ObjectiveComposer | 模型内 `_mtp_loss_recursive/_contract_c` 的加权+ACC 聚合 | 权重/深度表上浮 LF；模型只报分量 |
| TrainableMatrix | create_optimizer 补丁 + freeze 钩子 + env | 声明进 yaml；删除 GLM53_BASE_LORA_FREEZE_AT 等 env |
| draft ckpt | _save 补丁（mtp_ft/adapter） | 泛化 draft_files 声明 |
| probe | run32_v* 外部双臂 + `_mtp_serving_chain_metrics` | probe.every 内置（外部双臂评测保留） |
| contract 路由 | GLM53_MTP_CONTRACT_JOINT env | yaml `contract:` 透传 |
| precision | GLM53_ADAM_CPU/ADAM_ZEROS/branch-1 | precision 声明收编 |

**兼容律**：迁移期两形态并存（env 门控旧路 + yaml 新路），T-UNIT 双路逐位
对齐后旧路退役——与 RFC 的 A/B 纪律一致。

## 7. 验收 Gate（绿地特有）

| ID | Gate | 测量 |
|---|---|---|
| G-L1 | 等价性 | 同 checkpoint 同配置：新 Composer loss 与 V5 现实现逐位一致（±1e-6） |
| G-L2 | 配方免改模型 | 改 yaml 深度权重/新增目标项 → 模型代码 0 diff |
| G-L3 | 新模型族接入 | 仿 DeepSeek MTP 假想包：仅 provider 实现 + 权重包即训练（LF 0 diff） |
| G-L4 | 多机 | 2×16 卡 smoke：G-A4/G-A5 等价 |
| G-L5 | 声明一致性 | lora_inject 注入完成数==声明数；mtp_depths 与 depth_weights 长度一致；draft_files 加载键集合一致（声明漂移三防线） |

## 8. 里程碑与特性分解（绿地线，独立于 V5 探针时序）

| ID | 内容 | 出口 |
|---|---|---|
| L-M1 | mtp/ 子模块骨架（composer/matrix/ckpt/probe）+ 单测 | G-L1 小张量 |
| L-M2 | yaml schema + trainer 泛化 branch-1 | G-L2 |
| L-M3 | provider 接口定稿 + GLM 包适配 | G-L1 全量 |
| L-M4 | 1-rank canary + 16 卡 smoke | G-A5 等价 |
| L-M5 | 多机 smoke（可选） | G-L4 |
| L-M6 | 默认切换（V5 env 路退役） | 用户决策 |

特性树（LF-MTP-F1~F6，见 LF-ARCH-ANALYSIS §7）与上表一一对应；B.4 前置确认
= 训练机窗口、LF fork 策略（fork 特性分支 vs upstream PR）、基线重测。

## 9. 原始诉求追溯矩阵（RFC-001 REQ → 绿地组件 → 责任方）

责任方：**LF**=绿地组件承载；**模型**=几何/精度实现（trust_remote_code）；
**管线**=build/评测外部流程；**OPS**=运维纪律。"间接"=由配方承载而非设施本身。

| RFC-001 REQ | 绿地组件 | 责任方 | 状态 |
|---|---|---|---|
| ARCH-1（K=5） | depth_weights 长度校验（§3.1） | LF 校验/配方声明 | ✅ 间接 |
| ARCH-2 参数全景 | TrainableMatrix 声明式（§3.2） | LF 分组/模型注入 | ✅ |
| ARCH-3 扩参规则 | 矩阵增行 + OPEN 流程 | LF | ✅ |
| DATA-1/2/3 | B 事实（单列 labels 足够）+ F4 可选列 | LF 数据面（近乎零改） | ✅ |
| LOSS-1 recursive 逐位 | 路由缺省 recursive | LF/模型 | ✅ |
| LOSS-2 链式目标态 | contract yaml 键 + 模型契约 C | 模型几何 | ✅（目标态 M4 后默认） |
| LOSS-3 ACC margin | ObjectiveComposer margin 项 | LF 加权/模型计算 | ✅ |
| LOSS-4 深度权重 | composer 归一化/退化 | LF | ✅ |
| LOSS-5 聚合语义 | sum/count 分量契约 | 模型产出/LF 消费 | ✅ |
| LOSS-6 禁 0.0 边 | 契约 C 静态守卫 | 模型 | ✅ |
| FWD-1 chunked | 分量接口允许模型自查 | 模型 | ✅ |
| FWD-2 契约 C 几何 | 模型私有 | 模型 | ✅ |
| FWD-4 HBM 上界 | SMOKE（精度声明配套） | OPS+模型 | ✅ |
| FWD-5 算子符合度 | 模型验证责任（矩阵在 RFC-001） | 模型 | ✅（边界表声明） |
| OPT-1/2 分组+freeze | TrainableMatrix freeze_at | LF | ✅ |
| RT-1 env 表 | yaml 迁移（§6） | LF | ✅（双形态并存） |
| RT-2 生效标记 | 矩阵生效标记+启动标记 | LF/模型 | ✅（G-L5） |
| RT-3 版本锁 | out of scope（OPS E-1） | OPS | ✅ |
| RT-5 训练精度链 | precision 声明（§3.4） | LF | ✅ |
| RT-6 交付精度链 | out of scope（管线） | 管线 | 边界表声明 |
| DIST-1 DDP | 事实 D（per-rank 局部） | LF 原生 | ✅ |
| DIST-2 续训 | draft_files 加载语义（§3.3） | LF/模型 | ✅（本轮补） |
| DIST-3 多机 | G-L4 + 事实 D | LF 原生+配方 | ✅ |
| SRV-1/2 构建开关 | out of scope（管线；draft_files 供数） | 管线 | 边界表声明 |
| CLI-1 命名空间 | out of scope（管线纪律） | 管线/OPS | 边界表声明 |
| EVAL-1/2/4 阶段化/功效/参数固定 | probe.every（内置部分）+ 外部管线 | LF 探针+管线 | ✅ 部分内置 |
| EVAL-3 泛化 | F6 + OPEN-7 | 管线 | OPEN |

**审计结论**：34 条 REQ 中，绿地应承载的 24 条全部有组件对应（本轮补齐
LoRA 注入声明、resume 语义、深度校验、生效标记 4 处缺口）；管线/OPS 责任
6 条以边界表显式排除；模型责任 4 条为既有实现。

## 10. 职责边界表（防范围误解）

| 层 | 负责 | 明确不负责 |
|---|---|---|
| **LF（绿地组件）** | 声明消费、分组/freeze、损失记账、draft 文件存取、探针调度、多机编排 | MTP 数学、算子语义、交付形态决策、命名空间纪律 |
| **模型（trust_remote_code）** | 链式几何、topk 语义、lora 注入执行、精度实现、probe 计算 | 权重与配方（yaml 进）、全局日志规范 |
| **管线（build/run32）** | 交付形态（FLOAT_SET）、合并验证块、双臂评测、命名空间守卫 | 训练语义 |
| **OPS** | 版本锁、卡资源纪律、事故清单 | — |

## 11. 风险与回退

| 风险 | 回退 |
|---|---|
| Composer 聚合与现实现数值漂移 | G-L1 门控；不通过则保留模型内聚合（GD-2 延后） |
| yaml schema 过度设计 | 先覆盖三契约两精度模式，拒绝 speculative 泛化 |
| 上游 PR 兼容 | 长期路线独立于 L-M*（fork 先行） |
| 与 V5 探针争卡 | 绿地线为纯代码+canary 级，不占大窗口 |
