# V11 计划:多深度链式蒸馏(2026-11-25 立项)

> **版本声明**:V11 编号重授。旧 V11(W8A8 QLoRA MTP 线,`launch_v11_train_40.sh` /
> `glm53_v11_train_40.yaml`,源仓 docs/V11_W8A8_QLORA_PLAN.md)**作废**——从未完成训练,
> 其遗留资产(命名契约 053926d、venv 验证)仍有效;其"单变量 A/B"框架被本计划取代。

## 0. 触发事件:B1 serving 塌陷(2026-11-24/25)

B1(on-policy 数据,500 步,V10.3 配方)训练探针全面向好
(EVAL-FIXED vs-main agree 0.55→0.835,dacc-D1 0.94,KL→0.04),
serving 却结构性塌陷:

| 首窗 | K̄ | D1/D2/D3/D4/D5 |
|---|---|---|
| base(工厂) | 3.91 | 0.90/0.71/0.57/0.42/0.32 |
| Arm A cp100(离策略) | 3.41 | 0.85/0.64/0.43/0.30/0.19 |
| **B1 cp500 / cp300** | **1.56-1.65** | **0.50/0.12/0.01/0/0** |

cp300 ≈ cp500(**与步数无关**,排除断纯漂移剂量)。main 文本连贯(主模型完好)。
流形实验(训练原文 vs 新 prompt 经 serving 栈)A≈B(0.64/0.76)——**流形绑定不成立**。

## 1. 根因(证据链闭环)

**根因一:深度 1-only 训练目标摧毁自链。** 训练时 draft 永远条件于 main 的
hidden(`dec.detach()`,native_loss),从不条件于自己的输出;serving 深度 ≥2 必须
吃 draft 自己的 hidden。铁证 = 训练日志里的**链式 dacc 全程崩塌**(此前被误读为
样本噪声):

| B1 step | D1 | D2 | D3 | D4 | D5 |
|---|---|---|---|---|---|
| 1(≈factory) | 0.785 | 0.845 | 0.766 | 0.575 | 0.438 |
| 98 | 0.916 | 0.617 | — | — | — |
| 467 | 0.916 | **0.324** | **0.163** | **0.110** | **0.083** |

深度 1 越练越好、深度 2-5 越练越死——**目标与 serving 需求结构性分离**。
Arm A(100 步)损伤轻微(serving D2 0.53),B1(500 步 × 自相似数据)致命。

**根因二:stale-policy。** rollout 轨迹由 base-main 生成(训练前固定),而 KL/agree
的锚是**漂移中的 main**——训练对 = base 轨迹 × 漂移目标,serving 对 = 漂移轨迹 ×
漂移目标,组合从未在训练中出现。D1 的 serving 跌幅(0.85→0.49-0.64)由此贡献。

**探针教训(第三次同义反复)**:dacc-D1 / EVAL-FIXED agree / KL 全部只测深度 1
且只在训练分布上测——**链式 dacc(深度 ≥2 条件于 draft 自身输出)才是 serving
忠实探针**,它全程如实报告了崩塌,被当成噪声忽略了。

## 2. V11 设计

### 2.1 目标函数(核心变更)

深度 1 块:**V10.3 配方逐字不变**(CE + KL(main‖draft) 0.8 + agree-CE 0.2,
条件于 `dec.detach()`)——D1 侧收益已被 Arm A/B1 双向验证。

深度 2-5 块(**新增,修复自链**):

```
for k in 2..5:
    ids_k   = roll(ids, -(k-1))                 # no-roll patch: 位置 t 吃 ids[t+k-1]
    h_prev  = d_{k-1}.hidden.detach().requires_grad_(True)   # R8 锚点同款
    d_k     = MTP(ids_k, position_ids, h_prev)  # 独立前向(逐深度 detach)
    loss_k  = CE(d_k_logits[:S-k], ids[:, k:])  # vendor 深度 k 约定
    loss   += w_k * loss_k
```

- **逐深度 detach**(v1):教"处理自己的输出分布"(流形教学),不做 BPTT——
  BPTT(深度 5 的梯度穿过 1-4)列为升级项,仅在 detach 版平台时启用
- 深度权重 `GLM53_V11_DEPTH_W`,默认 `1,0.75,0.5,0.35`(对应深度 2-5 共 4 项;
  D2 最重——serving 崩得最深;D1 块的监督是 KL/agree,不在本 env 内)
- position parity:训练链用 batch position_ids(arange),与 native 路径一致;
  **pre-flight 必须与 vLLM draft loop 的位置/roll 约定逐项对拍**

### 2.2 策略冻结(第二变更,升格为核心)

**`--lr 0` 冻结主模型**(原 T7b 消融升格):轨迹生成者 = 对齐目标 = 永不漂移
→ **真·on-policy**(stale-policy 结构性消失)。附带红利:现有 rollout 数据
(base-main 生成)对冻结 main **天然永续 on-policy,无需重跑 rollout**。
LoRA 侧零梯度后,`main_ce` 保留计算但无更新路径(GRAD AUDIT 的
`lora-grads=0` 属预期);export 时 LoRA 增量为零,serve main = 纯 base。

### 2.3 不变项

- 冻结面:router + 全部 RMSNorm(V10.3)+ indexer(V10.3_INDEXER_FREEZE_VERDICT)
- 数据:`flash_onpolicy_all`(29959 条,质检 PASS)
- 步数 500,CP 每 100(V10.1 式 CP×深度矩阵);base 2.60 = 唯一判决线

### 2.4 实施清单

| # | 项 | 内容 |
|---|---|---|
| E1 | native_loss 链式化 | 上述循环 + `GLM53_V11_DEPTH_W` env + `GLM53_V11_CHAIN=1` 开关(0 = V10.3 replay 字节等价) |
| E2 | R8 审计增补 | 显式列出 `requires_grad=True 且 grad is None` 的参数(indexer 盲点);链式后每个深度桶独立列 |
| E3 | 健康门(**会拦住 B1 事故的那道闸**) | 每 diag 步:链式 dacc D2-D5 不得低于 factory 参照(0.845/0.766/0.575/0.438)的 85%,跌破即告警并阻断 CP 判决 |
| E4 | CPU 单测 | 链式损失:各深度梯度入 draft、零梯度入冻结面/LoRA-main、roll/tgt 形状、CHAIN=0 replay 等价 |
| E5 | pre-flight | 训练链 vs vLLM draft loop 的位置/roll/停止符约定对拍(防止"训练教的是 serving 不用的东西") |
| E6 | 内存/时长 | 5 次链式前向 + 各深度 detach,估 step 78→~90s;500 步 ≈ 12.5h |

### 2.5 判决与消融阶梯

- **判决**:cp300+cp500 两点 382 bench,**K̄ > 2.60 = V11 成立**;
  serving D2-D5 必须 ≥ factory 链水平(0.65/0.42/0.30/0.20 量级)
- 消融阶梯(主判决后按需):
  - V11-B:同配方换 cmb 离策略数据(数据归因)
  - V11-C:detach → BPTT(链式优化强度)
  - 多深度 × 离策略复刻 Arm A 对照( objective 归因)
- 诚实声明:V11 相对 B1 同时改**目标(多深度)+ 策略(冻 main)**两个变量——
  B1 的崩塌有两个已证机制,只修一个留另一个;归因由消融阶梯补齐

## 3. 风险

| 风险 | 缓解 |
|---|---|
| 深度 2-5 gold-CE 重蹈 V10.1 覆辙 | on-policy 数据的 gold = 自己的轨迹(冻结 main),与 V10.1 的窄域外部语料不同;D1 侧 KL/agree 锚不变 |
| 链式前向触发 mcore recompute 梯度陷阱(R8 类) | h_prev 逐深度 `requires_grad_(True)` 锚点;step1 GRAD AUDIT 硬校验 |
| 5 链前向拖慢步时 | MTP 单层前向 ~0.6s,估 +12s/步;超预期则砍深度到 3(D4/D5 serving 占比最小) |
| eval_step0 --load-cp 实验未完成(cp300 在 mcore 图的配对值) | **.40 解禁后第一件事**:结果决定是否需追加 serve/train 前向 parity 调查项 |

## 4. 执行顺序

1. E1-E4 实现 + CPU 测试全绿(工作站,不占卡)
2. E5 pre-flight(需 16 卡,预约时段)
3. V11 主跑 500 步(E3 健康门全程盯)
4. cp300+cp500 导出 + 382 bench → 对 2.60 终审

---

## 5. A3 增补:条件存活率诊断与权重陡化(2026-11-25)

base 基准表(b382 三组均值)的条件存活率 P(Dk|D{k-1}) = Dk/D{k-1}:

| 深度 | 无条件率 | 条件存活率 | 衰减 |
|---|---|---|---|
| D1 | 0.8507 | — | main→draft 接口(绝对强项) |
| D2 | 0.6551 | **77.01%** | **-8.06%(最大断崖:main 空间→MTP 自循环空间)** |
| D3 | 0.4831 | 73.74% | -3.27% |
| D4 | 0.3523 | 72.92% | -0.82%(稳态) |
| D5 | 0.2573 | 73.03% | 走平(稳态) |

**定位**:病灶不在 D4/D5(D3 起稳态 ~73% 走平),而在 D1→D2 接口跃迁。
B1 崩塌态同接口存活率 = 24%(0.12/0.50)——接口被深度 1-only 训练打碎的直接度量。

**A3 调整**:
1. layer0 = 纯接口翻译器(100% 冻结,零改动)——0.8507 基本盘 + 泛化下限零损伤
2. layer1 = D2/D3 桥接专精:DEPTH_W 陡化 `1,0.65,0.35,0.15`(D2+D3 占 76.7%),
   D4/D5 为稳态维持正则
3. D2 接口 KL(`GLM53_V11_D2_KL`):layer1 深度 2 输出对齐 main 的 t+2 软分布
   (teacher-forced 下 main_logits[t+1] 现成)——P(D2|D1) 77%→82% 即 K̄ 2.60→2.72+
4. 监控:步行日志新增 `surv=`(条件存活率四项);阈值与告警同 E3 门
5. D5 serving 截断账:K̄ 2.60、draft 成本 c≈0.08 时,保 D5 吞吐 1.86 > 截断 1.77——不截
