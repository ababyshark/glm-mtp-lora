# V10.1 训练契约与实现 — LoRA(BF16)+full-MTP（V10.1 = V10 + V8 配方成分全量恢复）

## 〇、目的与北极星（2026-11-22 成文，铁律 #8 方案先行）

**目的**：GLM-5.3-Flash 的 serving 投机解码加速——训练一个与目标模型分布
对齐的 MTP draft，使 vLLM 投机解码的接受长度 K̄（原生前缀口径，382 用例
三组，vLLM /metrics 计数器）显著超过 base 的 3.521。

**北极星指标**：K̄（serving 真值）。一切训练有效性结论以它为准；
mcore 探针（CE/agree on 训练分布）只是诊断工具（§11.5 教训）。

**版本**：
- V10：LF + mcore 迁移首代（native CE-only）——serving 裁决负向
  （K̄ 3.52→2.49，§11）
- **V10.1 = V10 + V8 配方成分全量恢复**：KL(main‖draft)×2.33 +
  agree-CE×0.5 + margin×0.3(m=0.5)——成分清查见 git 2d7f951
- 判据：V10.1 的 K̄ ≥ 3.52 ⇔ 配方修复成立 ⇔ LF+mcore 迁移正名
- **V10.2 = V10.1 + draft-CE 降权（1.0→0.25，蒸馏主导）**：
  §11.6 机理分析的核心推论——CE-on-gold 在"主模型 argmax ≠ gold"的
  位置上与接受率指标**反向**（把 draft 概率推离主模型的选择）；
  拆分 bench 实测训过样本反而更差（1.372 < 1.445，−0.073）证实
  anti-alignment 存在。单变量 A/B：V10.1(CE 1.0) vs V10.2(CE 0.25)，
  其余配方不变（KL 2.33 / agree 0.5 / margin 0.3 / 数据/seed 一致）。
  判据：同步数下 V10.2 的 K̄ 曲线高于 V10.1 ⇒ anti-alignment 假说成立。



> 栈：LF（数据管线）+ Megatron-Core 0.18.2（venv_v8，运行时源 /root/mcore0182-src
> @571370c82）+ V10 自定义层。⚠️ 容器里 MindSpeed-MM 在 sys.path 上（部分
> warning 的 import 走到它的 megatron 副本——版本混装隐患，待清理，当前不影响正确性）。
> 基座：GLM-5.3-Flash-BF16 直存（无量化；项目名里的"QLoRA"是历史命名）。

---

## 一、训练对象（用户裁决：主权重+MTP 都微调；真全参不可行 ~225G/卡）

### 1.1 LoRA 精确清单（hook 注入式，fill 键名零破坏，B 零初始化）

| 模块 | 层覆盖 | 适配内容 |
|---|---|---|
| KDA 注意力 | 34 层全部 | `linear_q/k/v/o_proj` |
| DSA 注意力 | 11 层全部 | `q_b_proj`、`kv_b_proj`、`o_proj` |
| Dense MLP | 前 3 层 | `linear_fc1/fc2` |
| 共享专家 | 42 MoE 层 ×1 | `fc1/fc2` |
| **路由专家** | 42 层 ×288 | **排除**（`GLM53_V10_LORA_EXPERTS=1` 可开） |
| indexer/norms/mHC/conv1d | — | 不适配（设计） |

- 259 targets = 518 个 A/B 张量（GRAD AUDIT 运行时证实 ✓）
- 注入 = forward hook（不替换模块）+ B 零初始化（step0 = fill 验证模型逐位一致）
- Column 目标本地加、Row 目标经 mcore 可微 reduce；r=16, alpha=32
- 导出：merge 回基座（W + scale·BA）一次确定性矩阵加法

### 1.2 MTP 层（layers.45）：全参训练

7.4B（592 张量）：enorm/hnorm/eh_proj/内层 DSA+MoE(288)/final_layernorm。
draft head = 主 lm_head（冻结，共享）。

---

## 二、优化器与 loss

- **fp32 master + fp32 Adam 动量**（NPU fused adam 拒绝混合 dtype；
  bf16 动量在 lr=5e-6 有量化噪声）。每步：bf16 梯度→cast fp32→clip→
  master step→回写 bf16。
- **p.grad=None（#16 修复）**：autograd 跨步累加 p.grad——不修则优化器训
  "梯度总和"（gnorm 6.4→46 线性爬升）。修复后 gnorm 平台 5-9 震荡。
- **nan-guard**：非有限 gnorm → 跳过 optimizer.step（一次污染不再毁全部权重）。
- **loss（native_loss v2，V10 自有）**：main CE + draft CE（都配对 ids[t+1]，
  vendor 约定 (h[t], e[t])），可微 gather 路径（内部 vocab CE 黑名单 #10），
  chunked 无 checkpoint（NPU checkpoint-recompute 毒化梯度）。draft 项权重
  GLM53_V10_DRAFT_W=1.0。

---

## 三、约定终审（diag 探针实测，2026-11-20）

| 约定 | 结论 |
|---|---|
| eh_proj 交换 | checkpoint 原序 [e\|h] 即正确；交换破坏 draft（12.72 vs 1.37）。fill 默认不交换 |
| mtp 输入 embedding | e(t)（当前 token），mcore 的 roll(-1) 已 patch 掉（E4 实测 11.73 vs 12.72）|
| h 来源 | pre/post final-norm 无差别（E1/E2 ≈） |
| recompute | 单机只能 full（off 激活账 ~26G+45.8=72G 不可行）；torch_npu 非 reentrant 不释放激活；MTP checkpoint padding_mask 签名 bug 已 patch |
| label 配对 | mcore 主项 = self-pairing（labels 未预移位）→ 原生 loss 值无意义（17.09）；V10 自有 loss 修正 |

---

## 四、可观测性（每 run 自带）

1. **GRAD AUDIT step1**：lora-grads / mtp-grads / frozen-with-grad=0 ——
   运行时契约证明（当前 518+592+0 ✓）
2. **双轨 diag**（每 10 步）：train batch（信号）+ EVAL-FIXED（趋势，
   同 token 含 argmax-agree）
3. **group-grads (pre-clip)**：lora / mtp 分组梯度范数（量测点 = cast 循环，
   clip 前零前——旧 post-zero_grad 读数是死读数已删）
4. **B_moved**：lora_B 权重离零检查（B 恒零 = 注入/接线 bug，硬判据）
5. **相位计时**：dec_fwd / mtp_fwd / CE 每步分解 + fwdbwd/opt 拆分 +
   S（batch maxlen）字段
6. **nan-guard + NONFINITE 计数 + 非 finite 梯度名字清单（step≤2）**

---

## 五、当前结果（50 步探针 + CP50 评测，.40 mtpV8）

| 指标 | 基线 (fresh fill) | CP50 (50 步) | Δ |
|---|---|---|---|
| main CE（固定 token） | 0.8864 | 0.5291 | −0.357 ✓ |
| draft CE（固定 token） | 1.3749 | 0.7050 | −0.670 ✓ |
| **argmax-agree** | **0.7548** | **0.8956** | **+14.1 点** ✓✓ |

draft CE < main CE（"draft 比主头更准"）在 step 40 即达成 ✓。
CP50 master round-trip 首验通过。**续训**：1000 步从 CP50 起
（`--load-cp saves/v10_t4/cp50_rank{rank}.pt`，步数计数器已接 start_step）。

---

## 六、性能账（profiling 结论）

- CPU-bound：主进程 143% CPU vs AI Core 1-5%——launch-bound
- ~2 万小发射/步：MoE 顺序循环 24k（最大头）+ KDA chunk 5.4k +
  mHC sinkhorn/DSA indexer/CE 3k
- 修复阶梯：① MoE grouped 化（`torch_npu.npu_grouped_matmul` 可用 ✓，
  正确姿势 = vllm-ascend 的 `x=[单张量], weight=[3D], split_item=2,
  group_type=0, group_list=累计int64`；autograd 需自封 Function——
  dx 同款 op 可行 ✓，dw 走 pad+bmm；已写 pad+bmm 版
  V10GroupedExperts 待接线）② KDA fla_npu kernel（挂账）③ B 上调
- 注意：npu_grouped_matmul 无注册 backward（x.grad=None）——训练用必须
  自封 Function；weight 列表 ≤128 元素（288 专家需分批调用）

---

## 七、运行配置档案（launch_v10_train_40.sh）

```
B=1 S=1024(yaml cutoff) recompute=full clip=5.0 warmup=20 lr=5e-6
lora_r=16/alpha=32 draft_w=1.0 cp_interval=250（~92GB/CP）
CP 策略：250/500/750/1000；续训 --load-cp ...cp{N}_rank{rank}.pt
数据：cmb_eagle_30k（30000 样本；1000 步仅消耗 3.3%，另备 330k 混合集）
```

### 错误台账（续）

| # | 错误 | 状态 |
|---|---|---|
| 16 | **p.grad 跨步累加**：autograd 语义 p.grad += 新梯度，trainer 只清了 master 的梯度——优化器训"梯度总和"（gnorm 6.4→46 线性爬升 ≈ 每步 +1.6 = 单步范数；loss 平坦震荡；warmup/clip 均无效）。修复 = master cast 后 `p.grad = None`。修复后 gnorm 平台 5-9 震荡 ✓ | ✅ 修（用户诊断） |
| 17 | **launch-bound 性能定性**：~2 万小算子发射/步（MoE 顺序循环 24k + KDA chunk 5.4k + 其他 3k）——143% CPU vs AI Core 1-5%。修复阶梯 = grouped matmul（npu_grouped_matmul 可用，姿势 = vllm-ascend 的 `x=[单张量], weight=[3D], split_item=2, group_type=0`）+ fla_npu KDA | 📋 分组专家模块已写待接线 |
| 18 | **recipe `_vocab_gather` 不可微**：dist.all_gather 作用在带梯度 draft 行——TP16 下 recipe CE 项梯度静默切断；**V8 的 TP16 训练同款**（V8 接受率提升实为 KL/main 项贡献，CE 项无效——历史认知修正）。修复 = mcore 可微 gather | ✅ 修（recipe 启用前） |
| 19 | **diag EVAL-FIXED argmax-agree 广播 bug**：`full.argmax(-1)` 带 batch 维 [S-1,1] 与 tgt [S-1] 比较广播成 [S-1,S-1]（~100 万 mostly-False 对）→ agree 恒报 ~0.013 垃圾值。**数学矛盾检获**：draft CE 1.30 → 命中率 ≥27%，1.29% 不可能。危害 = 假"对齐崩溃"信号 → 错误干预（降 lr/回退配方）。修复 = argmax 后 squeeze。**当前 run（旧代码在跑）的 EVAL-FIXED agree 仍报垃圾值——忽略之，看 train-batch agree（正确）与 EVAL-FIXED CE（正确）；下次重启恢复** | ✅ 修（下次启动生效） |
| 20 | **eval_d15 深度循环缩进脱出**：`accept.append`/`ces.append`/`h_prev=d_k.detach()` 三行缩进在 `for k` 循环外 → 25 格矩阵退化成 5 个跨深度混合值，且 h_prev 从不链式（D2-D5 全用主 hidden）→"baseline 最优、CE 反升至 9.54"的反常假象数据。修复 = 三行缩进进循环（每深度独立记录 + 真链式）。**教训：评测脚本先验形态（基线应劣于训练后、深度衰减应单调）再采信数字**；本次靠"每行只有 1 列 + 与旧跑数字逐位一致"两条线索定位 | ✅ 修（2026-11-21，矩阵已重测） |
| 21 | **grouped 专家三连 bug（接线冒烟检获）**：① `_grouped_fc` 对 [in,out] 转置存储参数做了 `bmm(x, w.transpose)`（期望 [out,in]）——fc2 形状必炸、fc1 仅因单测 H=fc1_out 巧合潜伏（真实 7168/256 必炸）；修复 = 去 transpose。② dispatcher `tokens_per_expert` 在 CPU 与 NPU 张量混算；修复 = `tpe.to(device)` + max_t 取 CPU 值免同步。③ **TP 双重 reduce**：RowParallelLinear 专家路径 `explicit_expert_comm=True` 跳过内部 reduce（layers.py:1341），combine 由 dispatcher `token_combine` 的 reduce-scatter 负责——grouped 模块自己又 reduce 一次 = 16× 过冲 → 饱和 → baseline CE 10.8（均匀随机）。修复 = 删手动 reduce。**终审锚点：grouped baseline eval 与 sequential 逐位对拍**（D1 accept 0.7140 = 0.7140，CE 1.3551 vs 1.3749 = bmm 求和序容差） | ✅ 修（2026-11-21） |
| 22 | **grouped pad+bmm 性能倒退（裁决：默认关）**：chunk 扫描 64/256/512 稳态 dec=43/35/33s vs 顺序 25-28s——pad/gather/scatter 工作区流量（[E,chunk,H] 全物化，偏斜下 E×chunk×H ≫ T×H）> 省下的 ~24k launch。正确性 ✓（chunk 间 loss/gnorm 逐位一致 + baseline 对拍）但慢 15-20%。**裁决：GLM53_V10_GROUPED 旋钮保留默认 OFF，长跑用顺序路径**；真修复 = npu_grouped_matmul（无 backward，需 fwd 用算子 + 自定义 dw backward）→ V11 范围 | 📋 V11 挂账 |
| 24 | **评测 tokenization 失配（检获于分组评测核查）**：d15 矩阵（含历史 eh 审计）用的 ce16p ids 是**无 system 段**的 tokenize 管线（882 tok），而训练消费的是 LF 模板形式（`<|system|>Reasoning Effort: Max`，cutoff 1024）——分布轻度漂移。d15 数字内部自洽但与训练形式不完全一致；分组评测改用训练忠实 tokenization（train_ds[ds_idx]），**两套数字不可横向对比**。已验证 train_ds 与 jsonl 同序（30000 条，ds[0]=jsonl[0]）
| 23 | **EVAL-FIXED off-by-one（#19 修复引入）**：`_fixed_diag` 循环 `range(0, logits.shape[0]-1, 256)` 只约束起点——末块 `logits[768:1024]` 取满 256 行而 `tgt=input_ids[0,1:]` 只有 1023 → `am==tgt` 广播 1024 vs 1023 爆炸。首跑进程（内存=旧代码）不崩；续跑进程（新代码）每 10 步 16 连炸——**非致命包装按设计兜住，训练无损**，train-batch diag（真信号）完好。修复 = 循环以 tgt 长度为准 + end 钳位 + am reshape(-1)。教训：**"下次启动生效"的修复必须带着自己的回归测试**（#19 无测试→#23 引入） | ✅ 修（2026-11-21，下次启动生效） |

## 6b. D1-D5 深度接受率评测协议（CP×深度矩阵，2026-11-21 实测完成）

**评测设计**：5 个 CP 点（baseline/cp250/cp500/cp750/cp1000）× 5 个深度
（D1-D5）= 25 格接受率矩阵。深度 k 的链式推演（DSv3 约定 + no-roll）：

```
h^(0) = decoder 输出（主 hidden，pre-head）
深度 k:  e_k = Emb(roll(ids, -(k-1)))            # e(t+k-1)
         x_k = eh_proj(cat([enorm(e_k), hnorm(h^{k-1})]))
         d_k = inner(x_k) → final_norm            # 深度输出链式传递
         logits_k = lm_head(d_k)                  # 预测 ids[t+k]
         h^{k} = d_k                              # 深度输出作为下一深度 hidden
指标:    接受代理 = P(argmax(logits_k) == ids[t+k])   + 深度 k 的 CE
```

- 数据：训练数据 cmb_eagle_30k 的 text0（882 token，含模板前缀 ✓）——
  用户指定“评测数据 = 训练数据”
- **预期判读**：D1 = 训练过的深度（accept ~0.85-0.91 ✓）；**D2-D5 =
  未训练视界**（当前训练只有 D1）——接受率预期随深度骤降。**这个矩阵
  就是“多深度训练必要性”的量化证据**（V5.4a 深度展开迁移的直接动机数据）
- 脚本：`scripts/v10/eval_d15_40.py`（已运行 ✓，2026-11-21）；运行前置 = 16 卡空闲
  （1000 步长跑完成后）；启动 = `h0-test/eval35/launch_v10_d15_40.sh`（mtpV8 容器
  venv_v8，torchrun×16，master_port=29779）；实测 ~12 分钟
- 产出：`h0-test/eval35/v10_d15_matrix_40.json` + 控制台矩阵

**判读框架**：
| 模式 | 判读 |
|---|---|
| D1 高、D2-D5 骤降 | 单深度训练的预期形态 → **多深度训练（V5.4a 迁移）的必要性证据** |
| D1-D5 全高 | 当前实现已隐式覆盖多深度 → 直接进入 serving 抽测 |
| D1 低 | 训练本身有问题（回查）|

### 6b.1 实测结果（2026-11-21，.40 mtpV8，torchrun×16 TP16，S=882）

**接受率矩阵**（行=CP，列=D1..D5，P(argmax_draft_k == ids[t+k])）：

| CP | D1 | D2 | D3 | D4 | D5 |
|---|---|---|---|---|---|
| baseline | 0.7140 | 0.7841 | 0.6189 | 0.4317 | 0.3158 |
| cp250 | 0.9137 | 0.8761 | 0.6701 | 0.4066 | 0.2748 |
| cp500 | 0.9183 | 0.8852 | 0.6542 | 0.3952 | 0.2406 |
| cp750 | 0.9217 | 0.8773 | 0.6405 | 0.4032 | 0.2725 |
| cp1000 | **0.9228** | 0.8773 | 0.6223 | 0.4282 | 0.2965 |

**CE 矩阵**（同布局）：

| CP | D1 | D2 | D3 | D4 | D5 |
|---|---|---|---|---|---|
| baseline | 1.3749 | 0.9823 | 1.9718 | 3.2781 | 4.4345 |
| cp250 | 0.3863 | 0.5548 | 1.6190 | 3.3386 | 4.8226 |
| cp500 | 0.3570 | 0.5191 | 1.5890 | 3.4261 | 4.6214 |
| cp750 | 0.3568 | 0.5483 | 1.7159 | 3.4040 | 4.3948 |
| cp1000 | 0.3516 | 0.5613 | 1.7975 | 3.3764 | 4.3960 |

**判读（命中判读框架第 1 行）**：

1. **D1（训练视界）单调改善并饱和**：0.7140 → 0.9137（cp250，+21 点）→
   0.9228（cp1000）；CE 1.37 → 0.35（-74%）。1000 步 D1-only 训练有效，
   且 ~500 步后饱和（步数预算/LR 决策参考）。
2. **D2 链式溢出收益**：0.7841 → 0.877-0.885（+9~10 点）——D1 训练经
   链式隐层部分惠及 D2（免费收益，但 < D1 增益）。
3. **D3-D5（未训练视界）原地踏步甚至略降**：0.619/0.432/0.316 →
   0.622/0.428/0.297；D3-D5 CE 对 CP 不敏感（1.6-1.8 / 3.3-3.4 / 4.4-4.8
   波动）。**D1-only 配方在深度 2 饱和，D3-D5 完全吃不到收益 → 多深度
   训练必要性成立**——V11 深度展开配方迁移的直接动机数据。
4. 旧"混合单值"数据（baseline 0.2064 最优、CE 反升 9.54）为评测 bug
   产物，作废（台账 #20）。

**正确性锚点**：
- baseline = 原生 vendor 权重（不加载 CP，LoRA B 零初始化 = 恒等）✓；
  CP 依次 baseline→cp250→cp500→cp750→cp1000 顺序加载 ✓
- baseline D1 CE = 1.3749 ≈ eh 审计已知正确 draft CE（1.37）——评测链路
  与历史终审互证 ✓
- 训练后 D1>D2 恢复正常排序（baseline 的 D2>D1 反常见挂账 #6）

## 6d. 分组评测设计：32 类 × 3 组 × CP 阶梯（2026-11-21 用户裁决）

**目的 = 分析训练有效性**。分类逻辑落码于 `scripts/v10/v10_eval_taxonomy.py`
（2 语言 × 4 任务 × 2 轮次 × 2 长度 = 32 类；操作定义见模块 docstring；
math+code 归 math 保 4 任务）。采样机制事实：batcher = randperm(30000,
seed=42+epoch) 无放回、batch=1 → 排列位置 p = 该样本的消费步号（0 基），
1000 步窗口内每个被消费样本**恰好训练 1 次**（已实测确认）。

**三组用例**（`data/eval_groups/*.jsonl`，seed=20261121 确定性，组间零重叠，
每类目标 10 例压单例方差；严格窗口裁决 + 矩阵逐格标 n）：

| 组 | 采样窗 | CP250 | CP500 | CP750 | CP1000 | 回答的问题 |
|---|---|---|---|---|---|---|
| G1 unseen (197/23类) | 位置 ≥2000* | 未训 | 未训 | 未训 | 未训 | 没训过的会不会有提升（域适应）|
| G2 cp250 (98/16类) | 位置 <250 | 已训1 | 已训1 | 已训1 | 已训1 | 训过的随继续训练巩固还是遗忘 |
| G3 late (87/14类) | [750,1000) | 未训 | 未训 | 未训 | 已训1 | recency（G3@1000 vs G2@1000）|

*G1 取 ≥2000 而非用户字面的 ≥1000：续训 run 消费 [1000,2000)，取 ≥2000
使 G1 到 CP2000 都保持"未见"——是用户"完全不在前 1000 步"的加强版。

**覆盖度现实**：9/32 类全池结构性为零（en|yuwen×4、zh|math|multi×2、
zh|yuwen|multi×2、zh|code|multi|mid）→ 可评 23 类；G2/G3 窗口仅 250 条
摊 23 类，稀缺类 n=1-6（n<3 只看方向不读绝对值）。

**评测协议**（`scripts/v10/eval_groups_40.py`，内核测试
test_v10_group_eval.py 锁定回显对齐/掩码/链式）：
- tokenization = **训练忠实**（load_lf_data → train_ds[ds_idx]，cutoff 1024，
  含 system 模板）——见台账 #24
- 批量化 B=16（按长度排序降 pad 浪费；right-pad 因 causal 对真实位置安全；
  逐 case 指标按 true_len 掩码），指标 = D1-D5 链 accept/CE（同 d15 约定）
- CP 阶梯 = base + cp250/500/750/1000（1250-2000 落盘后自动纳入）
- 聚合 = 逐 (组, 类, 深度) 均值带 n + 逐组池化；产出
  h0-test/eval35/v10_group_eval_40.json

**读法骨架**：G1 跨 CP 平 = 无域外溢出 / 升 = 域适应有效；G2@250 > base =
单次触达有效，G2 跨 CP 降 = 遗忘信号；G3@1000 > G2@1000 = recency；
G3 前三 CP ≈ G1 = 窗口语义自洽校验。

### 6d.1 base 分片结果 + K̄ 接受长度估算（2026-11-22）

**base（原生未训练 vendor）三组矩阵**（382 case × 5 深度，训练忠实
tokenization，分片 base.json）：

| 组（n）| D1 | D2 | D3 | D4 | D5 |
|---|---|---|---|---|---|
| G1 未见（197）| 0.5759 | 0.5988 | 0.4623 | 0.3391 | 0.2423 |
| G2 cp250 窗（98）| 0.6021 | 0.6314 | 0.5037 | 0.3796 | 0.2767 |
| G3 晚窗（87）| 0.5942 | 0.6150 | 0.4836 | 0.3623 | 0.2601 |
| 全池均值 | 0.5868 | 0.6109 | 0.4778 | 0.3548 | 0.2552 |

- D2→D5 干净衰减；D2>D1 的 baseline 反常**三组全部再现**（与 d15
  baseline 形态一致）——坐实挂账 #6：vendor MTP 未训练态 D2>D1 是模型
  特性非评测 bug
- 三组 base 差异 = 采样构成差，训练效应读 CP−base 组内净差
- 样本长度：token 均值 983 / 中位 1024 / 87% 打满 cutoff——评测窗口 =
  训练窗口（≤1024），3k+ 长上下文是独立协议（外推语义，未立项）

**K̄ 接受长度**（K=5 理想全收；标准估算 K̄ = 1 + Σ_{k=1..5} Π_{j≤k} p_j，
1 = 主模型 bonus token）：

| 状态 | p(D1..D5) | K̄ |
|---|---|---|
| base（全池）| .587/.611/.478/.355/.255 | **≈2.19**（纯 draft ≈1.19）|
| 训练后 cp1000（d15 单样本参考）| .923/.877/.622/.428/.297 | **≈3.52** |

- 训练把 K̄ 从 ~2.2 抬到 ~3.5，距理想 5 的缺口主要在 D3+ 未训练视界
  ——多深度训练（V11）的直接收益空间
- **口径警告**：此为边际率独立乘积估计；真实 serving 验证是前缀条件式
  （k 仅在 k−1 全对时检查），分片只存逐深度聚合未存逐位置联合 → 精确
  K̄ 需内核加联合计数（TODO：batch_metrics 累加 per-position 联合接受，
  一次改动，重启后产出真值）

## 7. 训练数据来源、构成与评测关系（2026-11-20 全量实测）

### 7.1 来源与管线

```
cmb_eagle_30k.jsonl（30000 条 sharegpt 对话）
  = 符号链接 → /mnt/share_space/l00840580/mtp/lf_glm53_qlora_mtp/data/lf_data/
  ✓ .40 容器内可解析（DATA_OK 已验）
  ↓ LF 数据管线（fork）
  load_tokenizer（Flash-BF16，md5 恒等 ✓）+ glm5_next_text 模板
  + get_dataset（sharegpt 格式，dataset_info.json 注册 ✓）
  ↓ 截断 cutoff=1024 + 分 batch（pad 到 batch 最长）
McoreBatcher → [B, S] 张量
```

- 格式：sharegpt 对话（human/gpt 轮次）；95% 为 2 轮对话
- 注册：data/lf_data/dataset_info.json ✓

### 7.2 构成（30000 条全量实测，非抽样）

| 维度 | 实测 |
|---|---|
| 语言 | **含中文 34% / 纯英文 66%** |
| 对话轮数 | 2 轮 28432（95%）；多轮 4-38 轮共 1568（5%）|
| 字符长度 | min=2 / 中位=166 / 均值=367 / max=36355 |
| 内容类型 | **数学应用题为主**（GSM 风格中英文混合）+ 通用问答；医疗关键词仅 2%；无 markdown 代码块 |
| 内容例 | “Danielle的公寓有6个房间…Grant的公寓有多少房间？”（数学）/ “I am using Fabric to start a celerycam process…”（devops）|

**认知修正**：数据集名 cmb_eagle（CMB=医疗 benchmark 的直觉）**与实测不符**——
内容以**数学应用题 + 通用问答**为主（K-12 风格数学占显著比例），医疗占比低。
这与判定训练场景（通用文本的 draft 对齐）**匹配良好** ✓。

### 7.3 与判定训练的匹配度

| 维度 | 匹配 |
|---|---|
| 语言分布 | 中英混合 ✓（serving 场景两类都有）|
| 内容分布 | 数学/通用 QA ✓（通用能力训练 ✓）；医疗/垂直领域占比低 ⚠️（若 serving 以垂直域为主，考虑 330k 混合集补充）|
| 长度分布 | 中位 166 字符 ≈ 80-100 token——**多数样本远短于 cutoff 1024**（pad 浪费的来源 ✓ §十分析）；长样本（>1024 token）占比小 |
| 难度 | 数学题的推理链 = draft 对齐的高价值训练信号 ✓ |

### 7.4 评测数据与训练数据的关系（ containment 关系）

**评测数据 ⊂ 训练数据（字面包含）**：

| 评测资产 | 来源 | 与训练集的关系 |
|---|---|---|
| 固定 eval batch（diag 的 EVAL-FIXED）| **train_ds[0]** = 训练集第 0 条 | **在训练集内**（in-sample）|
| D1-D5 评测的 text0（882 token）| cmb_eagle_30k 样本（16 文本提取的前缀）| **在训练集内** |
| CE 门 16 texts | cmb_eagle_30k 前 16 条 human turns | **在训练集内** |
| serving 抽测 prompt 集 | cmb 200 + 通用 200（计划）| cmb 部分 in-domain（同分布）；通用部分 = 泛化探针 |

**三层含义**：

1. **EVAL-FIXED 量的是拟合度，不是泛化度**——同 token 的 CE 下降 =
   “模型在训练分布上拟合更好” ✓（趋势有效），但**不能证明泛化**。
   泛化评测需要 held-out split（当前未构建——**V11 待办：从 30k 尾部
   隔离 2000 条做 held-out**，训练时排除）
2. **serving 抽测的对照设计**：fresh fill 基线 = 同一批 prompt 的未训练
   draft——Δacceptance 排除“数据本身简单”的混淆 ✓（对照在同分布内 ✓）
3. **若需要严格泛化结论**：构建 held-out split（train/eval 分离）——
   当前“评测=训练子集”的设计对**管线验证**足够，对**泛化声明**不足

### 7.5 配比现状与决策点

| 项 | 现状 | 决策点 |
|---|---|---|
| 数据源 | 单源 100%（cmb_eagle_30k）| 是否混入 330k（垂直域补充）？|
| 掩码 | 全 LM（含 pad/用户轮）| assistant-only / IGNORE_INDEX / 维持——**待 A/B**（§3.2 ①标签行）|
| 截断 | 右截断 1024 | 长样本（36k 字符 ≈ 9k token）截断丢失率待统计 |
| on-policy | 未启用 | T2 线（.38 T0 修复后）——serving 分布采样的训练价值 |

## 八、挂账

1. HF 三方对拍（#13 strict-parity 离散分歧裁决）
2. KDA fla_npu kernel（吞吐+AI Core 根治）
3. 专家 LoRA 旋钮验证（GLM53_V10_LORA_EXPERTS=1）
4. serving 侧接受率抽测（agree ~0.90 已近门槛；与 eh 双份导出同批）
5. MindSpeed-MM 混装 import 清理
6. baseline 的 D2>D1 反常（0.7841>0.7140，CE 0.98<1.37）——vendor MTP
   链式隐层语义疑点（final_norm 后链式 vs 前-norm 输入的交互），训练后
   消失（cp1000 D2<D1 正常）；V11 多深度配方迁移时复查
7. held-out 评测 split 未构建（当前评测 ids ⊂ 训练数据，见 §7.4）
8. grouped 专家真修复（#22）：npu_grouped_matmul fwd-only + 自定义 dw
   backward（vllm-ascend 姿势 `x=[x], weight=[w3d], split_item=2,
   group_type=0, group_list=cum`）——归 V11
9. **双向 sync 纪律**（#21 检获）：tar 单向覆盖曾把 .40 的修复版
   v10_grouped_experts 打回未修复版——同步前先 md5 双向对比，.40 本地
   改动必须先回流 git 再同步

## 6c. grouped 专家改造（2026-11-21 接线完成，默认 OFF）

**范围**：`GLM53_V10_GROUPED=1` 旋钮一处生效三处——主栈 spec
（v10_model.build_layer_specs）、MTP 块 spec（v10_mtp.build_mtp_block_spec）、
fill 路径（v10_fill._mlp_rule 记录 + flush_grouped 一次成型 w1/w2）。
注意 `get_moe_module_spec` 返回 `partial(MoELayer, submodules=...)`，
交换点 = `.keywords["submodules"].experts`。

**验证链**（全部绿）：
- 单测 7 项：pad+bmm 数学 vs 逐专家循环参考（含空专家/梯度流/分块边界
  chunk=2）；flush_grouped w1/w2 与 SequentialMLP fill 内容逐位对拍
  （fc1=cat(shard(gate),shard(up)) 转置、fc2=shard(down,row) 转置）；
  全套 90 passed
- NPU 端到端：grouped baseline eval vs sequential 基线矩阵
  accept D1 0.7140 = 0.7140（逐位），CE 1.3551 vs 1.3749（bmm 求和序
  容差）；fill 台账 1243/1243 零 unwritten/unconsumed
- 分块确定性：chunk=64/256/512 的 loss/gnorm 逐位一致

**性能终审（#22）**：稳态 dec 33-43s vs 顺序 25-28s——慢 15-20%，
默认 OFF。三个修复（#21）+ 分块 OOM 防护（偏斜 max_t~2848 → 全物化
11.75GB，chunk 后工作区 [E,chunk,H] ~0.26-2.1GB）均为可复用资产。


---

## 九、与 V5.4 的参数对比 + 配方迁移计划（2026-11-20 分析）

### 9.1 超参对比（V5.4 = launch_v54_40.sh + glm53_joint_v54_60.yaml 实测）

| 项 | V5.4 (W4A8 78L) | V10 (BF16 45L) | 一致？ |
|---|---|---|---|
| lr | 5e-6 | 5e-6 | ✅ |
| warmup | 20 | 20 | ✅ |
| batch | 1 (+accum 1) | 1 | ✅ |
| LoRA r/alpha | 16 / 32 | 16 / 32 | ✅ |
| 数据集 | cmb_eagle_30k | cmb_eagle_30k | ✅ |
| clip | 1.0 | 5.0 | ⚠️ V10 为治 clip-throttling 放宽 |
| cutoff | 2048 | 1024 | ⚠️ 显存取舍 |
| LoRA dropout | 0.05 | 0.0 | ⚠️ 小 |
| optimizer | AdamW (HF 默认 β₂=0.999, wd=0) | AdamW (β₂=0.95, wd=0) | ⚠️ 小 |
| staging | BASE_LORA_FREEZE_AT=20 | 无 | ⚠️ |
| **loss 配方** | **serving_chain：0.3CE+0.7KL + agree 0.5 + margin 0.3/0.5 + 深度权重 3,4,3,1,1** | **native：main CE + draft CE（各 1.0）** | ❌ **最大差异** |

结构性不可比：V5 = W4A8 78 层（MTP layers.78）；V10 = BF16 45 层（layers.45）。

### 9.2 关键发现

**loss 配方是最大差异**。V5.4a 的 KL（对冻结主头分布的蒸馏，0.7 权重）+
agree-CE（argmax 追赶 0.5）+ margin（0.3/0.5）+ 深度权重 = V5 验证过的
接受率驱动器；V10 当前的 native 双项 CE 是**简化目标**——纯 CE 已拿到
argmax-agree +14 点/50 步，但没有蒸馏项。

### 9.3 配方迁移计划（单变量纪律）

1. 当前 1000 步 native 长跑跑完 = **纯 CE 基线**（管线验证 + 基线曲线）
2. 下一个 run 切 `GLM53_V10_RECIPE=1`（v10_recipe.serving_chain 已实现，
   trainer 原生支持）——**一次只动 loss 配方**：
   - 迁移项：KL(0.7)/agree(0.5)/margin(0.3/0.5)/深度权重
   - 同时对齐 V5 的 clip=1.0 / dropout=0.05 / cutoff=2048
     （⚠️ 这违反单变量纪律——若严格单变量，第一轮只切配方，对齐项第二轮）
3. A/B 判据：EVAL-FIXED draft CE 斜率 + argmax-agree 斜率（同 eval batch）
4. 注意：V5.4a 配方跨模型迁移（W4A8 78L → BF16 45L）是**假设待验证**——
   深度权重/margin 的调参经验不一定平移，斜率若劣于纯 CE 则回退拆项归因


---

## 十、步时与 loss 的跳动分析（2026-11-20，step 567-601 实测）

### 10.1 步时跳动 = 数据长度，严格线性

| S（batch maxlen） | 步时 | dec 前向 |
|---|---|---|
| 353 | 40.4s | 12.2s |
| 488 | 47.0s | 15.1s |
| 653 | 52.6s | 18.2s |
| 888 | 64.5s | 23.2s |
| 1024 | 72-75s | 27-28s |

batcher 按 batch 内最长序列 pad；样本长度 353-1024 不等 → 步时线性跟随
（相位计时同步缩放 ✓，fwdbwd/opt 拆分同样跟随：72.2s vs 39.7s）。
不是 bug。

### 10.2 loss 跳动 = 样本难度方差

B=1 时每步 = 一个样本；cmb 混合医疗/代码/推理文本，单样本 CE 天然
0.57-4.06 波动。同 S 的样本 loss 也不同（S=1024：1.14 vs 2.11）→
**内容难度方差，不只是长度**。不是 bug。

### 10.3 趋势判读（跳动的噪声明者）

- **train-batch argmax-agree 持续上行**：0.8428@60 → 0.9141@600 ✓
- EVAL-FIXED main/draft CE 平台 ~0.995（当前 lr 下的收敛水平）
- 单步 loss 跳动让趋势难读——**趋势只看两条线**：固定 eval batch 的
  main/draft CE + train-batch 的 argmax-agree

### 10.4 可选平滑（未实施）

1. **长度分组 batching**（按样本长度排序分桶）：步时方差消失 +
   pad 浪费（现 ~25%）回收，吞吐 +20% 左右——标准 LLM 实践
2. loss 10 步 EMA 随步打印

## 十一、serving 评测裁决：训练负向迁移 + 根因 + 修复方向（2026-11-22）

### 11.1 数据（vLLM 原生前缀接受长度 K̄，382 用例三组，G1 口径）

| 臂 | K̄ | D1 条件接受 | 说明 |
|---|---|---|---|
| base（纯 vendor）| **3.521** | 0.843 | 唯一零工程介入臂，最可信 |
| **cp500（完整导出）** | **2.487** | **0.702** | **比 base 差 1.03 token/步（−29%）** |
| bisect（LoRA-only，MTP 回滚 vendor）| **3.20+**（爬升中）| — | **恢复近 base** |

### 11.2 归因（bisect 二分实验，干净）

- **LoRA 合并（decoder）无罪**：MTP 回滚后接受率恢复 → 导出流水线双向验证清白
- **毒源 = MTP 子树的训练产物**：训练后的 draft 在 serving 上比原装差

### 11.3 根因：V10 配方丢了 KL 对齐项

- V10 native loss = draft CE 对语料（mcore process_mtp_loss，无 KL）
- V5.4 配方 = CE + margin + **KL(main‖draft)** + agree（V10_TRAIN §9.1）
- 后果：draft 被 CE 拉向**语料分布**，脱离**目标模型分布**——而投机解码接受率
  只取决于 draft 与目标模型的吻合度。vendor MTP 是 GLM 预训练对齐过的；
  500 步 CE 微调把它推离 → 接受率退化。**退化随步数剂量递增**（待量化）。
- mcore 探针为何说"大幅改善"：它测的是**训练目标自身**（CE on 训练格式
  全序列、训练分布）——近乎同义反复。serving 才是产品指标。两口径的矛盾
  = 分布偏移的直接证据。

### 11.4 修复方向（按代价排序）

1. **配方修复（正解）**：native loss 加 KL(main‖draft) 项（v10_recipe.py 已有
   可微 KL 实现——ledger #18 修过它的 gather），权重取 V5.4 配方值。
   重训 250 步 → 导出 → serving 复测，K̄ 应 ≥ base。
2. 退化剂量曲线（可选证据）：cp250/750/1000 导出跑 serving，量化 K̄ 随步数
   的衰减——为 KL 修复提供 before/after 对照。
3. serving 侧兜底（若短期不重训）：serving 用 vendor MTP + 仅 LoRA 合并
   （bisect 目录 = 已验证恢复 3.20+）——**主模型质量提升保留，draft 不劣化**。

### 11.5 教训（并入铁律）

- 评测必须先测**产品指标**（serving 口径），训练目标的自身指标只能做诊断
- "训练目标优化得好" ≈ 同义反复，不构成有效性证据
- serving-first（铁律 #7）的真正含义：**第一臂就跑 serving 对照**，而不是
  先跑 10 小时内部探针再补 serving

### 11.6 机理定案：exposure bias + 分布收窄（2026-11-22，与用户对齐后修正）

**用户的原理（教科书正确）**：主干预训练 = 语料 N+1；MTP = 语料 N+2
（输入 [h(N), e(N+1)]）。vendor 的 MTP 同样是语料 N+2 训练，且做到
K̄ 3.52——**"语料训练"本身不是罪**。

**真正的差异 = 训练分布覆盖面**：

| | vendor MTP 训练 | 我们的 500-1000 步微调 |
|---|---|---|
| 语料 | 数万亿 token 全域 | 30k 数学/QA（R1 think 链）|
| 效果 | N+2 规律全域覆盖 | 窄域梯度覆写通用先验（灾难性遗忘）|
| serving | 自生成上下文 ⊆ 覆盖内 | 自生成通用文本 ∌ 窄域外 → 崩 |

**两层偏移（同一根源：微调收窄分布，serving 发生在宽分布）**：

1. 输入偏移：训练 e(N+1)=gold（teacher forcing）；serving e(g_k)=主模型
   自选 token（exposure bias）。V8 文档已标注此失配，self-feed 修法被证伪。
2. 目标偏移：训练目标=语料 N+2；serving 验收=draft 是否等于主模型会选的
   N+2。窄域语料上两者近似重合；分布外分叉。

**token 流实例**：训练让 draft 学会"解题文本'答案是'后接数字/\\frac"；
serving 收到历史题 → 主模型生成历史叙述 → 窄域 draft 猜数学味续写 →
主模型拒绝 → 接受崩塌。base 不死 = 万亿覆盖 ⊇ 自生成合理文本。

**可证伪预测**：serving 上数学/代码 prompt 不应退化（分布内），通用类
显著退化 → 类切片 bench 可直接证实/证伪（待跑）。

**修复排序（修正后）**：
1. on-policy 自生成数据（训练输入=serving 输入，直击两层偏移；V8 Phase 5）
2. 通用语料回放混训（保覆盖面）
3. KL 锚定（已做，只在训练分布内有效——100 步仍崩的实测结论）
4. 兜底：serving 按域路由 draft（数学流量走训练版，通用走 vendor）

**教训入铁律**：成分清单必须含**数据分布覆盖**项——"loss 项齐了"不等于
"配方齐了"，数据分布偏移与 loss 项缺失同样致命。
