# V10 On-Policy 蒸馏方案（GLM-5.3-Flash 非量化）

> 状态：方案（2026-11-19）。V10 的**子方案**，与 `docs/V10_DESIGN.md` / `docs/V10_IRON_RULES.md` 对齐；落点在 V10 的 Phase 里程碑上。
> 资源：**.38 16 卡当前全空** → rollout 在 .38 启动。不涉及 .40。

---

## 一、定位与依赖

- 基座：**GLM-5.3-Flash 非量化（BF16）**——45 层 hybrid（34 linear_attention + 11 DSA）+ MoE 288 experts + MTP（layers.45），hidden 4096，vocab 154880。
- 栈：与 V10 主方案同栈（LF 数据 + mcore 0.18.2 + `v10_trainer.py` 胶水，V8 模式）。
- 依赖关系：
  - **rollout（数据生成）不依赖 V10 build**——用 HF `Glm5NextForConditionalGeneration` 直接加载 BF16 权重即可，现在就能在 .38 跑；
  - **训练（阶段 B）依赖 V10 里程碑**：B1 需 Phase 2（fill + 训练循环）就绪；B2 需 Phase 4（V5.4a 配方迁移）就绪。

## 二、Step ZERO（铁律 §4 语义 CE 门，继承 V8 教训）

**生成器自身先过语义门**：在 .38 加载 Flash-BF16 后，主头在真实文本（cmb_eagle_30k 若干条）上的 next-token CE 必须落在 vendor 量级（~2–4，远低于 ln V≈11.95）。
- 不过门 → 不 rollout（垃圾生成器产出的"on-policy 数据"仍是垃圾）。
- 这一门同时验证：权重加载/模板/tokenizer 全链路正确。

## 三、阶段 A —— Rollout（.38，现在可启动）

1. **五域覆盖（要求）**：语文 / 数学 / 代码 / 英语 / 科学推理。域源候选（`/mnt/share_space/l00840580/sft-old-data/` 现成 + repo 现有）：

| 域 | 数据源候选 | 备注 |
|---|---|---|
| 语文 | `megatron_chinese_cosmopedia`（+ `megatron_skypile`） | 中文 |
| 数学 | `megatron_openwebmath`（+ cmb 数学样本） | |
| 代码 | `megatron_github_code`（+ cmb 代码样本） | cmb_eagle_30k 本身代码占比高 |
| 英语 | `megatron_english_cosmopedia`（+ `megatron_fineweb_english_*`） | |
| 科学推理 | cmb_eagle_30k 的 `<think>` 推理样本（+ `megatron_pes2o_v3` 科学文本） | |

2. **口径**：greedy/argmax（对齐接受率验证）；**总长 = 4096 token/条**（prompt ≤1024 + 续写至 4096；`max_new_tokens = 4096 − len(prompt)`）。
3. **规模**：`N_seqs = 2000 步 × GBS`（GBS 由 V10 主方案定）。**按 32k 条储备** → **五域等权 6.4k/域起步**（可按需加权）；每条 prompt 抽取时打 `domain` 标签。
4. **产物**：`data/lf_data/flash_onpolicy_<domain>.jsonl`（conversations 结构：human=原 prompt，gpt=Flash 自生成 response，外加 `domain` 字段）——LF 侧无感替换，且验证可按域报接受率。
5. **对照设计（保单变量）**：每域取**同一批 prompt**，两边各跑一次——off-policy（人类原文本）vs on-policy（Base 生成）；同域内唯一变量 = response 分布。域间覆盖扩大与 response 蒸馏两个效应**分开归因**。

## 四、阶段 B —— 训练（挂在 V10 里程碑上，单变量递进）

| 步 | 内容 | 解锁条件 | 变量 |
|---|---|---|---|
| **B1** | `v10_trainer` + on-policy 数据，配方=V10 现有 MTP loss，**只换数据** | V10 Phase 2 就绪 | 数据 |
| **B2（可选）** | 移植 serving_chain 软标签项（CE + w·KL(verifier‖draft) + agree），verifier=冻结 Flash 主干、vlp detach | V10 Phase 4 就绪，且 B1 有信号 | 配方 |

- B1 单独回答「on-policy 数据是否有效」；B2 才引入软标签。不混合。

## 五、验证口径

1. **接受率（主）**：draft argmax vs verifier argmax 命中率，按 depth 1..K 分别 + 整体（复用 `bench_acceptance.py` 思路）。
2. **吞吐（次）**：tokens/s。
3. **对照基线**：**同栈同模型（Flash）+ off-policy（cmb_eagle_30k）**——V10 主方案 Phase 2 的首次训练即是基线；V7.0(W4A8) 结果不可比（模型不同）。

## 六、执行顺序（现在 → 以后）

1. **现在（.38 空）**：① 语义 CE 门（HF 加载 Flash-BF16 验主头 CE）→ ② 写 rollout 脚本（prompt 抽取 + greedy 生成 + 拼 jsonl）→ ③ 按 V10 主计划定的 GBS 出数。
2. **V10 Phase 2 就绪后**：B1 训练（.38 或主方案指定机器）。
3. **V10 Phase 4 就绪后**：B2（可选）。

## 七、纪律

- 单变量：rollout 口径（greedy）、数据量、配方各自单独消融。
- 诚实汇报：接受率 per depth / 吞吐 / 与 off-policy 基线 diff / 没验证什么。
- 本方案不碰 .40、不碰 V8（已冻结）。
