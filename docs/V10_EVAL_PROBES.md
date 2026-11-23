# V10 评测探针脚本说明（作用与背景）

> 成文 2026-11-23。读者：接手 V10 训练/评测的任何人。
> 目标：说清 `scripts/v10/` 下这些脚本"为什么存在、各自测什么、怎么用"，
> 避免再犯"用内部探针的数字冒充有效性证据"的错误。

## 〇、背景：为什么需要这一套探针

V10 的目标是让 GLM-5.3-Flash 的 MTP draft 在 vLLM 投机解码下接受长度
K̄ 显著超过 base 的 3.521。整个 V10 系列的关键教训（`V10_TRAIN.md §11`）：

> **"训练目标优化得好" ≈ 同义反复，不构成有效性证据。**

V10 原版（纯 CE-on-gold）在 mcore 探针上显示大幅改善（argmax-agree
+14 点），但 serving 真值 K̄ 反而从 3.52 掉到 2.49（−29%）。根因：探针
测的是训练目标自身（CE on 训练分布），而 serving 验收按的是"draft 是否
等于 main 会选的 token"——两者在窄域内重合、在分布外分叉。

因此 V10 的评测分成**三个互不替代的层次**：

| 层次 | 口径 | 可信度 | 用途 |
|---|---|---|---|
| serving K̄（vLLM /metrics）| 原生前缀接受长度 | **真值** | 唯一有效性结论 |
| EVAL-FIXED 探针（固定 token）| main/draft CE + argmax 一致率 | 趋势 | 训练健康监控 |
| train-batch 探针（随 step 的 batch）| 同上 | 信号（有噪声）| 同 step 的即时读数 |

## 一、脚本清单

### 1. `v10_trainer.py`（训练器，自带双轨 diag）

- **作用**：MTP 全参 + 主模型 LoRA 的训练主循环。`native_loss` 含
  KL(main‖draft)/agree-CE/margin/draft-CE 四项，LR 分组（MTP 快 / LoRA 慢）。
- **自带 diag**（每 10 步打印，无需另跑脚本）：
  - `GRAD AUDIT step1`：lora-grads / mtp-grads / frozen-with-grad。
    **`mtp-grads` 必须非零**——它曾在 `dec.detach()` 引入后掉到 0
    （MTP 全参冻结，见 `V10.2_CONSULT.md §五b`），是"训练是否真的在动 MTP"
    的硬判据。
  - `train batch — the signal`：当前 batch 的 main/draft CE + argmax-agree
    （draft **vs main** 口径）。
  - `EVAL-FIXED — the trend`：固定 token（`train_ds[0]`）的 main/draft CE +
    argmax-agree（draft **vs gold** 口径）。

### 2. `v10_diag.py`（诊断原语，可微 loss 的实现）

- **作用**：`chunked_full_vocab_ce` / `chunked_topk_kl` / `chunked_agree_margin`
  等分块全词表可微算子。所有训练 loss 数字和探针 CE 都走这里的
  `gather_from_tensor_model_parallel_region`（内部 vocab CE 在黑名单 #10）。
- **注意**：`chunked_topk_kl` 是 Top-128 截断 KL（非全词表），语义是"只对齐
  main 的 top-128 支撑内排序"，支撑外的质量由 agree-CE 兜底——这是已知取舍，
  不是 bug。

### 3. `eval_step0_baseline.py`（Step0 基准探针，本次新增）

- **作用**：在**任何训练之前**，用与训练器完全相同的固定 token
  （`train_ds[0]`，pad 到 seq_len）跑一次 EVAL-FIXED，打印 4 个数：
  `main CE / draft CE / draft-vs-gold / draft-vs-main`。
- **为什么存在**：训练器 diag 从 step 10 才开始打印，没有 step 0 的锚点，
  导致"0.6491 是涨了还是跌了"无法判断（详见 §二）。本脚本补上这个锚点。
- **用法**（16 卡，同训练环境）：
  ```bash
  export V10_VENDOR_DIR=/root/serve_v101_cp100   # 或真正的 vendor BF16 目录
  torchrun --nproc_per_node=16 --master_port=29xxx scripts/v10/eval_step0_baseline.py \
      --yaml configs/glm53_v10_train.yaml
  # 可选：同时快照某个训练 CP，并打印相对 step0 的 delta
  torchrun ... eval_step0_baseline.py --yaml ... \
      --load-cp saves/v10_t4/cp100_rank{rank}.pt
  ```
- **判读**：拿 `draft-vs-main`（不是 `draft-vs-gold`）作为 V10.2 有效性的趋势
  指标——因为 serving 验收按 main 的 argmax 计。

### 4. `eval_cp50_40.py` / `eval_d15_40.py` / `eval_groups_40.py`（历史评测）

- 分别是 CP50 效果、D1-D5 深度矩阵、分组（G1/G2/G3）评测。它们用
  `h0-test/eval35/v10_mcore_ce16p_40.json` 或训练数据做 token，口径与
  训练器 EVAL-FIXED **不完全一致**（见 `V10_TRAIN.md` 台账 #24），只作诊断，
  不作 serving 替代。

### 5. `v10_export_serving.py`（导出器）

- 把 CP 分片 + vendor 合并成 vLLM 可 serve 的目录。方案见 `V10_EXPORT_SERVING.md`。

## 二、关键口径区分（容易踩的坑）

训练器 diag 的两行 `argmax-agree` **不是同一个量**：

| 打印行 | 实际含义 | 对 V10.2 的意义 |
|---|---|---|
| `diag ... argmax-agree=0.9099 (train batch)` | P(argmax(draft) == **argmax(main)**) | **相关**——acceptance 按 main argmax 计 |
| `EVAL-FIXED ... argmax-agree=0.6491` | P(argmax(draft) == **gold**) | 次要——CE 命中率，与 acceptance 无直接关系 |

- 0.9099 和 0.6491 差 26 点，是因为一个是"draft 跟 main 对齐"，一个是
  "draft 猜中 gold"。两者**都真实**，不是"假象"，但别拿错口径下结论。
- `agree-CE`（loss 里的一个分量，约 0.3）跟"argmax 一致率"也不是一回事：
  `agree-CE=0.33 ⇒ draft 对 va 的平均概率 exp(−0.33)≈0.72`，而 argmax 命中
  只需 va 概率最大（150k 词表上 P=0.1 就可能命中），所以
  "CE=0.33 与 argmax-agree=91%"**不矛盾**。

## 三、推荐工作流（每轮训练）

1. **先测 Step0 基准**：`eval_step0_baseline.py`（不用等训练），拿到
   `draft-vs-main` 的起点。
2. **训练中**：盯 `diag step N` 的两条线——`mtp-grads` 非零、`EVAL-FIXED
   draft-vs-main` 相对 Step0 的**趋势**（不是绝对值）。
3. **训练后**：CP 落盘 → `v10_export_serving.py` 导出 → vLLM serving 跑
   K̄（铁律 #7，唯一有效性结论）。探针只做诊断，不替代 serving。
