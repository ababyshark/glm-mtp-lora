# V10 fill 说明 — 它在做什么、解决什么问题

> 2026-11-20 固化。对应代码 `scripts/v10/v10_fill.py`；语义唯一依据 `docs/V10_DESIGN.md` §0。
> 本文回答两个问题：fill 在做什么 / 为什么必须有它。

---

## 一、fill 在做什么

把 HF 格式的 GLM-5.3-Flash-BF16 checkpoint（599GB、120 个 safetensors 分片、
38770 键）**移植进手工构建的 mcore 0.18.2 模型**。纯权重搬运，零量化反解
（铁律：BF16 直存 plain copy；FP8 weight_scale_inv 仅导出 serving 时涉及）。

### 1. 键名/结构映射（HF → mcore）

mcore 没有原生 GLM Flash 结构，键名全对不上，逐键映射：

| HF 键 | mcore 键 |
|---|---|
| `model.language_model.layers.N.self_attn.q_a_proj` | `decoder.layers.N.self_attention.q_a_proj` |
| `self_attn.indexer.wq_b` | `self_attention.indexer.linear_wq_b` |
| `self_attn.indexer.k_norm.{weight,bias}` | `self_attention.indexer.k_norm.norm.{weight,bias}` |
| `layers.N.hc_attn_fn/base/scale` | `decoder.layers.N.attn_hc.{fn,base,scale}` |
| `layers.N.post_attention_layernorm` | `decoder.layers.N.pre_mlp_layernorm` |
| `mlp.experts.{e}.{gate,up,down}_proj` | `mlp.experts.local_experts.{e}.linear_fc{1,2}` |

### 2. 复合变换（多键合一）

- `q/k/v_conv1d` 三键 → cat 成 fused `conv1d_weight [24576,1,4]`
- `gate_proj + up_proj` 两键 → cat 成 gated `linear_fc1`（**逐 rank interleave，见 §三**）
- `mlp.gate.e_score_correction_bias` → mcore router 的 `expert_bias` buffer（fp32 no-grad）

### 3. TP16 切分

每 rank 只拿 1/16，切分约定与 mcore 内核严格一致：

| 模式 | 参数 | 规则 |
|---|---|---|
| column/vocab | q/k/v/q_b/kv_b/fb/gb/b_proj、embedding/output_layer | dim0 连续切 |
| row | o_proj、linear_fc2 | dim1 连续切 |
| replicated | q_a/kv_a（lora 维）、indexer、hc、norms、router、expert_bias | 全量 |
| column(dim0) | A_log/dt_bias/conv1d_weight | per-head / per-channel 布局 |

**约定来源**：mcore 0.18.2 `mlp.py` 的 `glu()` 在 **TP 本地输出**上
`torch.chunk(x, 2, dim=-1)` —— 所以 rank r 的 fused fc1 必须是
`[gate[r·f/tp:(r+1)·f/tp]; up[r·f/tp:(r+1)·f/tp]]`，即**先切后拼**。

### 4. dtype 策略 + coverage ledger

- 大矩阵 bf16→bf16 逐位拷贝；小参数 bf16→fp32 无损上转
  （A_log/dt_bias/hc fn/norms/ape/conv——probe 2026-11-19 dtype 事实表）
- **ledger 硬校验**：每个 mcore 参数恰好写一次（unwritten=∅）、
  每个非 visual 键恰好消费一次（unconsumed=∅）、`model.visual.*`（347 键）入 skipped、
  MTP 键（layers.45.*）入 deferred（Phase 2 接）。任一爆红即失败，**不给静默错的机会**。
- 形状守卫：shard 结果与参数形状不符直接 raise（v10_fill.py `assign`）。

---

## 二、解决什么问题

1. **mcore 没法直接吃 HF checkpoint** —— 不填权重，45 层模型就是 torch.empty
   垃圾，训练必是静默灾难。fill 是"可训练"的前置条件。
2. **它是设计映射表的"可执行验证"** —— fill 能 0 missing 跑通 = 架构理解
   （层类型/hc/indexer/MoE 命名/dtype）全对；差一个键，ledger 立刻爆红
   （V8 的 packed 权重就曾被 dist-ckpt 静默丢过）。
3. **它是语义 CE 门的地基** —— V8 血泪教训：有限 loss ≠ 语义正确（15+ 轮无效
   调试）。链条：`fill 正确 → 前向正确 → CE≈2-4 → 才允许训练`（V10 铁律第零步）。

---

## 三、关键陷阱（2026-11-20 修复，勿重蹈）

**cat-后-再-切 ≠ 先切-后-拼**。原实现把 `cat([gate,up])` 后对整条 dim0 连续切 TP——
rank 0 会拿到**纯 gate 段**（应得 gate[0:f/tp]+up[0:f/tp]）；conv1d 同理拿到纯 q 段。
**形状恰好全对、不报任何错**，只有语义 CE 门能抓住。修复：`composite_fuse()`
先按列切每段再 cat；回归测试 `test_fill_tp2_composite_per_rank_interleave`
（双进程 gloo 端到端）+ `test_composite_fuse_cat_then_shard_is_wrong`。

---

## 四、.40 运行手册（mtpV8 容器 / venv_v8 = torch2.10 + mcore 0.18.2）

```bash
# 1) 同步代码（.40 的 /mnt/share_space 是独立 NFS 副本，开发机改动不会自动出现）
tar -cf - scripts/v10 | ssh root@10.246.63.40 \
  "tar -xf - -C /mnt/share_space/l00840580/mtp_git/lf_glm53_qlora_mtp/"
# 2) 容器内 grep 验证（陷阱 #1：同步后必须验证）
# 3) 测试（CPU，不占卡）
/root/venv_v8/bin/python -m pytest scripts/v10/tests/ -q -p no:warnings   # 58/58
# 4) 语义 CE 门（TP16，第零步）
/root/venv_v8/bin/python -m torch.distributed.run --nproc_per_node=16 \
  --master_port=29777 scripts/v10/launch_ce_gate.py \
  > h0-test/eval35/v10_cegate_40.log 2>&1
# 判定：grep -E "gate|PASS|FAIL" —— PASS = CE_mean ∈ [1.5, 5.0] 且 CE_max < 12
```

实测（2026-11-20，SlicedCheckpoint 字节范围路径）：16 rank build 各 19.9B 参数；
fill 每 rank **只读 ~37GB**（TP× 读放大消除：mmap + data-offset 偏移直达本 rank 切片），
**~1 分钟**完成；HBM ~43G/64G 稳定无 OOM。
⚠️ 历史无效产物：首跑（01:10，offset-bug 时期的 get_tensor 全量路径 + 早期
SlicedCheckpoint 无 base 修正版）的任何 fill/CE 产物一律作废；以 base 修复后的
代码为准重跑。KDA fla_npu kernel 吞吐优化后置（正确优先）。

## 五、语义门两级制（2026-11-20 定型）

1. **启发式筛查**（launch_ce_gate.py）：CE_mean ∈ [1.5, 5.0]（均值带，max 阈值
   12.0 已废除——双语 code-switch/句首的固有高熵尖峰会误杀，实测位置 27
   的 14.65 是语言切换点而非错权）
2. **teacher 精确对拍**（teacher_ce_parity.py）：同文本同分词，vLLM teacher
   逐位置 CE + top-1 对拍。判定 = argmax 一致率 ≥0.99（铁律 argmax 主判据），
   并列翻转（teacher 侧 margin ≤0.2）单独归类不计入分叉
3. 已知残差（申报）：MoE 路由边界 + indexer bf16 matmul 的数值序差异会造成
   高熵分支位置的 argmax 翻转——CE 语义不受影响（分歧位置金标 CE 双方一致），
   最终裁决挂 T5 draft 接受率
4. **eh 半交换的裁决路径**：teacher 对拍**测不出**它（门跑 mtp=False，eh 不在
   前向路径上；对拍两侧共享 mcore 权重）——真正的裁决 = MTP 训完首次导出
   HF 时做 A/B 双份（swap/不 swap），一次实验消掉（用户令，挂 T5）

## 六、teacher 统计对拍判定（2026-11-20，.40 实测）

16 条 cmb 真实文本（带 `[gMASK]<sop>` 前缀，2093 位置）mcore vs vLLM teacher：

| 指标 | 值 | 判读 |
|---|---|---|
| CE_mean | mcore 1.7483 vs teacher 1.7212（Δ1.6%） | **分布级等价 ✓** |
| mean \|dCE\| | ~0.07 | 无系统性漂移 ✓ |
| argmax raw 一致率 | 94.65%（40 并列 + 72 非并列翻转） | 未达 0.99 |
| text0(881tok)/text7(309tok) 翻转率 | 5% / 11.3%；短文本 ~1-3% | **随长度增长、成簇爆发** |

**分歧签名（关键取证）**：爆发簇内多处"teacher 对金标高度自信（teCE≈0.1-0.2）、
mcore 惊讶（mcCE 6-16）"——信息量丢失而非双向噪声。长度相关 + 成簇 +
mcore 侧丢失置信度 ⇒ 嫌疑排序：① **DSA indexer topk 数值序**（选池分歧 =
注意力支持集丢失，直接掉信息）；② KDA chunk-state 累积；③ MoE 路由边界。
门格式伪影假设已排除（加模板前缀后一致率不升反降）。

**裁决（对拍双边共享内核序差异，vLLM 非逐位基准）**：
1. 分布级：PASS；结构级：**申报残差，不判 PASS 也不判 FAIL**
2. **不阻塞 T4 训练**：MTP 训练在 mcore 内部自洽（draft 学 mcore 自己的
   hidden states），mcore↔vLLM 差异是常数偏置，训练前后的接受率对比不受其污染
3. 残差在 **T5 导出 serving** 时由接受率实测裁决（与 eh A/B 同批实验）；
   若接受率异常，bisect 路线 = HF eager 层流式对拍（分离 KDA/DSA/MoE 三嫌疑）
4. 原始产物：`h0-test/eval35/v10_mcore_ce16p_40.json`（mcore 逐位置 CE+argmax）、
   `..._parity.json`（teacher 对拍全量），均已回传 .38 树
