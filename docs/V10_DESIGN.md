# V10 设计文档 v2 — GLM-5.3-Flash MTP 判定微调（LF + Megatron-Core 0.18.2）

> 训练目标：GLM-5.3-Flash 的 MTP 层（layers.45）判定微调，产物用于投机解码
> 栈：**LlamaFactory + Megatron-Core 0.18.2 + venv_v7**（铁律锁 A 线）
> 开发目录：scripts/v10/
> 铁律：docs/V10_IRON_RULES.md
>
> **2026-11-19 栈决策（用户令）**：V10 = 描述微调线；mcore 0.20 预训练式路线由用户另行实现。
> 选 A 线（0.18.2）的依据：**昇腾上唯一有训练级执行证据的栈**（V8 在 .38 实跑
> fwdbwd+optimizer+CP+TP16）；mcore 0.20 + MindSpeed 线在昇腾只有 tiny 冒烟且 NPU 配对未验证。
>
> **语义唯一依据**：/mnt/share_space/l00840580/AIfromScratch/glm53flash-mtp/docs/ARCHITECTURE_MAPPING.md
>（参考项目 tiny/medium parity rel≤1.7e-06 验证过的语义表）+ code/glm_kda.py + code/glm_mhc.py。
> 本文档只写 0.18.2 实现方案与 Phase 0 探针结论，语义细节不重复抄写、以映射文档为准。

---

## 0. Phase 0 探针结论（2026-11-19，全部实测）

### 0.1 训练基座

- **基座 = /mnt/weight2/GLM-5.3-Flash-BF16**（599GB，120 shards，38770 键，BF16 直存 + F32 小参数）——用户裁决
- FP8 版（/mnt/weight2/GLM-5.3-Flash，62 shards，quant_method=fp8 + weight_scale_inv 128×128 块量化）**不用于训练**，仅导出 serving 时可能涉及
- **fill = plain copy，零反量化数学**（SlimW4A8Linear 不涉及）

### 0.2 架构事实（probe 实测）

| 项 | 值 |
|---|---|
| 层 | 45 层 + MTP layers.45；hidden 4096；vocab 154880 |
| 层类型 | 34 KDA（linear_attention）+ 11 DSA（full_attention，层号 3,7,...,43） |
| FFN | 前 3 层 dense（12288），其余 42 层 MoE（288 experts + 1 shared，topk 8） |
| DSA 维度 | q_lora 1536 / kv_lora 512 / nope 256 / **rope 0（全 NoPE）** / v 256；kv_b_proj **[32768,512]** |
| KDA 维度 | q/k/v_proj 各 [8192,4096]（64 heads × **head_dim 128**）；b_proj [64,4096]；f_a/g_a [128,4096]；f_b/g_b [8192,128]；o_norm [128]；o_proj [4096,8192] |
| mHC | 每层 **两套**：hc_attn_*（attn 子层）+ hc_ffn_*（ffn 子层）；fn [24,16384] / base [24] / scale [3]；n=4 流 |
| MTP | DSA attention + MoE + shared_experts + enorm/hnorm/eh_proj [4096,8192]（combined）+ shared_head.norm；**无 hc 键，标准残差** |
| 权重前缀 | `model.language_model.layers.N.`；visual.*（347 键）跳过 |

### 0.3 dtype 事实（probe 实测，修正映射文档两处）

| 参数 | dtype |
|---|---|
| 大矩阵（qkv/o/experts/fn 等） | BF16 |
| A_log [64]、dt_bias [8192]、hc_attn/ffn_base、hc_attn/ffn_scale、e_score_correction_bias [288] | **F32** |
| **hc_attn/ffn_fn [24,16384]** | **BF16**（映射文档 §7 F32 清单多写了 fn——以本探针为准；模块内 fp32 计算时 upcast，值无损） |
| **indexer ape [4,128]** | **BF16**（映射文档 §7 写 F32——以本探针为准） |
| conv1d（q/k/v 各 [8192,1,4]） | BF16（SESSION_LOG 已修正过 WEIGHT_KEY_INVENTORY 的同类错误） |

### 0.4 语义结论（映射文档 + parity 佐证）

1. **KDA ≠ GatedDeltaNet2**：g = −5·sigmoid(exp(A_log)·(f_b(f_a(x))+dt_bias)) ∈(−5,0) per head-channel；
   beta = sigmoid(b_proj) per head；chunk=64 fp32 chunk_kimi_delta_attention（Householder 三角修正，
   朴素重排数值不等价）；l2norm = x/√(Σx²+1e-6)（非 F.normalize）；q × 128^{−0.5}；
   o_norm = RMSNormGated(128, eps=1e-5) × sigmoid(低秩门)
2. **mHC**：每子层独立 HC；collapsed = Σ pre·streams → 子层输入；new_streams = post·y + combᵀ·residual；
   comb 走 softmax + Sinkhorn（列归一 1 次 + (行,列)×19）；最终收缩 = **无条件 mean**（非 learned）；
   MTP 输入单流
3. **DSA**：MLA nope-only，scaling 256^{−0.5}，无 q/k norm、无 RoPE；indexer 整体 @no_grad（冻结）：
   kpool=4 池化 + ReLU 在缩放之后 + weights_proj 32 头混合 → topk 2048 pool（展开 2051 宽 int32）；
   稀疏 mask 由 indexer 编码因果，**无独立 causal mask**
4. **MoE router**（noaux_tc sigmoid 精确公式）：fp32 logits → sigmoid → (+bias 选人) → gather 无 bias 分数
   → /(sum+1e-20) → ×2.5；e_score_correction_bias 是无梯度 Buffer
5. **swiglu clamp**：gate.clamp(max=10)（只有上界）+ up.clamp(±10)
6. **dense 层 norm 位置**：post_attention_layernorm → pre_mlp_layernorm（**不能**用 input_layernorm，
   residual 泄漏 bug 是参考项目 tiny parity rel=0.46 的根因，parity 修正过）

---

## 1. mcore 0.18.2 实现方案

### 1.1 层构建：自定义 Glm5NextLayer（Design 2，组合层）

不用参考项目的"展平 attn-only/ffn-only 90 层 + hybrid pattern"（那是 0.20 HybridStack 专属）。
用 **V8 已验证的 TransformerBlockSubmodules per-layer spec**：45 个自定义层 + MTP。

```
Glm5NextLayer(MegatronModule)  — 每层一个，全权控制 forward（不借 mcore BDA 残差）：
  子模块: attn_hc / ffn_hc (GlmHyperConnectionModule 移植)
          input_layernorm / pre_mlp_layernorm (GlmRMSNorm 移植)
          self_attention (GlmKDAAttention 或 V10DSAAttention)
          mlp (MLP dense 或 MoELayer)
  forward(streams [S,B,n·H]):
    # attn 段（HF MG:1293-1327 语义，直接实现，无 y_net−collapsed 技巧）
    pre_a, post_a, comb_a = attn_hc(streams); collapsed = Σ pre_a·streams
    y = self_attention(input_layernorm(collapsed))
    streams = post_a·y + comb_aᵀ @ streams
    # ffn 段同构
    pre_f, post_f, comb_f = ffn_hc(streams); collapsed = Σ pre_f·streams
    z = mlp(pre_mlp_layernorm(collapsed))
    streams = post_f·z + comb_fᵀ @ streams
    return streams, None
```

- 首层前 expand（embed 1→4 流复制，无参数）；最后一层 ffn 段后 mean 收缩 4→1 → model.norm
- **优点**：残差由 HC comb 流显式表达，无 BDA bypass 的 bf16 1ulp 舍入差异（参考项目 gap #4 消除）
- MTP：复用 V8 v8_mtp.py contract（enorm/hnorm/eh_proj/内层 DSA+MoE/final norm = shared_head.norm），
  维度换 Flash；mtp=False ⇔ config.mtp_num_layers=None

### 1.2 模块移植清单（参考项目 → v10，全部纯 torch）

| 源 | 改动 |
|---|---|
| glm_kda.py GlmKDAAttention | 基本直移（tensor_parallel.Column/Row、MegatronModule 0.18.2 均有）；forward 返回 (out, bias) 0.18.2 约定一致；conv1d 三键 cat 成 [24576,1,4] 移到 fill/转换 |
| glm_mhc.py GlmHyperConnectionModule + expand/mean_contract | 直移；fn 参数声明 F32（fill 时 bf16→fp32 upcast 值无损）；mark_keep_in_fp32 保留 try/except |
| GlmRMSNorm | 直移（0.18.2 non-TE 无 RMSNorm 类，同 V8 的 HFCompatRMSNorm 模式，eps 按名分配） |
| DSA（0.20 experimental_attention_variant 的 reference 纯 torch 路径） | **从 0.20 源码移植**成 V10DSAAttention（0.18.2 无此模块）；indexer @no_grad 冻结；mask 语义按 NPU 规则（masked_fill 必须 bool mask） |
| MoE | 0.18.2 TopKRouter + SequentialMLP（V8 已验证 sigmoid+expert_bias 路线）；**clamp 走 0.18.2 原生 activation_func_clamp_value（本地 Megatron-LM mlp.py:221-223 已核实语义一致：gate.clamp(max=v) + up.clamp(±v)），无需自写** |
| norm eps | 主/q_a/kv_a norm eps=1e-5（rms_norm_eps）；indexer k_norm 是 LayerNorm(128, eps=1e-6) 带 bias；hc 内 RMSNorm 无参数 eps=1e-5 |

### 1.3 TP 布局（同参考项目，0.18.2 语义相同）

KDA：q/k/v/b/fb/gb Column、o Row、fa/ga duplicated、A_log/dt_bias/conv TP-dim0。
DSA：q_a/q_b/kv_a/kv_b Column、o Row、indexer duplicated。
MoE：SequentialMLP + router（V8 模式）。mHC：fn/base/scale replicated。
首期 TP=1 验证语义 CE 门，TP>1 的 MLA down_proj duplicated gap 见映射文档 §8.8。

### 1.4 fill 与训练循环

- fill：BF16 plain copy（F32 参数保持 F32；hc_fn/ape 等 BF16 键 → 模块参数 upcast 规则见 0.3）；
  键名映射按映射文档 §7（mcore 键名按 0.18.2 实际构建结果重新盘点，不照抄 0.20 键名）
- trainer：v8_trainer.py 骨架（fwdbwd + LoRA optimizer + CP）— **LoRA(BF16 基座) + MTP 全训**（2026-11-20 用户裁决：主权重 LoRA + MTP 都微调；基座本体冻结，无量化）
- recipe：V5.4a serving_chain（CE+margin+KL+agree），chunk-local depth chains
- data：v8_data.py LF 管线；tokenizer 与 Flash 同 vocab（154880），模板确认后复用

---

## 2. 验证阶梯（顺序不可换）

1. pytest 单元全绿（含 KDA vs HF 递归黄金参考对拍、MHC sinkhorn 双随机、DSA topk 形状）
2. **语义 CE 门（第零步）**：build → fill → 真实文本 → 主头 next-token CE 落 vendor 量级
   （预期 ~2-4）→ 不过门不得继续
3. 逐算子 bisect：emb → mHC 门 → KDA/DSA → MoE → 逐层 streams 对拍（对照参考项目
   verify_parity_cpu.py 的 HF 侧）
4. 1 卡 smoke：fill 全覆盖 + fwdbwd + determinism + frozen-leak（base/MTP 冻结面正确）
5. TP 部署验证

## 3. Phase 计划

- ~~Phase 0 探针~~ ✅ 2026-11-19 完成（本文档 §0）
- **Phase 1**：v10_model.py 骨架（Glm5NextLayer ×45 + MTP）→ 语义 CE 门
  - 前置 ✅ 2026-11-19 完成：① 0.18.2 原生 activation_func_clamp_value（mlp.py:221-223）；
    ② mcore 0.20 DSA 移植底本已 vendored：scripts/v10/reference/mcore020/{dsa,absorbed_mla,dsa_masking}.py
    （commit 45b68f914 锁定——参考项目 parity 验证过的同一版本）
- Phase 2：mapping + fill + trainer/recipe 迁移
- Phase 3：TP 部署验证
- Phase 4：V5.4a 配方 + LF 数据管线合流

## 4. 风险登记

| 风险 | 等级 | 缓解 |
|---|---|---|
| KDA chunk=64 纯 torch 循环 NPU 性能 | ★★ | 正确性优先；fla_npu kernel 后置（两栈共同瓶颈，不阻塞判定训练） |
| 0.20 DSA 移植（indexer kpool/topk/mask） | ★★ | 只移 reference 纯 torch 路径（底本已 vendored @45b68f914）；NPU masked_fill bool 红线 |
| 0.18.2 无 activation_func_clamp_value | ~~★~~ ✅ 已消除 | 0.18.2 原生支持（mlp.py:221-223 语义一致） |
| depthwise conv1d NPU 支持 | ★ | 1 卡 smoke 早期验证 |
| hc_fn/ape dtype（BF16）与模块 F32 参数 | ★ | fill upcast 值无损；测试覆盖 |
| mHC bf16 流精度 | ★ | 参考 parity 已过（rel≤1.7e-06 量级）；CE 门兜底 |
| index_share_for_mtp_iteration serving 差异 | ★ | 训练 depth=1 无影响（参考项目已确认） |
