# V8 铁律（2026-11-18 立法，用户令）

> **用户决策**：启动真正的 V8 = **LlamaFactory + Megatron** 架构。
> 本文档是 V8 全生命周期的不可违反规则。任何一条被违反 = 停下修流程，不是绕过。

---

## 第 0 条（最高优先级）：栈锁定，禁止静默替换

**V8 的栈 = LlamaFactory + Megatron，一个字都不许偷偷改。**

1. 组件锁定表（下表）中的每一项都是**合同**，不是建议。
2. 任何组件的更换/替换/绕过/降级——包括"临时试试""等价替换""顺手改一下"——**必须先显式提案，用户批准后写入本文档，才能动手**。
3. 代码里出现锁定表之外的框架 import（mindspeed / megatron-bridge / deepep / 任何未列组件）= 违规，review 必须打回。
4. 每次迭代的交付说明必须列出**本次实际使用的组件清单**，与锁定表逐项对齐；出现新增依赖 = 显式申报。
5. V7 教训固化为条款：megatron-bridge（无 glm_moe_dsa，已否决）与 mindspeed（0.12.1 与 0.18.2 错配，仅作 DSA 参考源码）**永不进入 V8 依赖**。

## 第 1 条：组件锁定表

| 层 | 组件 | 版本/形态 | 状态 |
|---|---|---|---|
| 编排/数据 | **LlamaFactory** | git tree（/mnt/share_space/l00840580/LlamaFactory），数据管线 cmb_eagle_30k + glm5_next_text | 锁定 |
| 训练核 | **Megatron-Core** | **0.18.2**（venv_v7 source install + mc0182_dpa_k_channels.patch，用户批准扩展：dot_product_attention k_channels + schedules.py cuda→npu×3）；pypi 无 0.18.x，main=0.21 不追 | 锁定 |
| 量化驻留 | **SlimW4A8Linear**（自研 scripts/v8/v8_slim.py） | packed int8 驻留 + per-use unpack/dequant + LoRA（旋钮：V8_LORA_R/ALPHA、GLM53_LORA_DROPOUT）；bf16-native 逐位一致已验证。**dropout×TP 限制（方案已裁决）**：row-parallel 的 x 跨 rank 复制，F.dropout 各 rank 独立采样 → 求和后 ≠ 单次 dropout 语义。三个选项：
(A) mcore 标准解——dropout 包进 `tensor_parallel.get_cuda_rng_tracker().fork()`（TP forked generator 各 rank 同序列 = 同 mask，mcore 对 TP 区内 dropout 的原生机制）；
(B) rank0 生成 mask → TP broadcast；
(C) dropout 跑 TP1。
**Phase 5 裁决**：同配方对照跑两边 **dropout=0**（branch-1 设 GLM53_LORA_DROPOUT=0）——dropout 是正则项不是配方 loss 语义，置 0 隔离栈差归因；dropout=0.05 只进最终判决训练（配方案 A）。门 2c 逐位判据在 dropout=0 下不受影响。 | 锁定 |
| MTP 契约 | **scripts/v8/v8_mtp.py** | own embed + rot（每深度）+ draft head + _postprocess 单 delta | 锁定 |
| norm | **HFCompatRMSNorm**（scripts/v8/v8_model.py） | fp32 mean/rsqrt + bf16 weight-multiply；q/kv_layernorm eps=1e-6，主 norm 1e-5 | 锁定 |
| NPU 运行时 | torch 2.10.0 + torch_npu **2.10.0.post4**（venv_v7） | 红线：系统 post7 段错误史 | 锁定 |
| numpy | **2.4.6** | pyarrow≥2.0；mcore source install 会降级——装完必查 | 锁定 |
| MoE 专家路径 | SequentialMLP（moe_grouped_gemm=False）| CUTLASS grouped_gemm CUDA-only；npu_grouped_matmul 自研路径 = 待显式提案的候选，未批准不用 | 锁定 |

## 第 2 条：数值铁律（全部实测校准，禁止拍脑袋）

1. **bf16 噪声地板（实测）**：单 attention context 1.85e-3；softmax 放大后 ~3.9e-2；4 层 logits ~12%。**bf16 差异 ≠ 结构 bug**。结构判定 = fp32 对拍或 argmax 主判据；rel_l2 容差必须按实测地板校准。
2. **norm**：fp32 mean/rsqrt + bf16 weight-multiply（HFCompatRMSNorm，实测 L0norm1 逐位归零）；q/kv_layernorm eps=1e-6（transformers/vendor 默认），主/final norm = rms_norm_eps 1e-5。
3. **activation = SiLU**。TransformerConfig 默认 GELU 是陷阱（V7 parity 第一根因）。
4. **mtp=False ⇔ config.mtp_num_layers=None**（上游不变量；违反 = process_mtp_loss 把 logits 砍半）。
5. **rope**：mcore MLA [evens|odds]+rotate_half 与 vendor interleave 逐值等价（已验证）；q/kv 切片 [192|64]；theta 8e6 进 config（GPTModel ctor 的 rotary_base 被 config 遮蔽）。
6. **scale = 1/sqrt(256)** 两侧一致；softmax 必须 fp32（attention_softmax_in_fp32 默认 True，不许关）。
7. **DSA spec 在 mcore 是 TE-coupled**（assert transformer_engine）——NPU 一律走 V7CoreAttention wrapper（吸收 k_channels/v_channels）。GroupedMLP 的 CUTLASS grouped_gemm CUDA-only——moe_grouped_gemm=False。
8. **LAYOUT TRUTH（rope-probe rel=0.0 + parity 双验证，审计 round-13 定案）**：
   GPTModel.forward / model.embedding 收 **[B, S]**；embedding 内部 transpose(0,1) →
   decoder/attention 走 **seq-first [S, B, H]**（transformer_block docstring 正确，
   "端到端 batch-first" 的早期论断作废）；logits 回 [B, S, V]。recipe 的 h0 索引 =
   `h0[c0:, 0]` / `h0[:, 0]`（seq-first）。chain 注意力 = `hd_seq.transpose(0,1)` +
   `attn_out[-1]`（最后一次序行 = 当前深度）。
9. **MLA rope 来源**：rotary_seq_len（顺序位置），position_ids 对 rope 无效
   （gpt_model.py:352 对 MLA 跳过模型级 rotary）。chain 的深度移位位置 c0+i+j+1
   与顺序 0..k 按 rope 相对性等价（q 深度 k vs keys j：偏移 k-j 与 c0/i 无关）——
   probe 实证 h0 rel=0.0。
10. **verifier 流按行分块**（GLM53_VERIFIER_ROWS，默认 128）：一次性 [S, V] fp32
    logits+log_softmax ≈ 3.8G 会 OOM；行分块瞬态 ~80MB。branch-1 用 vocab 切片
    流式，二者等价（argmax 逐位、log-prob bf16 GEMM tiling 级漂移 <0.1）。

## 第 3 条：环境铁律

1. **PATH**：run_joint_lf.sh 类入口必须 `dirname $PY` 前置（venv 优先）。系统 python 前置 = torchrun 解析错、ranks 无 peft（V7 首启失败根因）。
2. **装 mcore 后必查 numpy**（source install 副作用降级 1.26）。
3. **禁 dynamo/triton-ascend 前提下**：TORCHDYNAMO_DISABLE=1（jit_fuser=torch.compile 在 NPU 无 driver 会崩）。
4. 16 卡资源在 .38：训练前由用户释放 teacher 负载；v8 启动器不得抢占在跑作业。

## 第 4 条：验证铁律（design-code-test-deploy 的 "test" 定义）

每一迭代步的四件套缺一不算完成：

1. **单元**：pytest 全绿（现有 40+ 锚 + 新增），锚值改动必须附原因。
2. **1 卡 smoke**：fill 全覆盖（0 missing）+ forward/backward + determinism bit-exact + 冻结泄漏=0。
3. **边界 bisect**：emb→norm→attn→norm→mlp 逐层 rel——第一个非零层命名 culprit；中间量视图优先于隔模块推理。
4. **parity 门**：结构判定 fp32 对拍或 argmax≥0.99；容差按第 2 条实测地板。
5. **16 卡部署验证（.38）**：TP16 fill 全 rank 一致 + 首步 + CP 保存/加载 round-trip。

## 第 5 条：工作方式铁律

1. **单变量纪律**：一次一个变量；栈迁移与配方验证永不同做。
2. **根因断言纪律**：未经测量证实的"根因"不得作为结论陈述（V7 教训：norms→rope→scale 三次改口）。
3. **两线分离**：V7.0 判决训练（branch-1，在跑）与 V8 迁移线严格分称，资源不互占。
4. **诚实汇报**：每次迭代交付必须包含"本次改了什么/验证了什么/没验证什么"，禁止只报好数字。有限 loss ≠ 语义正确（V8 链注意力轴向互换的教训：loss 有限且下降、轴向却反了）。
5. **语义差异申报**：与 branch-1 参照实现的每一处语义偏差必须书面记录。当前已申报的 known-open：① ladder 形状 vs V7.0 锚相反（13.8→10.9 vs 2.46→7.33，门 2b 未过，需同 batch 同数据 per-depth CE bisect）；② LF -100 监督掩码未迁（Phase 2 用 labels=input_ids 全 LM loss，Phase 4 正式门必须迁）；③ recipe 模式 ce_main 归属——h0 detach + 无主干 CE 项，主干 decoder 的 LoRA 在 recipe 模式不更新（branch-1 同构语义，待文档化确认）。
6. **报告字段 detach**：report-only 的张量取值一律 .detach() 后 float（requires_grad 标量 UserWarning = 卫生违规）。

## 第 5.5 条：V8 自包含（round-15 立法）

V8 的全部代码/资产/测试/补丁位于 **scripts/v8/**（v8_model/v8_slim/v8_mtp/
fill_vendor/v8_trainer/v8_recipe/v8_data/vendor_mcore_mapping*/mc0182 patch/
parity_check + tests/）。scripts/v7/ 仅存 V7-era 侦察/PoC 工件（骨架、w4_poc、
slim_tp_linear、日志、param dump），不是 V8 依赖。符号与环境旋钮去 v7 化
（build_v8_model / apply_v8_mtp_contract / V8_LORA_R / V8_DEBUG_SHAPES）。
新增 V8 代码一律进 scripts/v8/，禁止再向 v7/ 添加 V8 依赖。

## 第 6 条：V8 Phase 计划（持续迭代）

- **Phase 0（完成 ✓）**：venv_v7 + mcore 0.18.2 + patch + numpy 2.4.6
- **Phase 1（完成 ✓）**：骨架 39864 参数、映射 v4（39864/39864 全覆盖）、fill、slim 驻留、MTP 契约（含 K 链每深度 rot）、HFCompatRMSNorm；40/40 测试绿；dense 层 parity 逐位一致
- **Phase 2（当前）**：**训练循环 = LF 数据 + mcore 核**
  - 设计裁决：mcore_adapter McaTrainer（patcher 不适用 W4A8、需注册 glm_moe_dsa）vs **薄自定义 trainer**（LF dataset + mcore forward_backward_func + LoRA optimizer）——按最小新代码面裁决，结论写入设计文档
  - 交付：v8_data.py + v8_trainer.py + launch_v8_38.sh
- **Phase 3**：16 卡部署验证（.38）：TP16 fill → 首步 → 步时 → CP round-trip
- **Phase 4**：V5.4a 配方语义迁移（KL/agree-CE/margin/serving_chain 几何）+ 判决门接入
- **Phase 5**：与 V7.0 判决结果对照（同 recipe 同数据，栈差归因）
