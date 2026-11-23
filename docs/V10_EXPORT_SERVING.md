# V10 导出技术方案：CP（mcore 分片）→ vLLM serving 目录

> 本文 = 导出器的权威设计。脚本（scripts/v10/v10_export_serving.py）必须与
> 本文逐条对得上；不一致 = 脚本 bug。变更流程：先改本文，再改代码。
> 状态：v1（2026-11-22，本会话全部踩坑固化后成文）。

## 1. 目标与边界

- 输入：V10 训练检查点（16 rank 分片，`cp{step}_rank{r}.pt`，每片 = fp32
  master 的**可训练集**：decoder LoRA hooks + 完整 MTP 子树）+ vendor BF16
  目录（GLM-5.3-Flash-BF16）
- 输出：vLLM 可直接 serve 的模型目录（HF 格式，vendor 命名）
- 不变量：**未训练参数必须与 vendor 逐字节一致**；训练参数 =
  vendor + 训练 delta（可逐元素对拍）
- 不做的事：主模型 routed experts/embedding/lm_head 永不动（冻结契约）；
  router expert_bias 是 buffer 且 update_rate=0 → 用 vendor 值

## 2. 数据流三阶段

```
阶段 L（加载）：16 rank 分片逐个 torch.load（CPU，map_location=cpu）
                → 必须断言 blob["step"] == 期望步数（防错 CP）
阶段 M（合并）：按 §3 映射表逐参数装配 → put()（shape 断言 + 审计记录）
阶段 W（写出）：symlink 农场 + 分组补丁（§4）
```

## 3. 命名与装配规则（权威表）

### 3.1 LoRA 合并（decoder，259 对，r=16 α=32 scaling=2.0）

| 目标（mcore 后缀） | 层型 | vendor 键 | TP 装配 |
|---|---|---|---|
| self_attention.linear_q/k/v_proj.lora_* | KDA | self_attn.q/k/v_proj.weight | column |
| self_attention.linear_o_proj.lora_* | KDA | self_attn.o_proj.weight | row |
| self_attention.q_b_proj.lora_* | DSA | self_attn.q_b_proj.weight | column |
| self_attention.kv_b_proj.lora_* | DSA | self_attn.kv_b_proj.weight | column |
| self_attention.o_proj.lora_* | DSA | self_attn.o_proj.weight | row |
| mlp.linear_fc1.lora_* | dense(li<3) | mlp.gate/up_proj.weight | column + **fc1 交错逆拆** |
| mlp.linear_fc2.lora_* | dense | mlp.down_proj.weight | row |
| mlp.shared_experts.linear_fc1.lora_* | moe | shared_experts.gate/up_proj.weight | column + **fc1 交错逆拆** |
| mlp.shared_experts.linear_fc2.lora_* | moe | shared_experts.down_proj.weight | row |

- column：A = rank0（replicated），B = 16 rank 沿 dim0 拼接
- row：A = 16 rank 沿 dim1 拼接，B = rank0（replicated）
- delta = (B @ A) × scaling

### 3.2 fc1 rank-interleave 规则（本会话最重要陷阱）

mcore fc1 权重按 rank 交错：`fc1_r = [gate_r; up_r]`，其中
`gate_r = vendor_gate[r*f_loc:(r+1)*f_loc]`。16 rank 拼接后 =
`[g₀;u₀;g₁;u₁;…]`。**逆拆必须按 rank 位置抽取**：

```
gate[i*f_loc:(i+1)*f_loc] = full[i*2f_loc : i*2f_loc + f_loc]
up  同理取后半段
```

朴素 `[:half]/[half:]` 切法 = gate 混入 up（已实测：served draft 崩塌
D1=0.098）。适用于：dense fc1（f_loc=768）、42 层 shared experts、
MTP shared experts、MTP 288 routed experts（f_loc=128）。
LoRA delta 的行布局同为交错，同样逆拆。

### 3.3 MTP 子树（vendor = layers.45.*，mcore = mtp.layers.0.*）

| mcore | vendor | TP 装配 |
|---|---|---|
| enorm/hnorm | enorm/hnorm.weight | replicated |
| eh_proj | eh_proj.weight | column（**raw 顺序不换**，fill 终审）|
| final_layernorm | shared_head.norm.weight | replicated |
| mtp_model_layer.input_layernorm | input_layernorm.weight | replicated |
| mtp_model_layer.pre_mlp_layernorm | post_attention_layernorm.weight | replicated |
| mlp.router.weight | mlp.gate.weight | replicated |
| mtp_model_layer.self_attention.*（14 键）| self_attn.*（_DSA_MAP 逆）| q_b/kv_b=column，o_proj=row，**其余 replicated**（q_a/kv_a_proj、全部 norms、全部 indexer.*）|
| mlp.shared_experts.fc1/fc2 | shared_experts.gate/up/down | column+交错逆拆 / row |
| mlp.experts.local_experts.{0..287}.fc1/fc2 | mlp.experts.{e}.gate/up/down | column+交错逆拆 / row |

TP 装配模式的**权威来源 = v10_fill.param_shard_mode(name)**（fill 用什么
模式写，导出就逆什么模式）。禁止凭形状猜测。

## 4. 写出策略

- serve 目录必须在 **NFS**（容器可写层 100% 满陷阱）或本地大盘
- 未修改文件 = **symlink** 指向 vendor（vLLM 可读穿透）；仅修改分片实体拷贝
- 补丁 = **按分片分组、每分片只 load+save 一次**（864 个 MTP 专家键如果
  逐键重写 2GB 分片 = ~2h 重复 IO，已实测）
- 换臂幂等：所有合并从 vendor 原始张量重算 → 同一 serve 目录可逐臂重打

## 5. 验证协议（四级）

1. **逐参数 shape 断言**（put 内置，vendor 形状不符即 raise）
2. **审计块**：逐组件 delta_norm 的 n/mean/max —— 非零且量级合理
   （LoRA delta ~0.1-0.2）；全零 = 没合上，爆炸 = 映射错
3. **数学对拍**（人工，标准步骤）：随机抽一列训练目标，
   `(serve - vendor) == CP 分片重算的 delta`，逐元素 max < 1e-2
   （2026-11-22 cp500 q_proj 实测 6.98e-09 PASS）
4. **serving 功能验证**：vLLM 起服后 SpecDecoding 指标 D1 应回到正常域
   （base=0.843 域）；崩塌（如 0.098）= 权重映射错误

## 6. 已知陷阱清单（本会话实录，全部已入代码或流程）

| # | 陷阱 | 对策 |
|---|---|---|
| 1 | fc1 gate/up rank-interleave | fc1_uninterleave + 单测 |
| 2 | 补丁逐键重写分片（864 键 ~2h）| 按分片分组，每片一次 |
| 3 | vendor 全量 copytree 打爆容器盘 | symlink 农场 |
| 4 | safetensors 写 NFS（EFAULT）| 写本地盘 |
| 5 | blob["step"] 不校验 | 阶段 L 断言（待入码）|
| 6 | TP 模式凭形状猜 | 只认 param_shard_mode |
| 7 | /dev/shm noexec + NFS TMPDIR | TMPDIR=/tmp（/data2 bind）|

## 7. 审计结论（脚本 vs 本方案）

见 git 历史对应版本；任何 ✗ 项 = 阻断交付。
EOF
git add docs/V10_EXPORT_SERVING.md && git commit -m "docs(V10): 导出技术方案 v1——命名/TP装配/fc1交错/写出/四级验证 + 陷阱清单(铁律#8 方案先行)" && bash /mnt/share_space/l00840580/mtp_git_push.sh push master 2>&1 | tail -1