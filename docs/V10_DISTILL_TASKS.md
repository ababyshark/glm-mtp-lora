# V10 蒸馏任务清单（On-Policy Rollout + 训练接入）

> 2026-11-19 立项。配套：`V10_ONPOLICY_DISTILL.md`（方案）/ `V10_DESIGN.md` / `V10_IRON_RULES.md`。
> 本文件 = 可执行任务分解 + 已核实的机器状态。**执行顺序 = T0 → T1 → T2 → T3 → T4 → T5，不跳步。**

---

## 零、已核实的机器状态（2026-11-19 实测，勿再重复探测）

### .38（A3-syn-28，蒸馏执行机，16 卡全空，SSH: root/Huawei@1234）
- **机器被重置过**：旧 venv_v7（torch 2.10.0 + torch_npu post4）已不存在。
- 现有栈：miniconda **python 3.13.9** + torch **2.7.1** + torch_npu **2.7.1.post3.dev**（**加载失败**）+ transformers **5.5.4**（import 损坏，根因同 torch_npu 链）。
- CANN：8.3.RC1 / 8.3.RC2 / cann-8.5.T102 混装；driver 在 `/usr/local/Ascend/driver`。
- **torch_npu 报错根因**：`ImportError: libhccl.so: cannot open shared object file`；已试 driver/lib64 + toolkit latest lib64 进 LD_LIBRARY_PATH **未解决**（待定位匹配的 CANN 组合，或换 venv 栈）。
- **权重路径（注意：与 .40 不同！）**：.38 挂的是 `/mnt/weight`（NFS 10.246.63.101:/data2）：
  - ✅ `/mnt/weight/GLM-5.3-Flash-BF16`（**非量化，本方案用**，120 分片）
  - 另有 FP8 / W8A8 / 0day / DFlash2 等变体
  - ❌ `/mnt/weight2` 在 .38 **不存在**（那是 .40 的挂载点）
- `/mnt/share_space`（repo）✅ 可见。

### 本开发机（A3-syn-32）
- 0 卡 NPU；repo 与 Flash-BF16 权重经共享盘可见；仅用于写脚本/文档，不跑模型。

---

## T0 — 环境修复（.38，阻塞项，先做）

- [ ] **T0.1 torch_npu 可用**
  - 现象：libhccl.so 加载失败；LD_LIBRARY_PATH 直加 driver+latest lib64 无效。
  - 候选解法（按序试，一次一个）：
    a. 找到 `cann-8.5.T102/aarch64-linux/lib64`（libhccl.so 实际所在）全路径入 LD_LIBRARY_PATH 再试；
    b. 确认 torch_npu 2.7.1.post3.dev 要求的 CANN 版本，选 8.3.RC2 或 8.5.T102 对应 `set_env.sh` 完整 source（含 opp/compiler 子路径）；
    c. 若 a/b 不行：**新建 venv**，装与 .38 driver 匹配的 torch+torch_npu 组合（rollout 只做推理，版本不必对齐 V8 训练栈，能跑即可）。
  - 验收：`python -c "import torch, torch_npu; print(torch.npu.device_count())"` → `16`。
- [ ] **T0.2 transformers 可用且支持 glm5_next**
  - 验收：`AutoConfig.from_pretrained('/mnt/weight/GLM-5.3-Flash-BF16', trust_remote_code=True)` 打出 `model_type=glm5_next`；`AutoModelForCausalLM`/`Glm5NextForConditionalGeneration` 可加载（meta/CPU 先验结构，NPU 单卡实载在 T1）。

## T1 — 语义 CE 门（铁律 §4 step ZERO，不过门不继续）

- [ ] 单卡（.38 npu:0）加载 Flash-BF16，cmb_eagle_30k 取 8~16 条真实文本；
- [ ] 主头 next-token CE ≈ **vendor 量级（~2–4）**，远低于 ln V=11.95；
- [ ] CE 数值与样本 id 记录在案（日志文件）。
- **验收：CE 达标。不过门 → 回 T0 查权重加载/模板/tokenizer，禁止进 T2。**

## T2 — Rollout 脚本（.38）

- [ ] **T2.1** `scripts/v10/gen_onpolicy_prompts.py`：**五域** prompt 抽取（语文/数学/代码/英语/科学推理），源映射见 `V10_ONPOLICY_DISTILL.md` §三（sft-old-data 现成语料 + cmb 推理样本），每条打 `domain` 标签，五域等权。
- [ ] **T2.2** `scripts/v10/rollout_onpolicy.py`：16 卡并行（每卡 1 进程、shard 1/16 prompt），greedy，**总长 4096（prompt ≤1024 + 续写，`max_new_tokens = 4096 − len(prompt)`）**；断点续跑（按 shard 落盘）。
- [ ] **T2.3** `scripts/v10/pack_onpolicy.py`：拼 `flash_onpolicy_<domain>.jsonl`（conversations：human=原 prompt，gpt=Flash 自生成，`domain` 字段保留）。
- 验收：产物行数 = 目标条数；**五域各 ≥ 目标配额**；抽样 5 条/域人审（生成文本连贯、非乱码）。

## T3 — Rollout 规模

- [ ] `N_seqs = 2000 步 × GBS`（GBS 由 V10 主方案定）。
- [ ] 建议按 **32k 条**储备（覆盖 GBS≤16）；GBS 更大再补。
- [ ] 成本预估（HF 生成较慢）：先跑 T2.2 单卡测速 → 定 16 卡总时长 → 排窗口。

## T4 — 训练接入（B1，依赖 V10 Phase 2）

- [ ] 前置：V10 Phase 2（fill + `v10_trainer` 训练循环）就绪。
- [ ] `v10_trainer` + on-policy 数据：**只换 data-path，配方不动**（现 MTP CE loss）。
- [ ] 对照基线 = 同栈同模型 + cmb_eagle_30k（off-policy）——V10 Phase 2 首跑即基线；**V7.0(W4A8) 不可比**。

## T5 — 效果验证

- [ ] **接受率（主）**：draft argmax vs verifier argmax，按 depth 1..K + 整体，**并按五域（语文/数学/代码/英语/科学推理）分别报告**。
- [ ] **吞吐（次）**：tokens/s。
- [ ] 交付含：接受率 per depth × per domain / 吞吐 / 与 off-policy 基线 diff / 没验证什么（诚实汇报）。

---

## 风险与纪律

1. **.38 是重置后的新机器**：不要假设旧栈还在；T0 是硬前置。
2. **版本决策**：rollout 栈（torch 2.7.1 系）与 V10 训练栈（可能对齐 V8 的 2.10.0 系）**允许不同**——rollout 只做推理，独立验收。
3. 单变量：rollout 口径（greedy）→ 数据量 → 配方，各自单独消融。
4. B1/B2 不混合（B2 需 V10 Phase 4 + B1 有信号）。
5. 本清单不涉及 V8（已冻结）与 .40。
