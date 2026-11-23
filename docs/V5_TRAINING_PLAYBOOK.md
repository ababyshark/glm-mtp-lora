# V5 训练手册（后续 agent 按此执行）

> 更新：2026-11-16 深夜（V5.2 r2 运行中）。本文是 V5 系列训练的**唯一执行手册**：
> 目标、科学结论、机器与凭据、启动/评测/探针程序、运维红线、验收指标。
> 配套文档：`OPS_MISTAKES_AND_RULES.md`（事故清单）、`V5_JOINT_CONTRACTC_PLAN.md`
>（§21 addendum）、`V5_ACCEPT_R10_EVAL_REPORT.md`（评测账本）、
> `docs/GIT_TOKEN_WORKFLOW.md`（git 凭据）。

## 0. 目标与唯一权威指标

- **目标**：提升 GLM-5.3 W4A8 的 MTP=5 投机解码接受率（draft 头 + base-LoRA 联合微调）
- **权威指标**：vllm 引擎实测的每 draft-step 接受 token 数（本报告口径）；
  **EAL = tok/step + 1**（演进表口径，bonus token 计入）——两套账已对齐
- 两个题集：48p（`data/lf_data/consumed48_prompts.jsonl`，训练域）+
  200p（`data/lf_data/prompts200_tokenized.jsonl`，数学/通用域）——**域缺口是真实存在**的
  （vendor 在 200p 上 2.887 vs 我们 cp120 2.145，差距 -26% > 域内 -18%）
- **基线锚点**：vendor 同机 2.482（48p）/ 2.887（200p）；所有对比必须同机同配方

## 1. 轨道账本（实测，tok/step 口径）

| 轨道 | 几何 | 目标 | 关键结果 |
|---|---|---|---|
| V1 | h0-direct 递归 | 纯 CE | cp300 峰值 2.085；cp120 1.849 |
| V2 | 同 V1 + MTP-LoRA(10x lr) | 纯 CE | cp240 峰值 2.134；bf16o FLOAT 臂反而更差（1.880@cp120）——int8 地板非瓶颈 |
| **V5** | **serving_chain（interim 局部链）** | CE + ACC margin(0.1/0.5/1,1,1,2,2) | 150 步探针：cp120 **2.044**、cp150 2.033；斜率 ≈ V1/V2 的 2×；d2/d3 全配方史高 |
| V5.2（在跑） | 同 V5 + **二值自喂** | 同上 | step ~53/300，自喂冲击后恢复中（5.0→4.4） |
| V5.3（已打包待发射） | **contract_c（精确几何）** | **0.3CE + 0.7KL 蒸馏** + **scheduled-sampling ramp(20→100)** | 启动包就绪：`launch_v53_46.sh` |

## 2. 五大科学结论（后续决策的事实基础）

1. **几何错配**（interim 128-token 局部链 vs vllm 全 prompt KV）：阶梯探针实测 ckpt 的
   d5 exact-CE 比 interim 差 +0.97；d1 形态伤：exact-d1 一致率 64.1% vs vendor 78.6%
2. **曝光偏移是引擎缺口的主体**：同一权重 teacher-forced 一致率 d2-d5 大胜 vendor
   （74.6/57.2/47.8/36.1 vs 40.1/13.0/9.2/6.5），引擎却在 d2-d5 输——
   **训练喂 gold、serving 喂自己**。自喂链（V5.2/V5.3）直击此点
3. **vendor 头是单步预训练头**：深位（d4/d5）对一切 CE 训练臂都是结构性天花板
   （V1/V2/V5 引擎 d5 全部 < vendor 的 23.9%）
4. **vendor 一致率锚点**：exact 契约下 vendor d1 一致率 78.6% ≈ 引擎 d1 接受率 83-88%
   ——任何一致率探针先对齐这个锚再谈结论
5. **交付形态已排除**：int8 requant 审计通过（MTP reldiff 5.76e-3 充分表达）；
   bf16o FLOAT 臂被 V2 数据证伪——不要在这两个方向花时间

## 3. 机器、凭据、路径

### .46 (A3-syn-36) —— 训练机（16 卡）
- torch_npu **必须 2.10.0.post4**（post7 会段错误，OPS 规则 0）；python3.12.13
- LF 源码：`/mnt/share_space/l00840580/LlamaFactory/src`（含 branch-2 现实的 patched trainer）
- **长期 git 树**：`/mnt/share_space/l00840580/mtp_git`（CodeHub 已推送，最新 9d9a00e+）
- V5.2 运行树（**跑完前禁止归档**）：`/mnt/share_space/l00840580/mtp/lf_glm53_qlora_mtp`
- git 凭据：`/root/.mtp_git_token`（chmod 600）+ `/root/mtp_git_push.sh`；
  方法见 `docs/GIT_TOKEN_WORKFLOW.md`（用户名 l00840580 + token，curl 走不通该域名、用 git）

### .35 (A3-syn-25) —— 评测机（卡 4-11）
- 容器 `l00840580-glm53-32keval`（vllm 0.30.0-a3 nightly，全 16 davinci 设备）
- vendor 权重 `/mnt/weight2/GLM-5.3-W4A8`；评测脚本在原始树
  `/mnt/share_space/l00840580/mtp/lf_glm53_qlora_mtp/h0-test/eval35/`
- 卡 0 曾被批准探针用——**用前问用户**（机器分配以用户指令为准）

## 4. 训练启动程序（V5.2/V5.3 通用骨架）

**铁律**：一律脚本文件启动（不内联）；独立日志名；setsid 脱离；XDG_* 必须 unset
（曾把 HF 动态模块缓存劫持到 debug 目录， serving 过期代码事故）。

```bash
# 模板：h0-test/eval35/launch_v52_46.sh / launch_v53_46.sh（git 树）
REPO_DIR=<工作树>/lf_glm53_qlora_mtp   # V5.2=旧树(在跑)；V5.3 起=git 树
LF_SRC=/mnt/share_space/l00840580/LlamaFactory/src
WEIGHT_PATH=/mnt/weight2/GLM-5.3-W4A8
GLM53_RESUME_CKPT=<checkpoint 目录>     # 权重注入（copy_ 到新建模型）
# 可选：yaml 里加 resume_from_checkpoint=<同一目录> => 全量 resume
# （权重+optimizer_state_rank*.pt+scheduler+数据跳步；trainer.py 会用 yaml 的
#   save_steps 覆盖 checkpoint 内旧值——改 CP 节奏就是这么做的）
GLM53_MTP_CONTRACT_JOINT=serving_chain|serving_c
GLM53_CHAIN_SELF_FEED=0|1               # 自喂主开关（两契约都已实现）
GLM53_SELF_FEED_START / GLM53_SELF_FEED_ANNEAL  # ramp（ANNEAL=0 => 二值）
GLM53_MTP_DISTILL_WEIGHT=0|2.33         # KL 权重（2.33 ≈ 0.3CE+0.7KL）
setsid nohup bash <launch>.sh > <独立日志> 2>&1 &
```

- 启动后验证：`RESUME weights ... applied=127 missing=0`（**missing≠0 = 环境缺
  GLM53_MTP_LORA=1 等结构 env，查！**）；trainer_log 从注入点续步
- **已知偏差**：joint 路径的 lr 由模型内置 `mtp_lr_float=1e-5`/`mtp_lr_lora=1e-4`
  决定，**yaml 的 learning_rate 被忽略**（V5.2 实际满 lr 1e-5 续训，已知情）
- 步速 ~12min/步；300 步 ≈ 2.5 天

## 5. CP → 评测程序（.35）

```bash
# 1) 合并：vendor + checkpoint -> serving 包（打印逐张量 delta 审计块）
python3 scripts/build_trained_cp.py <ckpt_dir> /mnt/weight2/GLM-5.3-W4A8-<arm名>
#    env: WEIGHT_PATH + GLM53_LORA_RANK=16 ALPHA=32
# 2) 评测（脚本内置：HBM guard（防 health 假阳性）、每轮 IPC 清理、smoke、48p bench）
bash h0-test/eval35/run_vllm_ab_v5_35.sh <arm> <serving包> 8459 4,5,6,7,8,9,10,11 [prompts]
#    200p 用 data/lf_data/prompts200_tokenized.jsonl（tokenize_prompts200.py 生成）
# 3) 产物：h0-test/eval35/vllm_accept_<arm>.json —— EAL = tok/step + 1
```

- **容器劣化症状**：worker 静默死 + health 200 假阳性 = 泄漏信号量/孤儿进程堆积
  （2026-11-16 实录：12152 组信号量 + 5795 孤儿）→ 处置 = `docker restart` 容器
- 卡 0 需用户批准；默认 4-11

## 6. 探针程序（诊断用，1 卡 ~1h）

- `scripts/probe_v5_ladder.py`：三契约分深度 CE 阶梯（interim/exact/direct）
- `scripts/probe_v5_agree.py`：draft-vs-verifier argmax 一致率 + CE 阶梯，
  vendor/ckpt 两相位一次跑完；**lm_head 必须走 `_glm53_lm_pin` 流式**
  （`model.lm_head.weight` 默认不填充，直接调用得到随机权重 = 0.0% 事故根因）
- 一致率对齐口径：draft row j → verifier row c0+k+j（serving_chain）/
  k+j-1（contract_c dk）——已双对齐验证（m1 正确，alt≈噪声）

## 7. 运维红线（每条都是真实事故）

1. torch_npu 版本检查先于一切（post7 段错误）
2. 杀训练必须含 `launcher.py` 模式 + 强制校验杀净（worker 会从 torchrun 死亡中幸存）
3. 杀完等 60s 驱动沉淀（ERR99999 = 没等）
4. 容器劣化三联征（worker 静默死/health 假阳性/shm 广播等待）→ 重启容器
5. `npu-smi -t usagemem` 不存在——HBM 用主表解析（`grep 65536`，格式 `NNNNN/ 65536` 无空格）
6. 环境泄漏：launch 脚本 unset XDG_*；首启带 PYTHONFAULTHANDLER=1
7. 改模型代码后：HF 动态模块缓存会发旧版本——删缓存或核对 import 来源
8. 后台 watcher 的等待窗必须覆盖实际 ETA（超时自杀两次实录）；pkill 禁止自匹配
9. `npu-smi`/ps 在高负载下可能 >30s——监控命令带足 timeout
10. 日志/tqdm 的 \r 会破坏 grep 锚点——先 `tr '\r' '\n'`

## 8. 验收指标与决策门

- **V5.2（在跑）**：CP50-300 每 10 步一个；对比锚 = V5 探针 CP150（2.033）；
  看 ladder 深位（d5）能否压过冲击并持续下行
- **V5.3（发射后）**：首要验收 = **exact-d1 一致率 64.1% → ≈78.6%**；
  次要 = 200p 域缺口收窄；CP30/60/90/150 双题集评测
- **V5.4 候选**（单变量，逐个来）：深度权重/margin 矩阵（30 步 probe）、
  域数据配比、吞吐优化（trace_summary.md 已有 kernel 级数据：步速 12min→9min
  可让所有 A/B 提速 25%）——**详细设计已冻结：`V54_PLAN.md`（一致率蒸馏路线，
  2026-11-17 成文，待 V5.3 CP90/105 判决后执行）**

## 9. 当前快照（交接点）

- V5.2 r2：.46 16 卡，step ~53/300，**10 步/CP**（CP60 ~01:00 UTC），自 CP50 全量续跑，
  日志 `h0-test/eval35/v52_run_r2_46.log`（旧树）
- .35：空闲、链路就绪（build/serve/bench 一键）
- git：`/mnt/share_space/l00840580/mtp_git`，master 已推送（本手册即其中文件）
- 已中止项：CP50→25 步切换（用户改为 10 步直启）、.35 v52cp50 链（用户取消）
