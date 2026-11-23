# HowTo — 从 clone 到训练 / serving 验收

> 环境权威 = `deps.json`；仓库布局 = `README.md`；本文 = 操作顺序手册。
> 栈：dspark 系容器（aarch64）+ CANN 9.1.0 + 16×Ascend910B 单机 TP16。

---

## 1. 安装

### 1.1 容器与 NPU 栈
```bash
# dspark 系容器内，CANN 运行时（toolkit + nnal/atb）
source /mnt/share_space/l00840580/env.sh          # cann-9.1.0

python3 -m venv /root/venv_v10 && source /root/venv_v10/bin/activate
pip install torch==2.10.0+cpu torch_npu==2.10.0.post4 \
    transformers==5.14.1 safetensors numpy pyyaml pytest
```
版本以此处为准（出处见 `deps.json` §version_baseline_provenance）。

### 1.2 Megatron-Core 0.18.2 源码树（非 pip）
```bash
# upstream Megatron-LM，checkout 到 0.18.2 线的 571370c82
git clone https://github.com/NVIDIA/Megatron-LM.git /root/mcore0182-src
cd /root/mcore0182-src && git checkout 571370c82
export PYTHONPATH=/root/mcore0182-src            # v10 全部脚本经此注入
```

### 1.3 LlamaFactory 源码树（非 pip）
```bash
git clone https://github.com/hiyouga/LlamaFactory.git $LF_TREE
cd $LF_TREE && git checkout 4759a8de             # 上游基线，先打补丁（§2）
export LF_SRC=$LF_TREE/src                       # v10_data.py 运行时注入
export DISABLE_VERSION_CHECK=1                   # 必须（LF 钉 transformers<=5.8）
```

### 1.4 serving 机
vllm-ascend 0.30（BF16 无量化）——仅评测/rollout 需要，训练机可不装。

---

## 2. 打补丁

两棵源码树各有一个补丁，详见 **`patch/README.md`**（内容/来源/md5/已知边界）：

```bash
# mcore：MLA DPA 签名(k_channels/v_channels) + schedules×3 cuda→NPU（昇腾必需）
cd /root/mcore0182-src && patch -p1 --dry-run < $REPO/patch/mc0182_dpa_k_channels.patch
cd /root/mcore0182-src && patch -p1 <           $REPO/patch/mc0182_dpa_k_channels.patch

# LlamaFactory：trl>=1.0 兼容 + glm53 模型/模板注册面（glm5_next_text 模板来源）
cd $LF_TREE && patch -p1 --dry-run < $REPO/patch/llamafactory_4759a8de_to_83788da4.patch
cd $LF_TREE && patch -p1 <           $REPO/patch/llamafactory_4759a8de_to_83788da4.patch
```

注意：V10 对 mcore 的另外两处修复（MTP 输入 no-roll、MTP checkpoint
padding_mask 签名）是**代码内运行时 monkeypatch**（`scripts/v10/v10_mtp.py`），
无需树补丁。

---

## 3. 数据准备

训练语料本体不入库（`.gitignore`），md5/路径/分布记录在 `deps.json` §data_and_eval。

### 3.1 训练集（cmb_eagle_30k）
- 30000 行 sharegpt 格式 jsonl（数学应用题 + 通用 QA，中 34% / 英 66%）
- 权威副本：`/mnt/share_space/l00840580/mtp_git/lf_glm53_qlora_mtp/data/lf_data/cmb_eagle_30k.jsonl`
- 数据目录约定：`--yaml` 的 `dataset_dir` 指向的目录必须同时含
  语料 jsonl 和注册文件 `dataset_info.json`：

```json
{
  "cmb_eagle_30k": {
    "file_name": "cmb_eagle_30k.jsonl",
    "formatting": "sharegpt",
    "columns": {"messages": "conversations"},
    "tags": {"role_tag": "from", "content_tag": "value",
             "user_tag": "human", "assistant_tag": "assistant"}
  }
}
```

### 3.2 换自己的数据集
① 同格式生成 `<name>.jsonl`；② `dataset_info.json` 加同名条目；
③ `configs/glm53_v10_train.yaml` 改 `dataset: <name>`。
`load_lf_data` 只读 `dataset / dataset_dir / template / cutoff_len /
preprocessing_num_workers` 五个键。

### 3.3 模型与评测数据
| 用途 | 位置 | 说明 |
|---|---|---|
| tokenizer | yaml `model_name_or_path`（默认 `/root/serve_v101_cp100`） | 只取 tokenizer 文件 |
| 训练权重（fill） | `V10_VENDOR_DIR` | GLM-5.3-Flash-BF16 vendor 目录 |
| 382 评测 prompts | 源仓 `reports/v10_eval382/`（三组 197/98/87） | md5 见 deps.json，组间零重叠 |

---

## 4. 自检（训练前）

```bash
python3 -m py_compile scripts/v10/*.py                 # 语法
python3 -m pytest scripts/v10/tests -q                 # CPU 子集应全绿
```
已知限制：**未装 megatron 的机器**全量跑会有 mcore 子集 ERROR + 5 例
失败（dist 泄漏后 `_kl_chunk` 走 megatron 路径）——训练机上全绿，判据见 README。

---

## 5. 启动训练

```bash
bash deploy/launch_v102_train_34.sh   # V10.2 配方，1000 步，CP@100
bash deploy/launch_v102_train_40.sh   # V10.2 .40 臂（MTP LR 1e-5 / draft-CE 0）
bash deploy/launch_v103_train_34.sh   # V10.3 单变量臂，100 步（冻结面+全词表 KL）
```

- launcher 自推导仓库根；可覆盖 env：`PY`（venv python）、`PYTHONPATH`
  （mcore 树）、`V10_VENDOR_DIR`、`V10_OUTPUT_DIR`
- 配方全部经 `GLM53_*` env 注入（逐项见 `deps.json` §key_env）；
  对照臂的数据/seed/order 严格一致（单变量纪律）
- 日志：`logs/<launcher名>.log`；checkpoint：`<output-dir>/cp{N}_rank{rank}.pt`

### 5.1 运行中监控
```bash
python3 scripts/v10/check_train_log.py logs/v102_train_34.log          # 信号分级 CRITICAL/WARN/INFO
python3 scripts/v10/check_train_log.py logs/v102_train_34.log --json   # 机器可读
```
启动后先跑一次 Step0 锚点，此后每 10 步的 EVAL-FIXED 行看**趋势**：
```bash
torchrun --nproc_per_node=16 scripts/v10/eval_step0_baseline.py \
    --yaml configs/glm53_v10_train.yaml        # 见该文件 docstring
```

---

## 6. 导出 serving 目录

```bash
python3 scripts/v10/v10_export_serving.py <rank_shard_dir> <out_dir>
# env: V10_TP=16（默认）、V10_VENDOR_DIR、GLM53_LORA_RANK/ALPHA（16/32）
```

- 原理 = fill 的逆运算：vendor 拷贝 + LoRA 合并 + MTP 子树替换；
  每张量写前 assert 形状 == vendor 目标
- **out_dir 必须放 NFS**（容器 rootfs 会 100% 满，事故陷阱）；
  未修改文件保持 symlink 进 vendor（symlink farm）
- V10.3 起：CP 中缺失键 = 冻结键 = 保训练基值（export 自动跳过并打印清单）

---

## 7. serving 验收（北极星 K̄）

```bash
bash deploy/run_serve_v103.sh    # vllm-ascend serve + 382 三组 bench
python3 scripts/bench_acceptance.py   # A/B 验收（baseline vs CP）
```

- 协议：TP16 / K=5 / seed 1024 / greedy（确定性接受率）
- prompts 目录 env：`V10_BENCH_PROMPTS_DIR`（默认指向源仓 v10_eval382）
- 判据锚点（deps.json §baseline_anchors）：base **K̄=3.521**；V10.1 2.34–2.43；
  V10.3 目标 **3.15–3.28**。所有"有效性"结论只认 serving K̄

---

## 8. 故障排查速查

| 症状 | 首查 |
|---|---|
| fill 报 vendor 键缺失 | `V10_VENDOR_DIR` 是否指向真 BF16 目录 |
| 训练步 mtp-grads=0 | MTP checkpoint 输入 detach 陷阱（已在 v10_trainer 修复，勿回退） |
| 训练无效但探针"改善" | 探针只测训练目标自身——以 serving K̄ 为准（V10_TRAIN §11.5） |
| 容器盘满 | rootfs 陷阱：serving/导出目录一律 NFS + symlink |
| 任何异常信号 | `check_train_log.py`（每条信号对应一次真实事故） |
