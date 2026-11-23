# 依赖树补丁（V10.2/V10.3 训练栈）

V10.2/V10.3 的运行栈 = 两个**非 pip 依赖树**（Megatron-Core 源码树 + LlamaFactory
源码树）。除了本目录的补丁文件外，V10 对 mcore 的另外两处修复
（MTP 输入 no-roll、MTP checkpoint padding_mask 签名）是**代码内运行时
monkeypatch**（`scripts/v10/v10_mtp.py`），不依赖树补丁——见
`docs/V10_OVERVIEW.md`「mcore patch」行。

## 1. mc0182_dpa_k_channels.patch → Megatron-Core 0.18.2 源码树

- **来源**：V8 线 `scripts/v8/mc0182_dpa_k_channels.patch` verbatim（md5
  d6a750e47fbc84b2c73752097e39614e）；目标树 = 训练机 `/root/mcore0182-src`
  （`PYTHONPATH` 注入，版本 0.18.2 commit 571370c82）
- **内容**（3 处）：
  1. `transformer/dot_product_attention.py`：`__init__` 接受 `k_channels` /
     `v_channels` / `**kwargs`（MLA 传入的额外参数，本地 DPA 忽略其值）
  2. `pipeline_parallel/schedules.py` ×3：`device="cuda"` →
     `torch.npu.current_device()`（**昇腾必需**——schedules 里 total_num_tokens
     硬编码 cuda 设备）
- **应用**（在 mcore 源码树根）：
  ```bash
  cd /root/mcore0182-src && patch -p1 --dry-run < $REPO/patch/mc0182_dpa_k_channels.patch
  cd /root/mcore0182-src && patch -p1 < $REPO/patch/mc0182_dpa_k_channels.patch
  ```
- **验证**：dry-run 无 reject；训练冒烟步日志无 `invalid device` cuda 报错

## 2. llamafactory_4759a8de_to_83788da4.patch → LlamaFactory 源码树

- **来源**：V4 warpack `patch/llamafactory_4759a8de_to_83788da4.patch` verbatim
  （md5 9f9629a50bfafe0235ff62c5895decca）
- **基线**：upstream LlamaFactory @ `4759a8de`；应用后 = 本地线
  `glm53-w4a8-mtp-v0.1-7-g83788da4`（共享空间 `/mnt/share_space/l00840580/LlamaFactory`
  的工作副本即此状态；该目录不是 git 仓，本补丁是它的复现依据）
- **内容**（V10 用到的面）：`src/llamafactory/model/loader.py` 等——trl≥1.0
  兼容（`AutoModelForCausalLMWithValueHead` try/except，SFT 不用 valuehead）、
  glm53 模型/模板注册面。V10.2 数据管线（`v10_data.load_lf_data`）只用
  tokenizer + dataset/template 机制，加载此版本保证 `glm5_next_text` 模板可用
- **约束**：必须配 `DISABLE_VERSION_CHECK=1`（LF `misc.py` 钉 transformers<=5.8）
- **应用**（在 LF 树根，src/ 布局）：
  ```bash
  cd $LF_TREE && patch -p1 --dry-run < $REPO/patch/llamafactory_4759a8de_to_83788da4.patch
  cd $LF_TREE && patch -p1 < $REPO/patch/llamafactory_4759a8de_to_83788da4.patch
  ```

## 已知边界

- 训练机 `/root/mcore0182-src` 上除本补丁外可能还有 V8 时代的其他本地改动，
  未逐一审计入册；如遇 0.18.2 行为偏差，先 diff 工作站树与 upstream 571370c82
- V4 线的历史补丁（`v4_commit_*.patch`、`llamafactory-mtp-ft.patch`）不属
  V10.2/V10.3 依赖面，留在源仓 `reports/`/`warpack/`，未入库
