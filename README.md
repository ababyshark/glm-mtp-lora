# glm-mtp-lora

GLM-5.3-Flash（MoE, 45 层, KDA/DSA 混合注意力, 288 专家）的 **MTP draft 微调**
（主模型 LoRA + MTP 子树），目标：vLLM 投机解码（deepseek_mtp, K=5）的原生前缀
接受长度 **K̄** 显著超过 base 的 3.521。北极星指标 = vLLM serving 的 /metrics
K̄ 计数器（382 用例三组），所有有效性结论以 K̄ 为准。

当前入库版本：**V10.2**（KL+agree+margin+LR 分组全配方）与 **V10.3**
（冻结面 + 全词表 KL 单变量臂）。

## 布局

```
docs/            V10.2 方案文档 ×13（设计/训练/铁律/机理/导出/蒸馏/评测）
scripts/v10/     实现脚本 + tests/ 单元测试（自包含，无外部 v8 依赖）
scripts/v10/v10_data.py / v10_ce_margin.py   V8 线 verbatim 内联（来源见文件头）
scripts/v10/reference/mcore020/              DSA 移植底本（0.18.2 无此模块）
configs/         训练 yaml（glm53_v10_train.yaml — LF 数据键 + 模型路径）
deploy/          训练 launcher（.34/.40 工作站，16×910B TP16）
deps.json        依赖基线（软件版本 + env 契约 + 证据来源）——环境以此为准
```

## 环境（详见 deps.json）

- Python 3.12 + torch 2.10.0+cpu + torch_npu 2.10.0.post4 + CANN 9.1.0（16×Ascend910B）
- **Megatron-Core 0.18.2**：源码树注入 `PYTHONPATH=/root/mcore0182-src`（非 pip）
- **LlamaFactory**：git 树注入 `LF_SRC`（默认 `/mnt/share_space/l00840580/LlamaFactory/src`），
  必须 `DISABLE_VERSION_CHECK=1`
- vendor BF16 权重目录：`V10_VENDOR_DIR`
- serving/评测：vllm-ascend 0.30（BF16）

## 单元测试

```bash
python3 -m pytest scripts/v10/tests -q
```

- 纯 CPU 子集（agree_margin/diag/group_eval/mtp/recipe/tpfill/v103）任何机器可跑
- mcore 子集（build/fill/lora/units/grouped_experts 等）需 `PYTHONPATH=/root/mcore0182-src`，
  训练机上全绿
- 已知限制：**未装 megatron 的机器**上全量跑会有 mcore 子集 ERROR + 3 例
  test_v10_kl / 2 例 test_v10_v103 失败（`_kl_chunk` 在 dist 已被其他测试初始化时
  走 megatron 路径）——装了 megatron 的环境不受影响

## 训练

```bash
bash deploy/launch_v102_train_34.sh   # V10.2 配方（.34，1000 步，CP@100）
bash deploy/launch_v102_train_40.sh   # V10.2 配方（.40，MTP LR 1e-5 / draft-CE 0 臂）
bash deploy/launch_v103_train_34.sh   # V10.3 单变量臂（100 步：冻结面+全词表 KL）
bash deploy/launch_v103_train_40.sh   # V10.3 .40 臂（从零：全新 BF16 base，须先 cp -a 暂存到 /tmp/v103_base_bf16）
```

- launcher 自推导仓库根（`GIT_TREE`），venv/PYTHONPATH/vendor 路径可用 env 覆盖
  （`PY` / `PYTHONPATH` / `V10_VENDOR_DIR` / `V10_OUTPUT_DIR`）
- 配方全部经 `GLM53_*` env 旋钮注入（逐项含义见 deps.json `key_env`），
  两代臂的数据/seed/order 严格一致（单变量对照）

## 导出 / 评测

- CP（mcore 分片）→ vLLM serving 目录：`scripts/v10/v10_export_serving.py`
  （V10.3 起：CP 中缺失键 = 冻结 = 保训练基值）
- Step0 健康锚点：`scripts/v10/eval_step0_baseline.py`（EVAL-FIXED 趋势基线）
- serving 评测工具链（单臂 serve/watcher/bisect/382 三组）与 on-policy 蒸馏管线
  按方案文档 `docs/V10_EXPORT_SERVING.md` / `V10_ONPOLICY_DISTILL.md` 执行，
  脚本在 `scripts/v10/`（`eval_groups_40.py` / `eval_d15_40.py` / `ce_gate.py` 等）

## 版本脉络

| 版本 | 配方要点 | 结果 |
|---|---|---|
| V10 | 纯 corpus-CE | serving K̄ 3.52 → 2.49（−29%，负向） |
| V10.1 | + KL 2.33 + agree 0.5 + margin 0.3 | 三组全负 → bisect 定位毒源 = MTP 窄域微调本身 |
| **V10.2** | draft-CE 0/0.25 + top-128 KL + LR 分组 + cosine | 入库基线 |
| **V10.3** | mtp router+norms 冻结 + 全词表 KL(TOPK=0) + KL 0.8 + agree 0.2 + margin/draft-CE 关 | 单变量臂（目标 K̄ 3.15–3.28） |

设计依据：`docs/V10_DESIGN.md`、`docs/V10_TRAIN.md`（§11 机理定案）、
`docs/V10.2_CONSULT.md`（专家会诊 + 配置勘误）；V10.3 设计落在
`v10_trainer._mtp_frozen` / `_kl_reduce` 的 docstring 与 `tests/test_v10_v103.py`。
