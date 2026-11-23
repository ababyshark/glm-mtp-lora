#!/bin/bash
# V10.1 training: V10 + KL(2.33) + agree(0.5) + margin(0.3, m=0.5) — the V8
# recipe ingredients restored after the serving verdict (§11: the corpus-CE-
# only run degraded serving K̄ 3.52→2.49). Same data/seed/order as the
# original 1000-step run (perfect A/B: only the loss differs).
# CP every 100 steps. 16 cards, .40 mtpV8.
# NOTE: despite the historical name, the recipe knobs below were reused by the
# V10.2 arm on .40 (draft-CE=0.0 / TOPK_KL=128 / MTP LR 1e-5 block appended
# 2026-11-23) — kept verbatim from the source repo's launch_v10_train_v102_40.sh.
# (repo port: GIT_TREE derived from script location; yaml in-repo; logs under
#  $GIT_TREE/logs/)
set -u
export DISABLE_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export GLM53_V10_CLIP=5.0
export GLM53_V10_WARMUP=20
export GLM53_V10_RECOMPUTE=full
# V10.1 loss (the V8 frozen verdict weights)
export GLM53_MTP_DISTILL_WEIGHT=2.33
export GLM53_V10_AGREE_W=0.5
export GLM53_V10_MARGIN_W=0.3
export GLM53_V10_MARGIN_M=0.5
# V10.2 expert patch (2026-11-23): gold-CE=0, Top-128 KL, MTP LR 1e-5 + cosine
export GLM53_V10_DRAFT_W=0.0
export GLM53_V10_TOPK_KL=128
export GLM53_V10_MTP_LR=1e-5
export GLM53_V10_COSINE=1
export GLM53_V10_MIN_LR=1e-6
export HF_HOME=${HF_HOME:-/mnt/share_space/l00840580/mtp_git/.hf_cache_v10}
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TMPDIR=$HF_HOME/tmp
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
GIT_TREE="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=${PYTHONPATH:-/root/mcore0182-src}
PY=${PY:-/root/venv_v8/bin/python}
mkdir -p "$GIT_TREE/logs"
cd "$GIT_TREE"
exec $PY -m torch.distributed.run --nproc_per_node=16 --master_port=29778 \
  scripts/v10/v10_trainer.py \
  --yaml configs/glm53_v10_train.yaml \
  --max-steps 1000 --cp-interval 100 --tp-size 16 --batch-size 1 \
  --output-dir ${V10_OUTPUT_DIR:-$GIT_TREE/saves/v10_t4_v102} \
  > "$GIT_TREE/logs/v10_train_v102_40.log" 2>&1
