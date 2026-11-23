#!/bin/bash
# V10.2 training on .34 (A3-syn-24, bare container, 16 NPU) — the base =
# /root/serve_v101_cp100 (the cp100-patched weights; the serving comparison
# vs the cp100_34 bench results = the same-base clean A/B). The V10.2 loss:
# KL(2.33, top-128) + agree(0.5) + margin(0.3/m0.5, va-anchored) + draft-CE 0.25
# + LR groups (MTP 2e-5 / LoRA 5e-6) + cosine.
# (repo port: GIT_TREE derived from script location; yaml points at configs/
#  in-repo; logs under $GIT_TREE/logs/ — all recipe env VERBATIM from the
#  source repo's launch_v102_train_34.sh)
set -u
export DISABLE_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export GLM53_V10_CLIP=5.0
export GLM53_V10_WARMUP=20
export GLM53_V10_RECOMPUTE=full
export GLM53_MTP_DISTILL_WEIGHT=2.33
export GLM53_V10_AGREE_W=0.5
export GLM53_V10_MARGIN_W=0.3
export GLM53_V10_MARGIN_M=0.5
export GLM53_V10_DRAFT_W=0.25
export GLM53_V10_TOPK_KL=128
export GLM53_V10_MTP_LR=2e-5
export GLM53_V10_COSINE=1
export GLM53_V10_MIN_LR=1e-6
export V10_VENDOR_DIR=${V10_VENDOR_DIR:-/root/serve_v101_cp100}
export HF_HOME=${HF_HOME:-/root/hf_cache_v102}
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TMPDIR=/root/tmp_v102
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
GIT_TREE="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=${PYTHONPATH:-/root/mcore0182-src}
PY=${PY:-/root/venv_v8/bin/python}
mkdir -p "$GIT_TREE/logs"
cd "$GIT_TREE"
exec $PY -m torch.distributed.run --nproc_per_node=16 --master_port=29790 \
  scripts/v10/v10_trainer.py \
  --yaml configs/glm53_v10_train.yaml \
  --max-steps 1000 --cp-interval 100 --tp-size 16 --batch-size 1 \
  --output-dir ${V10_OUTPUT_DIR:-/root/saves_v102} \
  > "$GIT_TREE/logs/v102_train_34.log" 2>&1
