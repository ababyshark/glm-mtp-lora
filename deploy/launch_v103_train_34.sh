#!/bin/bash
# V10.3 Arm A training on .34 (A3-syn-24, same stack as V10.2) — SINGLE VARIABLE
# vs the V10.2 run: the recipe (loss/weights/KL-impl/LR/clip/freeze surface).
# Same yaml, same vendor base (/root/serve_v101_cp100), same 30k data, same
# seed/order. 100 steps → cp100 → export → 382-serving K̄ (expect: stops the
# bleeding, K̄ ≥ V10.1's 2.34, target 3.15-3.28 per the V10.3 plan).
#
# Recipe deltas vs V10.2 (all via existing env knobs — no new loss code):
#   KL       : top-128 → FULL-vocab chunked_vocab_kl (TOPK_KL=0), weight 2.33 → 0.8
#   agree-CE : 0.5 → 0.2 (va anchor, unchanged impl)
#   margin   : 0.3 → 0 (off)
#   draft-CE : 0.25 → 0 (off)
#   MTP LR   : 2e-5 → 3e-6, warmup 20 → 5 steps, clip 5.0 → 1.0, cosine → 5e-7
#   freeze   : mtp router + all RMSNorm frozen (V10.3 surface, trainer defaults;
#              eh_proj stays trainable) → CP shrinks ~599 → ~589 keys/rank
# (repo port: GIT_TREE derived from script location; yaml in-repo; logs under
#  $GIT_TREE/logs/. FIX vs source repo: the freeze env names were written
#  GLM53_V10.3_* (dotted, unmatched by the trainer) — corrected to
#  GLM53_V10_FREEZE_* which is what v10_trainer._mtp_frozen reads.)
set -u
export DISABLE_VERSION_CHECK=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export GLM53_V10_CLIP=1.0
export GLM53_V10_WARMUP=5
export GLM53_V10_RECOMPUTE=full
export GLM53_MTP_DISTILL_WEIGHT=0.8
export GLM53_V10_TOPK_KL=0
export GLM53_V10_AGREE_W=0.2
export GLM53_V10_MARGIN_W=0
export GLM53_V10_MARGIN_M=0.5
export GLM53_V10_DRAFT_W=0
export GLM53_V10_MTP_LR=3e-6
export GLM53_V10_COSINE=1
export GLM53_V10_MIN_LR=5e-7
# V10.3 freeze surface — explicit defaults (router + norms on, eh_proj trainable)
export GLM53_V10_FREEZE_MTP_ROUTER=1
export GLM53_V10_FREEZE_MTP_NORMS=1
export GLM53_V10_FREEZE_EH_PROJ=0
export V10_VENDOR_DIR=${V10_VENDOR_DIR:-/root/serve_v101_cp100}
export HF_HOME=${HF_HOME:-/root/hf_cache_v103}
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TMPDIR=/root/tmp_v103
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
GIT_TREE="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=${PYTHONPATH:-/root/mcore0182-src}
PY=${PY:-/root/venv_v8/bin/python}
mkdir -p "$GIT_TREE/logs"
cd "$GIT_TREE"
exec $PY -m torch.distributed.run --nproc_per_node=16 --master_port=29790 \
  scripts/v10/v10_trainer.py \
  --yaml configs/glm53_v10_train.yaml \
  --max-steps 100 --cp-interval 100 --tp-size 16 --batch-size 1 \
  --output-dir ${V10_OUTPUT_DIR:-/root/saves_v103} \
  > "$GIT_TREE/logs/v103_train_34.log" 2>&1
