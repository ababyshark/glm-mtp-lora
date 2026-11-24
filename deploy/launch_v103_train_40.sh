#!/bin/bash
# V10.3 training on .40 (A3-syn-30, container l00840580-v11, 16 NPU) — FROM
# SCRATCH on the fresh ZhipuAI BF16 base (NOT the W8A8-dequant v11 base).
#   base     = /tmp/v103_base_bf16 — LOCAL stage of
#              /mnt/share_space/l00840580/mtp_git/GLM-5.3-Flash-BF16
#              (120 shards, MTP = layers.45, raw model.language_model.* naming,
#              v10_fill contract: 38770 tensors, 347 visual FLOAT keys).
#              PRECONDITION: cp -a <NFS BF16>/. /tmp/v103_base_bf16/ — fill
#              does mmap page-fault reads; over NFS all-16-rank random reads
#              stalled 20+ min (2026-11-24), local NVMe is the V11-proven path.
#   recipe   = V10.3 deltas vs V10.2 (launch_v103_train_34.sh VERBATIM):
#     KL       : top-128 → FULL-vocab chunked_vocab_kl (TOPK_KL=0), weight 0.8
#     agree-CE : 0.2 (va anchor), margin off, draft-CE off
#     MTP LR   : 3e-6, warmup 5, clip 1.0, cosine → 5e-7
#     freeze   : mtp router + all RMSNorm frozen (eh_proj trainable)
#     diag     : DEPTH_DIAG=1 (default) → per-step D1..D5 dacc proxies
#   100 steps → cp100 → export → 382-serving K̄ (target 3.15-3.28 per plan).
# Stack = the V11-proven one (mtpv8-shm: torch_npu 2.10.0.post7, mcore
# 0.18.2 @ /root/mcore0182-src, vllm-ascend 0.30.0).
# (repo port: GIT_TREE derived from script location; yaml in-repo
#  configs/glm53_v103_train_40.yaml; logs under $GIT_TREE/logs/;
#  V10_VENDOR_DIR / HF_HOME / V10_OUTPUT_DIR / PY / PYTHONPATH env-overridable.)
set -u
if [ ! -d "${V10_VENDOR_DIR:-/tmp/v103_base_bf16}" ]; then
  echo "[launch_v103_train_40] staged base missing: ${V10_VENDOR_DIR:-/tmp/v103_base_bf16}
  PRECONDITION: cp -a /mnt/share_space/l00840580/mtp_git/GLM-5.3-Flash-BF16/. /tmp/v103_base_bf16/" >&2
  exit 1
fi
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
# From-scratch vendor = the staged local BF16 copy (fill + tokenizer +
# config.json via v10_config.VENDOR_CFG)
export V10_VENDOR_DIR=${V10_VENDOR_DIR:-/tmp/v103_base_bf16}
export HF_HOME=${HF_HOME:-/tmp/v103_hf_cache}
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TMPDIR=${TMPDIR:-/tmp/v103_tmp}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"
GIT_TREE="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH=${PYTHONPATH:-/root/mcore0182-src}
PY=${PY:-/root/venv_v8/bin/python}
mkdir -p "$GIT_TREE/logs"
cd "$GIT_TREE"
exec $PY -m torch.distributed.run --nproc_per_node=16 --master_port=29792 \
  scripts/v10/v10_trainer.py \
  --yaml configs/glm53_v103_train_40.yaml \
  --max-steps 100 --cp-interval 100 --tp-size 16 --batch-size 1 \
  --output-dir ${V10_OUTPUT_DIR:-/tmp/v103_saves} \
  > "$GIT_TREE/logs/v103_train_40.log" 2>&1
