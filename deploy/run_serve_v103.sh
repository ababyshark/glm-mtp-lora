#!/bin/bash
# V10.3 serving-arm runner: vllm-ascend serve + 382 three-group bench (K̄).
# Adapted from run_serve_cp100_34.sh (the .34-arm precedent) — repo-ported:
#   PY/VLLM paths overridable (venv_v8 container vs bare 3.12.13 hosts)
#   prompts stay on the source NFS tree (eval data intentionally NOT vendored
#   into this repo — deps.json §eval records md5 provenance instead)
# usage: run_serve_v103.sh <MODEL_DIR> [TAG] [PORT]
#   MODEL_DIR = exported serve dir (v10_export_serving.py output)
#   TAG       = ab_results tag prefix (default v103cp100)
set -u
MODEL="${1:?usage: run_serve_v103.sh <MODEL_DIR> [TAG] [PORT]}"
TAG="${2:-v103cp100}"
PORT="${3:-8460}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VLLM="${VLLM_BIN:-/usr/local/python3.12.16/bin/vllm}"
PY="${PY_BIN:-/root/venv_v8/bin/python}"
PROMPTS_DIR="${V10_BENCH_PROMPTS_DIR:-/mnt/share_space/l00840580/mtp_git/lf_glm53_qlora_mtp/reports/v10_eval382}"
LOGD="${V10_GEVAL_LOGD:-$REPO/reports/serve_logs}"
export DISABLE_VERSION_CHECK=1 PYTHONUNBUFFERED=1 TASK_QUEUE_ENABLE=1
export TORCHDYNAMO_DISABLE=1
export HF_HOME="${HF_HOME:-/tmp/hf_cache}"
unset TMPDIR
mkdir -p "$HF_HOME" "$LOGD"

SLOG="$LOGD/vllm_serve_${TAG}.log"
echo "[$(date -u +%H:%M:%S)] ${TAG} serve start ($MODEL)" >> "$SLOG"

setsid nohup "$VLLM" serve "$MODEL" \
  --served-model-name glm53 --port "$PORT" \
  --tensor-parallel-size 16 --max-model-len 4096 \
  --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --gpu-memory-utilization 0.92 --enforce-eager \
  --speculative-config '{"method":"deepseek_mtp","num_speculative_tokens":5,"enforce_eager":true}' \
  --trust-remote-code --seed 1024 >> "$SLOG" 2>&1 < /dev/null &

teardown() {
  pkill -TERM -f "vllm serve.*$PORT" 2>/dev/null; sleep 5
  pkill -KILL -f "vllm serve.*$PORT" 2>/dev/null
  pkill -KILL -f "VLLM::Worker" 2>/dev/null; sleep 2
}

for i in $(seq 1 160); do
  curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  sleep 10
done
curl -s -m 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
  || { echo "SERVER NOT READY" >> "$SLOG"; teardown; exit 3; }
echo "[$(date -u +%H:%M:%S)] server up" >> "$SLOG"

for g in group1_unseen group2_cp250 group3_late; do
  V10_BENCH_PROMPTS="$PROMPTS_DIR/prompts_$g.jsonl" \
    "$PY" "$REPO/scripts/bench_acceptance.py" \
    --tag "${TAG}_${g}" --port "$PORT" \
    > "$LOGD/bench_${TAG}_${g}.log" 2>&1 \
    || { echo "BENCH FAILED $g" >> "$SLOG"; teardown; exit 4; }
  echo "[$(date -u +%H:%M:%S)] bench $g done" >> "$SLOG"
done
teardown
echo "[$(date -u +%H:%M:%S)] ${TAG} COMPLETE" >> "$SLOG"
