#!/usr/bin/env bash
set -euo pipefail

# Full CHRONOA-EDGE GPU experiment launcher.
#
# Usage:
#   bash Chronoa/scripts/experiments/run_chronoa_edge_gpu_experiments.sh
#
# Common overrides:
#   GPU_ID=1 OUTPUT_ROOT=chronoa_edge_runs EPOCHS=120 BATCH_SIZE=24 \
#     bash Chronoa/scripts/experiments/run_chronoa_edge_gpu_experiments.sh
#
# Smoke test:
#   SMOKE=1 bash Chronoa/scripts/experiments/run_chronoa_edge_gpu_experiments.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_ROOT}"

PYTHON="${PYTHON:-python3}"
GPU_ID="${GPU_ID:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-chronoa_edge_experiments}"
SUITES="${SUITES:-standard,architecture,temporal,window}"
DATASETS="${DATASETS:-all}"

EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-8e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"
D_MODEL="${D_MODEL:-96}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.10}"
DELTA_KERNELS="${DELTA_KERNELS:-4}"
MAX_EVENTS="${MAX_EVENTS:-1536}"
MAX_QUERIES="${MAX_QUERIES:-512}"
SEED="${SEED:-1729}"
PROGRESS_EVERY="${PROGRESS_EVERY:-100}"
TRACE_EVERY="${TRACE_EVERY:-0}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PYTHONPATH:-}"

"${PYTHON}" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available. Check GPU drivers, CUDA_VISIBLE_DEVICES, or your torch install.")

print(f"CUDA OK: {torch.cuda.get_device_name(0)}")
PY

LIMIT_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  LIMIT_ARGS+=(
    --max-train-samples "${MAX_TRAIN_SAMPLES:-128}"
    --max-validation-samples "${MAX_VALIDATION_SAMPLES:-64}"
    --max-test-samples "${MAX_TEST_SAMPLES:-64}"
    --epochs "${SMOKE_EPOCHS:-2}"
    --patience "${SMOKE_PATIENCE:-1}"
  )
else
  if [[ -n "${MAX_TRAIN_SAMPLES:-}" ]]; then
    LIMIT_ARGS+=(--max-train-samples "${MAX_TRAIN_SAMPLES}")
  fi
  if [[ -n "${MAX_VALIDATION_SAMPLES:-}" ]]; then
    LIMIT_ARGS+=(--max-validation-samples "${MAX_VALIDATION_SAMPLES}")
  fi
  if [[ -n "${MAX_TEST_SAMPLES:-}" ]]; then
    LIMIT_ARGS+=(--max-test-samples "${MAX_TEST_SAMPLES}")
  fi
  LIMIT_ARGS+=(--epochs "${EPOCHS}" --patience "${PATIENCE}")
fi

OVERWRITE_ARGS=()
if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARGS+=(--overwrite)
fi

echo "Running CHRONOA-EDGE suites: ${SUITES}"
echo "Datasets: ${DATASETS}"
echo "GPU: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Output root: ${OUTPUT_ROOT}"

"${PYTHON}" scripts/experiments/run_chronoa_experiment_suite.py \
  --suite "${SUITES}" \
  --dataset "${DATASETS}" \
  --output-root "${OUTPUT_ROOT}" \
  --device cuda \
  --random-seed "${SEED}" \
  --batch-size "${BATCH_SIZE}" \
  --learning-rate "${LR}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --d-model "${D_MODEL}" \
  --layers "${LAYERS}" \
  --heads "${HEADS}" \
  --dropout "${DROPOUT}" \
  --delta-kernels "${DELTA_KERNELS}" \
  --max-events "${MAX_EVENTS}" \
  --max-queries-per-sample "${MAX_QUERIES}" \
  --progress-every "${PROGRESS_EVERY}" \
  --trace-every "${TRACE_EVERY}" \
  --build-tables \
  "${LIMIT_ARGS[@]}" \
  "${OVERWRITE_ARGS[@]}"

echo "Done. Summaries are under ${OUTPUT_ROOT}/"
