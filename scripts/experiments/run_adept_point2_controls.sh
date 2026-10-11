#!/usr/bin/env bash
set -euo pipefail

# Point-2 reviewer controls for ADEPT:
#   1) MSE-only objective ablation.
#   2) Default-loss gate/expert diagnostics.

ROOT="${CHRONOA_ROOT:-/home/nckh2/qa}"
cd "${ROOT}"

RUNNER="${RUNNER:-Chronoa/scripts/experiments/run_chronoa_forecasting.py}"
GATE_ANALYSIS="${GATE_ANALYSIS:-Chronoa/scripts/analysis/adept_gate_analysis.py}"
OBJECTIVE_ANALYSIS="${OBJECTIVE_ANALYSIS:-Chronoa/scripts/analysis/adept_objective_analysis.py}"
PYTHON_BIN="${PYTHON_BIN:-/home/nckh2/qa/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

BASE_ROOT="${OUTPUT_ROOT:-Chronoa/results/adept_point2_controls}"
LOG_DIR="${LOG_DIR:-${BASE_ROOT}/logs}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"

DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
RUN_MSE="${RUN_MSE:-1}"
RUN_GATE="${RUN_GATE:-1}"
MAX_GATE_LOG_ROWS="${MAX_GATE_LOG_ROWS:-}"

LEARNING_RATE="${LEARNING_RATE:-0.0006}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
D_MODEL="${D_MODEL:-96}"
LAYERS="${LAYERS:-2}"
HEADS="${HEADS:-4}"
DROPOUT="${DROPOUT:-0.08}"
DELTA_KERNELS="${DELTA_KERNELS:-6}"
MAX_EVENTS="${MAX_EVENTS:-2048}"
MAX_QUERIES_PER_SAMPLE="${MAX_QUERIES_PER_SAMPLE:-768}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${BASE_ROOT}" "${LOG_DIR}"

DATASETS=(p12 mimic ushcn humanactivity)
SEEDS=(2024 2025 2026)

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_args=(--overwrite)
fi

model_args=(
  --architecture-ablation apex
  --learning-rate "${LEARNING_RATE}"
  --weight-decay "${WEIGHT_DECAY}"
  --d-model "${D_MODEL}"
  --layers "${LAYERS}"
  --heads "${HEADS}"
  --dropout "${DROPOUT}"
  --delta-kernels "${DELTA_KERNELS}"
  --max-events "${MAX_EVENTS}"
  --max-queries-per-sample "${MAX_QUERIES_PER_SAMPLE}"
)

gate_args=(--save-gate-log)
if [[ -n "${MAX_GATE_LOG_ROWS}" ]]; then
  gate_args+=(--max-gate-log-rows "${MAX_GATE_LOG_ROWS}")
fi

active_job_count() {
  jobs -rp | wc -l
}

wait_for_slot() {
  while [[ "$(active_job_count)" -ge "${MAX_PARALLEL}" ]]; do
    sleep 10
  done
}

run_job() {
  local name="$1"
  local output_root="$2"
  shift 2

  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_root}/chronoa_summary.csv" ]]; then
    echo "[SKIP] ${name} already completed $(date -Is)"
    return 0
  fi

  mkdir -p "${output_root}"
  echo "[START] ${name} $(date -Is)"
  if "${PYTHON_BIN}" "${RUNNER}" "$@" \
      --output-root "${output_root}" \
      --device "${DEVICE}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --batch-size "${BATCH_SIZE}" \
      "${model_args[@]}" \
      "${overwrite_args[@]}"; then
    echo "[DONE] ${name} $(date -Is)"
  else
    local status=$?
    echo "[FAIL] ${name} status=${status} $(date -Is)"
    return "${status}"
  fi
}

submit() {
  wait_for_slot
  run_job "$@" &
}

for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    if [[ "${RUN_MSE}" == "1" ]]; then
      submit \
        "mse_only_seed${seed}_${dataset}" \
        "${BASE_ROOT}/mse_only/seed${seed}_${dataset}" \
        --dataset "${dataset}" \
        --loss-mode mse \
        --random-seed "${seed}" \
        --run-tag "apex_mse_only_seed${seed}"
    fi

    if [[ "${RUN_GATE}" == "1" ]]; then
      submit \
        "gate_default_seed${seed}_${dataset}" \
        "${BASE_ROOT}/gate_logs/seed${seed}_${dataset}" \
        --dataset "${dataset}" \
        --loss-mode default \
        --random-seed "${seed}" \
        --run-tag "apex_gate_seed${seed}" \
        "${gate_args[@]}"
    fi
  done
done

wait

if [[ "${RUN_MSE}" == "1" ]]; then
  "${PYTHON_BIN}" "${OBJECTIVE_ANALYSIS}" \
    --root "${BASE_ROOT}" \
    --output-dir "${BASE_ROOT}/objective_summaries"
fi

if [[ "${RUN_GATE}" == "1" ]]; then
  "${PYTHON_BIN}" "${GATE_ANALYSIS}" \
    --root "${BASE_ROOT}/gate_logs" \
    --output-dir "${BASE_ROOT}/gate_summaries"
fi

echo "[ALL DONE] $(date -Is)"
echo "MSE-only summaries: $(find "${BASE_ROOT}/mse_only" -name chronoa_summary.csv 2>/dev/null | wc -l)"
echo "Gate-log summaries: $(find "${BASE_ROOT}/gate_logs" -name chronoa_summary.csv 2>/dev/null | wc -l)"
