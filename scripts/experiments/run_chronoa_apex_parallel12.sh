#!/usr/bin/env bash
set -euo pipefail

# Full CHRONOA-APEX rerun.
# APEX is a trainable adaptive expert decoder inside CHRONOA, not a hidden
# external-model ensemble. It combines the neural event graph decoder with
# learned gates over temporal scalar experts.

ROOT="${CHRONOA_ROOT:-/home/nckh2/qa}"
cd "${ROOT}"

RUNNER="${RUNNER:-Chronoa/scripts/experiments/run_chronoa_forecasting.py}"
PYTHON_BIN="${PYTHON_BIN:-/home/nckh2/qa/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

BASE_ROOT="${OUTPUT_ROOT:-chronoa_apex_seeded_parallel12}"
LOG_DIR="${LOG_DIR:-logs}"
MAX_PARALLEL="${MAX_PARALLEL:-12}"

DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-100}"
PATIENCE="${PATIENCE:-16}"
BATCH_SIZE="${BATCH_SIZE:-8}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
GLOBAL_THROTTLE="${GLOBAL_THROTTLE:-0}"

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
ARCHES=(apex full no_delta_kernel no_graph_mixer no_transformer no_time_encoding direct_decoder)
PERTURBATIONS=(original shuffle_timestamps swap_halves drop_early_history keep_last_obs keep_time_gaps)
KEEP_FRACTIONS=(0.75 0.50 0.25)

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_args=(--overwrite)
fi

model_args=(
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

active_job_count() {
  if [[ "${GLOBAL_THROTTLE}" == "1" ]]; then
    pgrep -af '[r]un_chronoa_forecasting.py' | wc -l
  else
    jobs -rp | wc -l
  fi
}

wait_for_slot() {
  while [[ "$(active_job_count)" -ge "${MAX_PARALLEL}" ]]; do
    sleep 10
  done
}

job_is_running() {
  local output_root="$1"
  pgrep -af '[r]un_chronoa_forecasting.py' | grep -F -- "--output-root ${output_root}" >/dev/null
}

run_job() {
  local name="$1"
  shift

  local output_root="${BASE_ROOT}/${name}"
  if [[ "${SKIP_COMPLETED}" == "1" && -f "${output_root}/chronoa_summary.csv" ]]; then
    echo "[SKIP] ${name} already completed $(date -Is)"
    return 0
  fi
  if [[ "${SKIP_COMPLETED}" == "1" ]] && job_is_running "${output_root}"; then
    echo "[SKIP-RUNNING] ${name} already active $(date -Is)"
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

seeded_job() {
  local seed="$1"
  local name="$2"
  shift 2
  submit "seed${seed}_${name}" "$@" --random-seed "${seed}"
}

apex_job() {
  local seed="$1"
  local name="$2"
  shift 2
  seeded_job "${seed}" "${name}" "$@" --architecture-ablation apex
}

window_job() {
  local seed="$1"
  local dataset="$2"
  local seq_len="$3"
  local pred_len="$4"
  local label="${seq_len}_${pred_len}"
  apex_job "${seed}" "window_${dataset}_${label}" \
    --dataset "${dataset}" \
    --seq-len "${seq_len}" \
    --pred-len "${pred_len}" \
    --run-tag "apex_window_${label}_seed${seed}"
}

for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    apex_job "${seed}" "standard_${dataset}" \
      --dataset "${dataset}" \
      --run-tag "apex_standard_seed${seed}"
  done

  for dataset in "${DATASETS[@]}"; do
    for architecture in "${ARCHES[@]}"; do
      seeded_job "${seed}" "arch_${dataset}_${architecture}" \
        --dataset "${dataset}" \
        --architecture-ablation "${architecture}" \
        --run-tag "arch_${architecture}_seed${seed}"
    done
  done

  for dataset in "${DATASETS[@]}"; do
    for perturbation in "${PERTURBATIONS[@]}"; do
      apex_job "${seed}" "temporal_${dataset}_${perturbation}" \
        --dataset "${dataset}" \
        --history-perturbation "${perturbation}" \
        --run-tag "apex_temporal_${perturbation}_seed${seed}"
    done

    for keep_fraction in "${KEEP_FRACTIONS[@]}"; do
      keep_tag="${keep_fraction/./}"
      apex_job "${seed}" "temporal_${dataset}_keep_${keep_tag}" \
        --dataset "${dataset}" \
        --history-keep-fraction "${keep_fraction}" \
        --run-tag "apex_temporal_keep_${keep_tag}_seed${seed}"
    done
  done

  window_job "${seed}" humanactivity 1000 100
  window_job "${seed}" humanactivity 2000 200
  window_job "${seed}" humanactivity 3000 300
  window_job "${seed}" humanactivity 3000 600

  window_job "${seed}" mimic 12 1
  window_job "${seed}" mimic 24 3
  window_job "${seed}" mimic 72 3
  window_job "${seed}" mimic 72 6

  window_job "${seed}" p12 12 1
  window_job "${seed}" p12 24 3
  window_job "${seed}" p12 36 3
  window_job "${seed}" p12 36 6

  window_job "${seed}" ushcn 50 1
  window_job "${seed}" ushcn 100 3
  window_job "${seed}" ushcn 150 3
  window_job "${seed}" ushcn 150 7
done

wait || true
echo "[ALL DONE] $(date -Is)"
echo "Summaries: $(find "${BASE_ROOT}" -name chronoa_summary.csv | wc -l) / expected 252"
echo "Failures in this log can be found with: grep -n '\\[FAIL\\]' ${LOG_DIR}/chronoa_apex_parallel12.log"
