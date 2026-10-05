#!/usr/bin/env bash
set -euo pipefail

# Launch the full CHRONOA-EDGE experiment suite for three seeds with a fixed
# parallelism cap. Default jobs are left unchanged; USHCN-enhanced jobs are
# written to separate output folders so they cannot overwrite the main results.

ROOT="${CHRONOA_ROOT:-/home/nckh2/qa}"
cd "${ROOT}"

RUNNER="${RUNNER:-Chronoa/scripts/experiments/run_chronoa_forecasting.py}"
BASE_ROOT="${OUTPUT_ROOT:-chronoa_edge_seeded_parallel5}"
LOG_DIR="${LOG_DIR:-logs}"
MAX_PARALLEL="${MAX_PARALLEL:-5}"

DEVICE="${DEVICE:-cuda}"
GPU_ID="${GPU_ID:-0}"
EPOCHS="${EPOCHS:-80}"
PATIENCE="${PATIENCE:-12}"
BATCH_SIZE="${BATCH_SIZE:-16}"
OVERWRITE="${OVERWRITE:-1}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
GLOBAL_THROTTLE="${GLOBAL_THROTTLE:-0}"

RUN_MISSING_SEED1729="${RUN_MISSING_SEED1729:-1}"
RUN_USHCN_ENHANCED="${RUN_USHCN_ENHANCED:-1}"

# Conservative USHCN candidate: same architecture family, slightly richer
# temporal retrieval, separate outputs. These flags affect only the enhanced
# USHCN jobs below.
USHCN_LR="${USHCN_LR:-0.0006}"
USHCN_DROPOUT="${USHCN_DROPOUT:-0.05}"
USHCN_MAX_EVENTS="${USHCN_MAX_EVENTS:-2048}"
USHCN_DELTA_KERNELS="${USHCN_DELTA_KERNELS:-6}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
mkdir -p "${BASE_ROOT}" "${LOG_DIR}"

DATASETS=(p12 mimic ushcn humanactivity)
SEEDS=(2024 2025 2026)
ARCHES=(full no_delta_kernel no_graph_mixer no_transformer no_time_encoding direct_decoder)
PERTURBATIONS=(original shuffle_timestamps swap_halves drop_early_history keep_last_obs keep_time_gaps)
KEEP_FRACTIONS=(0.75 0.50 0.25)

overwrite_args=()
if [[ "${OVERWRITE}" == "1" ]]; then
  overwrite_args=(--overwrite)
fi

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
  if python3 "${RUNNER}" "$@" \
      --output-root "${output_root}" \
      --device "${DEVICE}" \
      --epochs "${EPOCHS}" \
      --patience "${PATIENCE}" \
      --batch-size "${BATCH_SIZE}" \
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

window_job() {
  local seed="$1"
  local dataset="$2"
  local seq_len="$3"
  local pred_len="$4"
  local label="${seq_len}_${pred_len}"
  seeded_job "${seed}" "window_${dataset}_${label}" \
    --dataset "${dataset}" \
    --seq-len "${seq_len}" \
    --pred-len "${pred_len}" \
    --run-tag "window_${label}_seed${seed}"
}

ushcn_enhanced_job() {
  local seed="$1"
  local seq_len="$2"
  local pred_len="$3"
  local label="${seq_len}_${pred_len}"
  seeded_job "${seed}" "ushcn_enhanced_${label}" \
    --dataset ushcn \
    --seq-len "${seq_len}" \
    --pred-len "${pred_len}" \
    --learning-rate "${USHCN_LR}" \
    --dropout "${USHCN_DROPOUT}" \
    --max-events "${USHCN_MAX_EVENTS}" \
    --delta-kernels "${USHCN_DELTA_KERNELS}" \
    --run-tag "ushcn_enhanced_${label}_seed${seed}"
}

if [[ "${RUN_MISSING_SEED1729}" == "1" ]]; then
  for keep_fraction in "${KEEP_FRACTIONS[@]}"; do
    keep_tag="${keep_fraction/./}"
    submit "fix_missing_seed1729_mimic_keep_${keep_tag}" \
      --dataset mimic \
      --history-keep-fraction "${keep_fraction}" \
      --run-tag "fix_missing_mimic_keep_${keep_tag}_seed1729" \
      --random-seed 1729
  done
fi

for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    seeded_job "${seed}" "standard_${dataset}" \
      --dataset "${dataset}" \
      --run-tag "standard_seed${seed}"
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
      seeded_job "${seed}" "temporal_${dataset}_${perturbation}" \
        --dataset "${dataset}" \
        --history-perturbation "${perturbation}" \
        --run-tag "temporal_${perturbation}_seed${seed}"
    done

    for keep_fraction in "${KEEP_FRACTIONS[@]}"; do
      keep_tag="${keep_fraction/./}"
      seeded_job "${seed}" "temporal_${dataset}_keep_${keep_tag}" \
        --dataset "${dataset}" \
        --history-keep-fraction "${keep_fraction}" \
        --run-tag "temporal_keep_${keep_tag}_seed${seed}"
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

  if [[ "${RUN_USHCN_ENHANCED}" == "1" ]]; then
    ushcn_enhanced_job "${seed}" 50 1
    ushcn_enhanced_job "${seed}" 100 3
    ushcn_enhanced_job "${seed}" 150 3
    ushcn_enhanced_job "${seed}" 150 7
  fi
done

wait || true
echo "[ALL DONE] $(date -Is)"
echo "Summaries: $(find "${BASE_ROOT}" -name chronoa_summary.csv | wc -l) / expected 255"
echo "Failures in this log can be found with: grep -n '\\[FAIL\\]' ${LOG_DIR}/chronoa_seeded_parallel5.log"
