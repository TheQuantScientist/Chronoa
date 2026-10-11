# ADEPT

Query-conditioned event graph forecasting with adaptive delta-kernel experts
for irregular multivariate time series.

ADEPT keeps asynchronous observations as event tokens, builds transformer event
memory, learns adaptive cross-variable relations, retrieves query-specific
evidence with continuous-time delta kernels, and decodes each future
time-variable target with a gated mixture of neural and history-derived experts.

The repository directory is named `Chronoa` for continuity with earlier
experiments. The paper-facing model name is **ADEPT**.

## What Is In This Repo

```text
src/adept/                         ADEPT source package
scripts/experiments/               benchmark and diagnostic runners
scripts/analysis/                  table, gate, and objective summaries
vendor/upstream_benchmark/         vendored IMTS benchmark code
docs/                              manuscript and bibliography files
results/                           generated experiment outputs
```

## Install

Use Python 3.10 or newer.

```bash
cd Chronoa
python -m venv .venv
source .venv/bin/activate
pip install -e ".[benchmark]"
```

If the project virtualenv already exists at `/home/nckh2/qa/.venv`, the scripts
can also be run with that interpreter.

## Quickstart

Run ADEPT on one benchmark split:

```bash
python scripts/experiments/run_chronoa_forecasting.py \
  --dataset P12 \
  --architecture-ablation apex \
  --loss-mode default \
  --random-seed 2024 \
  --output-root results/adept_quickstart \
  --device cuda \
  --epochs 100 \
  --patience 16 \
  --batch-size 8 \
  --learning-rate 0.0006 \
  --dropout 0.08 \
  --delta-kernels 6 \
  --max-events 2048 \
  --max-queries-per-sample 768 \
  --overwrite
```

Run all four datasets:

```bash
python scripts/experiments/run_chronoa_forecasting.py \
  --all \
  --architecture-ablation apex \
  --output-root results/adept_standard \
  --device cuda \
  --overwrite
```

## Main CLI Options

- `--dataset`: `P12`, `MIMIC`, `USHCN`, `HumanActivity`, or `all`.
- `--architecture-ablation`: `apex`, `full`, `no_delta_kernel`,
  `no_graph_mixer`, `no_transformer`, `no_time_encoding`, or
  `direct_decoder`.
- `--loss-mode`: `default` for the paper objective or `mse` for an objective
  control run.
- `--history-perturbation`: `original`, `shuffle_timestamps`, `swap_halves`,
  `drop_early_history`, `keep_last_obs`, or `keep_time_gaps`.
- `--save-gate-log`: export per-query adaptive expert gate diagnostics.
- `--max-train-samples`, `--max-validation-samples`, `--max-test-samples`:
  smoke-test caps for fast local checks.

## Diagnostics

Gate behavior:

```bash
python scripts/experiments/run_chronoa_forecasting.py \
  --dataset MIMIC \
  --architecture-ablation apex \
  --save-gate-log \
  --output-root results/adept_gate_logs \
  --device cuda \
  --overwrite

python scripts/analysis/adept_gate_analysis.py \
  --root results/adept_gate_logs \
  --output-dir results/adept_gate_logs/gate_summaries
```

Reviewer objective controls and gate logs:

```bash
MAX_PARALLEL=4 GPU_ID=0 \
bash scripts/experiments/run_adept_point2_controls.sh
```

The generated summaries are written under
`results/adept_point2_controls/objective_summaries` and
`results/adept_point2_controls/gate_summaries` unless `OUTPUT_ROOT` is
overridden.

## Outputs

The benchmark runner writes per-run CSVs, training logs, optional gate logs, and
a compatibility summary named `chronoa_summary.csv`. New rows record the model
as `ADEPT`; some filenames keep the older `chronoa` prefix so existing analysis
scripts continue to work.

Common outputs:

- `*_Results.csv`: global masked MAE/MSE for a run.
- `*_TrainingLog.csv`: epoch-level loss and validation scores.
- `*_DetailLog.csv`: per-entity or per-window evaluation details.
- `*_GateLog.csv`: optional per-query gate weights, expert values, and errors.
- `chronoa_summary.csv`: aggregate run summary consumed by analysis scripts.

## Source Package

The source package lives in `src/adept`. See `src/adept/README.md` for the
package map, model internals, and diagnostic conventions.
