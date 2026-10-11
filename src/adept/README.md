# ADEPT Source Package

ADEPT is a query-conditioned event graph forecaster for irregular multivariate
time-series forecasting. It keeps observed measurements as asynchronous
time-variable-value events, retrieves evidence for each future query, and
decodes the forecast through adaptive delta-kernel experts.

This directory contains the local Python package used by the benchmark runners
and analysis scripts. The repository directory is still named `Chronoa` for
continuity, but the model, paper, and source package are ADEPT-facing.

## Architecture

ADEPT is implemented around five components:

- Event tokenization: each finite historical observation is represented by a
  scaled value projection, a learned variable embedding, and continuous-time
  Fourier features.
- Transformer event memory: padded irregular event sets are contextualized with
  batch-first self-attention.
- Adaptive variable graph: normalized variable embeddings define a learned
  query-to-source variable affinity matrix.
- Query-conditioned retrieval: future time-variable queries attend to event
  memory and retrieve multi-scale continuous-time delta-kernel contexts.
- Adaptive expert decoder: the final forecast gates between a neural query head
  and history-derived scalar experts for same-variable latest value,
  same-variable exponential memory, local trend, graph-aware memory, and global
  entity memory.

The expert values are differentiable functions of the observed input history.
They do not use test labels, external baseline predictions, or post hoc routing.

## Package Map

```text
src/adept/
  benchmark.py                         # vendored benchmark path helpers
  cli_utils.py                         # shared CLI parsing helpers
  paths.py                             # repository, source, and benchmark paths
  experiments/chronoa_forecasting.py   # ADEPT model, data adapter, runner
```

The main implementation file keeps a few legacy class names such as
`ChronoaConfig` for backward compatibility with existing experiment scripts.
Paper-facing names, log output, and result tables should use `ADEPT`.

## Runner Entry Points

From the repository root:

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

Useful options:

- `--dataset`: `P12`, `MIMIC`, `USHCN`, `HumanActivity`, or `all`.
- `--architecture-ablation`: `apex` for the ADEPT expert-decoder model,
  `full`, `no_delta_kernel`, `no_graph_mixer`, `no_transformer`,
  `no_time_encoding`, or `direct_decoder`.
- `--loss-mode`: `default` for the paper objective or `mse` for the objective
  control run.
- `--save-gate-log`: export per-query gate weights and expert values.
- `--history-perturbation`: run timing/value perturbation diagnostics.
- `--max-train-samples`, `--max-validation-samples`, `--max-test-samples`:
  smoke-test caps.

Result artifact names still use some `chronoa_*` filenames for compatibility
with existing analysis scripts. The model name recorded in new outputs is
`ADEPT`.

## Diagnostics

Gate/expert diagnostics:

```bash
python scripts/experiments/run_chronoa_forecasting.py \
  --dataset MIMIC \
  --architecture-ablation apex \
  --loss-mode default \
  --save-gate-log \
  --random-seed 2024 \
  --output-root results/adept_gate_logs \
  --device cuda \
  --overwrite

python scripts/analysis/adept_gate_analysis.py \
  --root results/adept_gate_logs \
  --output-dir results/adept_gate_logs/gate_summaries
```

Objective-control aggregation:

```bash
python scripts/analysis/adept_objective_analysis.py \
  --root results/adept_point2_controls \
  --output-dir results/adept_point2_controls/objective_summaries
```

The repository also includes `scripts/experiments/run_adept_point2_controls.sh`,
which launches the MSE-only objective control and default-loss gate diagnostic
runs over the four benchmark datasets and seeds 2024--2026.

## Data and Benchmarks

ADEPT uses the vendored irregular time-series benchmark adapter under
`vendor/upstream_benchmark`. The benchmark datasets follow the upstream naming
conventions:

- `P12`: PhysioNet Challenge 2012.
- `MIMIC`: MIMIC-III De Brouwer protocol.
- `USHCN`: irregular climate station observations.
- `HumanActivity`: wearable-sensor activity trajectories.

All reported ADEPT metrics are computed in benchmark-scaled space on active
target-mask entries using global masked MAE and MSE.
