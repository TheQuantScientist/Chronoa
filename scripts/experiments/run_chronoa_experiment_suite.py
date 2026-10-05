#!/usr/bin/env python3
"""Run the Chronoa-EDGE experiment suites used by the paper.

Chronoa-EDGE is an end-to-end neural event graph forecaster. The suites are:
  * standard: main neural model on the four benchmark windows
  * architecture: neural component ablations
  * temporal: history perturbation and sparsity stress tests
  * window: lookback-horizon sensitivity sweeps

This runner does not launch external baselines. Use the upstream baseline runner for
APN, GraFITi, tPatchGNN, Warpformer, and related comparators.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chronoa.cli_utils import split_csv_values, unique_preserve_order  # noqa: E402
from chronoa.experiments.chronoa_forecasting import (  # noqa: E402
    ARCHITECTURE_ABLATIONS,
    HISTORY_PERTURBATIONS,
    ChronoaConfig,
    canonical_architecture_ablation,
    canonical_dataset_name,
    canonical_history_perturbation,
    run,
)

DATASET_ORDER = ["P12", "MIMIC", "USHCN", "HumanActivity"]
ARCHITECTURE_SUITE = ["full", "no_delta_kernel", "no_graph_mixer", "no_transformer", "no_time_encoding", "direct_decoder"]
TEMPORAL_ABLATIONS = ["original", "shuffle_timestamps", "swap_halves", "drop_early_history", "keep_last_obs", "keep_time_gaps"]
WINDOW_SWEEP = {
    "P12": [(12, 1), (24, 3), (36, 3), (36, 6)],
    "MIMIC": [(12, 1), (24, 3), (72, 3), (72, 6)],
    "USHCN": [(50, 1), (100, 3), (150, 3), (150, 7)],
    "HumanActivity": [(1000, 100), (2000, 200), (3000, 300), (3000, 600)],
}


@dataclass(frozen=True)
class Job:
    suite: str
    dataset: str
    run_tag: str
    seq_len: int | None = None
    pred_len: int | None = None
    architecture_ablation: str = "full"
    history_perturbation: str = "original"
    history_keep_fraction: float = 1.0

    def output_dir(self, output_root: Path) -> Path:
        return output_root / self.suite / self.dataset.lower()

    def run_name(self) -> str:
        return f"chronoa_{self.dataset.lower()}_{self.run_tag}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Chronoa-EDGE paper experiment suites.")
    parser.add_argument("--suite", action="append", default=[], help="standard, architecture, temporal, window, or all. Repeatable/comma-separated.")
    parser.add_argument("--dataset", action="append", default=[], help="P12, MIMIC, USHCN, HumanActivity, or all. Repeatable/comma-separated.")
    parser.add_argument("--output-root", type=Path, default=Path("chronoa_experiments"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional smoke-test cap per dataset.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on training samples.")
    parser.add_argument("--max-validation-samples", type=int, default=None, help="Optional cap on validation samples.")
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--delta-kernels", type=int, default=4)
    parser.add_argument("--max-events", type=int, default=1536)
    parser.add_argument("--max-queries-per-sample", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--trace-every", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true", help="Print planned jobs without running them.")
    parser.add_argument("--build-tables", action="store_true", help="Generate compact CSV/Markdown tables after each suite.")
    return parser.parse_args()


def selected_suites(raw: list[str]) -> list[str]:
    defaults = ["standard", "architecture", "temporal", "window"]
    aliases = {
        "all": "all",
        "ablation": "architecture",
        "ablations": "architecture",
        "arch": "architecture",
        "perturbation": "temporal",
        "sweep": "window",
    }
    suites: list[str] = []
    for item in split_csv_values(raw, default=defaults):
        normalized = item.strip().lower().replace("-", "_")
        normalized = aliases.get(normalized, normalized)
        if normalized == "all":
            suites.extend(defaults)
        elif normalized in defaults:
            suites.append(normalized)
        else:
            raise ValueError(f"Unsupported suite {item!r}. Use one of: {', '.join(defaults)}, all")
    return unique_preserve_order(suites)


def selected_datasets(raw: list[str]) -> list[str]:
    values = split_csv_values(raw, default=DATASET_ORDER)
    if any(value.lower() == "all" for value in values):
        return DATASET_ORDER
    return unique_preserve_order(canonical_dataset_name(value) for value in values)


def jobs_for_suite(suite: str, datasets: list[str]) -> list[Job]:
    jobs: list[Job] = []
    for dataset in datasets:
        if suite == "standard":
            jobs.append(Job(suite=suite, dataset=dataset, run_tag="standard"))
        elif suite == "architecture":
            for condition in ARCHITECTURE_SUITE:
                ablation = canonical_architecture_ablation(condition)
                tag = "full" if ablation == "full" else ablation
                jobs.append(Job(suite=suite, dataset=dataset, run_tag=tag, architecture_ablation=ablation))
        elif suite == "temporal":
            for perturbation in TEMPORAL_ABLATIONS:
                jobs.append(
                    Job(
                        suite=suite,
                        dataset=dataset,
                        run_tag=canonical_history_perturbation(perturbation),
                        history_perturbation=perturbation,
                    )
                )
            for keep_fraction in (0.75, 0.5, 0.25):
                tag = f"keep_{int(keep_fraction * 100)}pct_history"
                jobs.append(Job(suite=suite, dataset=dataset, run_tag=tag, history_keep_fraction=keep_fraction))
        elif suite == "window":
            for seq_len, pred_len in WINDOW_SWEEP[dataset]:
                jobs.append(Job(suite=suite, dataset=dataset, run_tag=f"sl{seq_len}_pl{pred_len}", seq_len=seq_len, pred_len=pred_len))
        else:
            raise ValueError(f"Unsupported suite: {suite}")
    return jobs


def run_job(job: Job, args: argparse.Namespace) -> dict[str, float | str]:
    config = ChronoaConfig(
        dataset_name=job.dataset,
        seq_len=job.seq_len,
        pred_len=job.pred_len,
        architecture_ablation=job.architecture_ablation,
        history_perturbation=job.history_perturbation,
        history_keep_fraction=job.history_keep_fraction,
        random_seed=args.random_seed,
        max_test_samples=args.max_test_samples,
        max_train_samples=args.max_train_samples,
        max_validation_samples=args.max_validation_samples,
        overwrite=args.overwrite,
        run_name=job.run_name(),
        output_dir=job.output_dir(args.output_root),
        trace_every=args.trace_every,
        progress_every=args.progress_every,
        max_events=args.max_events,
        max_queries_per_sample=args.max_queries_per_sample,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        d_model=args.d_model,
        layers=args.layers,
        heads=args.heads,
        dropout=args.dropout,
        delta_kernels=args.delta_kernels,
        device=args.device,
    )
    summary = run(config)
    summary["Suite"] = job.suite
    summary["Run_tag"] = job.run_tag
    summary["History_keep_fraction"] = job.history_keep_fraction
    return summary


def write_summary(output_root: Path, suite: str, rows: list[dict[str, float | str]]) -> Path:
    summary_dir = output_root / suite
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"chronoa_{suite}_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[summary] {summary_path}")
    return summary_path


def build_tables(summary_path: Path, suite: str) -> None:
    analysis_script = PROJECT_ROOT / "scripts" / "analysis" / "chronoa_experiment_tables.py"
    if not analysis_script.exists():
        print(f"[skip tables] missing {analysis_script}")
        return
    kind = {"architecture": "architecture", "temporal": "temporal", "window": "sweep"}.get(suite)
    if kind is None:
        efficiency_script = PROJECT_ROOT / "scripts" / "analysis" / "chronoa_efficiency.py"
        subprocess.run([sys.executable, str(efficiency_script), "--summary", str(summary_path)], check=True)
        return
    subprocess.run([sys.executable, str(analysis_script), "--summary", str(summary_path), "--kind", kind], check=True)


def main() -> None:
    args = parse_args()
    suites = selected_suites(args.suite)
    datasets = selected_datasets(args.dataset)
    print(f"Suites: {', '.join(suites)}")
    print(f"Datasets: {', '.join(datasets)}")

    for suite in suites:
        jobs = jobs_for_suite(suite, datasets)
        print(f"[{suite}] planned jobs: {len(jobs)}")
        for job in jobs:
            print(
                "  "
                f"{job.dataset}/{job.run_tag} seq={job.seq_len or 'default'} pred={job.pred_len or 'default'} "
                f"arch={job.architecture_ablation} perturb={job.history_perturbation} "
                f"keep={job.history_keep_fraction}"
            )
        if args.dry_run:
            continue
        rows = [run_job(job, args) for job in jobs]
        summary_path = write_summary(args.output_root, suite, rows)
        if args.build_tables:
            build_tables(summary_path, suite)


if __name__ == "__main__":
    main()
