#!/usr/bin/env python3
"""Run Chronoa-EDGE forecasting on upstream irregular time-series benchmark splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chronoa.cli_utils import slugify, split_csv_values, unique_preserve_order
from chronoa.experiments.chronoa_forecasting import (
    ARCHITECTURE_ABLATIONS,
    HISTORY_PERTURBATIONS,
    RUN_NAME_PREFIX,
    ChronoaConfig,
    canonical_architecture_ablation,
    canonical_dataset_name,
    canonical_history_perturbation,
    run,
)

DATASET_ORDER = ["P12", "MIMIC", "USHCN", "HumanActivity"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Chronoa-EDGE neural event graph forecaster.")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Dataset to run: P12, MIMIC, USHCN, HumanActivity, or all. May be repeated or comma-separated.",
    )
    parser.add_argument("--all", action="store_true", help="Run all supported datasets.")
    parser.add_argument("--output-root", type=Path, default=Path("chronoa_results"), help="Directory for result CSVs and logs.")
    parser.add_argument("--seq-len", type=int, default=None, help="Override benchmark lookback length.")
    parser.add_argument("--pred-len", type=int, default=None, help="Override benchmark prediction length.")
    parser.add_argument("--architecture-ablation", choices=ARCHITECTURE_ABLATIONS, default="full")
    parser.add_argument("--history-perturbation", choices=HISTORY_PERTURBATIONS, default="original")
    parser.add_argument("--history-keep-fraction", type=float, default=1.0)
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--run-tag", default="", help="Optional suffix added to output artifact names.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite this run's CSV/log artifacts.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Smoke-test limit on test samples per dataset.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on training samples.")
    parser.add_argument("--max-validation-samples", type=int, default=None, help="Optional cap on validation samples.")
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
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device string.")
    parser.add_argument("--trace-every", type=int, default=0, help="Log per-sample traces every N valid samples.")
    parser.add_argument("--progress-every", type=int, default=100, help="Log progress every N evaluated samples.")
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> list[str]:
    raw_names = split_csv_values(args.dataset)
    if args.all or any(name.lower() == "all" for name in raw_names) or not raw_names:
        return DATASET_ORDER
    return unique_preserve_order(canonical_dataset_name(name) for name in raw_names)


def run_name_suffix(args: argparse.Namespace) -> str:
    parts: list[str] = []
    if args.run_tag:
        parts.append(slugify(args.run_tag))
    ablation = canonical_architecture_ablation(args.architecture_ablation)
    if ablation != "full":
        parts.append(ablation)
    perturbation = canonical_history_perturbation(args.history_perturbation)
    if perturbation != "original":
        parts.append(perturbation)
    if float(args.history_keep_fraction) < 1.0:
        parts.append(f"keep{int(round(float(args.history_keep_fraction) * 100)):03d}")
    if args.seq_len is not None or args.pred_len is not None:
        parts.append(f"sl{args.seq_len or 'default'}_pl{args.pred_len or 'default'}")
    return "_".join(part for part in parts if part)


def main() -> None:
    args = parse_args()
    summaries: list[dict] = []

    for dataset in selected_datasets(args):
        output_dir = args.output_root / dataset.lower()
        base_run_name = f"{RUN_NAME_PREFIX}_{dataset.lower()}"
        suffix = run_name_suffix(args)
        run_name = f"{base_run_name}_{suffix}" if suffix else base_run_name
        config = ChronoaConfig(
            dataset_name=dataset,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            max_test_samples=args.max_test_samples,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
            architecture_ablation=args.architecture_ablation,
            history_perturbation=args.history_perturbation,
            history_keep_fraction=args.history_keep_fraction,
            random_seed=args.random_seed,
            overwrite=args.overwrite,
            run_name=run_name,
            output_dir=output_dir,
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
        summaries.append(run(config))

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "chronoa_summary.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False)
    print(f"Chronoa summary written to {summary_path}")


if __name__ == "__main__":
    main()
