#!/usr/bin/env python3
"""Run Chronoa-EDGE windows for a balanced cross-model L/H grid.

This fills neural Chronoa-EDGE results for lookback/horizon settings used by the
cross-baseline window sweep. Existing result CSVs are reused unless --overwrite
is passed.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from chronoa.experiments.chronoa_forecasting import ChronoaConfig, run  # noqa: E402


BALANCED_WINDOWS: dict[str, list[tuple[int, int]]] = {
    "HumanActivity": [(1000, 100), (2000, 200), (3000, 300), (3000, 600)],
    "MIMIC": [(12, 1), (24, 3), (72, 3), (72, 6)],
    "P12": [(12, 1), (24, 3), (36, 3), (36, 6)],
    "USHCN": [(50, 1), (100, 3), (150, 3), (150, 7)],
}


@dataclass(frozen=True)
class Job:
    dataset: str
    seq_len: int
    pred_len: int

    @property
    def tag(self) -> str:
        return f"sl{self.seq_len}_pl{self.pred_len}"

    @property
    def run_name(self) -> str:
        return f"chronoa_{self.dataset.lower()}_{self.tag}"

    def output_dir(self, output_root: Path) -> Path:
        return output_root / "window" / self.dataset.lower()

    def result_csv(self, output_root: Path) -> Path:
        return self.output_dir(output_root) / f"{self.run_name}_{self.dataset}_Results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("chronoa_experiments"))
    parser.add_argument("--dataset", action="append", default=[], help="Optional dataset filter; repeatable.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--random-seed", type=int, default=1729)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--trace-every", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_jobs(raw_datasets: list[str]) -> list[Job]:
    wanted = {item.lower() for value in raw_datasets for item in value.split(",") if item.strip()}
    jobs: list[Job] = []
    for dataset, windows in BALANCED_WINDOWS.items():
        if wanted and dataset.lower() not in wanted:
            continue
        for seq_len, pred_len in windows:
            jobs.append(Job(dataset=dataset, seq_len=seq_len, pred_len=pred_len))
    return jobs


def summarize_existing(job: Job, output_root: Path) -> dict[str, str | float | int]:
    result_csv = job.result_csv(output_root)
    total_predictions = 0
    mae_weighted = 0.0
    mse_weighted = 0.0
    with result_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            predictions = int(float(row["Predictions"]))
            total_predictions += predictions
            mae_weighted += float(row["MAE_scaled"]) * predictions
            mse_weighted += float(row["MSE_scaled"]) * predictions
    if total_predictions == 0:
        raise ValueError(f"No predictions in {result_csv}")
    return {
        "Dataset": job.dataset,
        "Condition": f"seq={job.seq_len}, pred={job.pred_len}",
        "Run_tag": job.tag,
        "Seq_len": job.seq_len,
        "Pred_len": job.pred_len,
        "MAE": mae_weighted / total_predictions,
        "MSE": mse_weighted / total_predictions,
        "Predictions": total_predictions,
        "Source": "existing",
    }


def normalize_summary(summary: dict[str, object], job: Job, source: str) -> dict[str, object]:
    mae = summary.get("MAE_scaled", summary.get("MAE"))
    mse = summary.get("MSE_scaled", summary.get("MSE"))
    return {
        "Dataset": job.dataset,
        "Condition": f"seq={job.seq_len}, pred={job.pred_len}",
        "Run_tag": job.tag,
        "Seq_len": job.seq_len,
        "Pred_len": job.pred_len,
        "MAE": mae,
        "MSE": mse,
        "Predictions": summary.get("Predictions"),
        "Source": source,
    }


def write_summary(output_root: Path, rows: list[dict[str, object]]) -> Path:
    summary_dir = output_root / "window"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / "chronoa_balanced_cross_window_summary.csv"
    fieldnames = ["Dataset", "Condition", "Run_tag", "Seq_len", "Pred_len", "MAE", "MSE", "Predictions", "Source"]
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path


def main() -> None:
    args = parse_args()
    if not args.output_root.is_absolute():
        args.output_root = (PROJECT_ROOT / args.output_root).resolve()

    rows: list[dict[str, object]] = []
    for job in selected_jobs(args.dataset):
        result_csv = job.result_csv(args.output_root)
        exists = result_csv.exists()
        action = "run"
        if exists and not args.overwrite:
            action = "skip"
        print(f"[{action}] {job.dataset} {job.seq_len}/{job.pred_len} -> {result_csv}")
        if args.dry_run:
            continue
        if exists and not args.overwrite:
            rows.append(summarize_existing(job, args.output_root))
            continue
        config = ChronoaConfig(
            dataset_name=job.dataset,
            seq_len=job.seq_len,
            pred_len=job.pred_len,
            overwrite=args.overwrite,
            random_seed=args.random_seed,
            max_train_samples=args.max_train_samples,
            max_validation_samples=args.max_validation_samples,
            max_test_samples=args.max_test_samples,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=args.device,
            run_name=job.run_name,
            output_dir=job.output_dir(args.output_root),
            progress_every=args.progress_every,
            trace_every=args.trace_every,
        )
        rows.append(normalize_summary(run(config), job, "computed"))

    if not args.dry_run:
        summary_path = write_summary(args.output_root, rows)
        print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
