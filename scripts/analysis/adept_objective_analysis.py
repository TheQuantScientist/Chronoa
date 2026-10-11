#!/usr/bin/env python3
"""Summarize ADEPT default-loss and MSE-only objective-control runs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ADEPT objective-control runs.")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing chronoa_summary.csv files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for summary CSVs.")
    return parser.parse_args()


def parse_seed(path: Path) -> int | None:
    for part in path.parts:
        match = re.search(r"seed(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def load_summaries(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.rglob("chronoa_summary.csv")):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["Seed"] = parse_seed(path)
        frame["Source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No non-empty chronoa_summary.csv files found under {root}")
    frame = pd.concat(frames, ignore_index=True)
    if "Loss_mode" not in frame.columns:
        frame["Loss_mode"] = "default"
    frame["Loss_mode"] = frame["Loss_mode"].fillna("default").replace("", "default")
    return frame


def aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ["MAE_scaled", "MSE_scaled", "Predictions"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = (
        frame.dropna(subset=["MAE_scaled", "MSE_scaled"])
        .groupby(["Dataset", "Architecture_ablation", "Loss_mode"], dropna=False)
        .agg(
            Seeds=("Seed", "nunique"),
            Runs=("Dataset", "size"),
            MAE_mean=("MAE_scaled", "mean"),
            MAE_std=("MAE_scaled", "std"),
            MSE_mean=("MSE_scaled", "mean"),
            MSE_std=("MSE_scaled", "std"),
            Predictions_mean=("Predictions", "mean"),
        )
        .reset_index()
    )
    numeric_cols = grouped.select_dtypes(include="number").columns
    grouped[numeric_cols] = grouped[numeric_cols].round(6)
    return grouped


def default_vs_mse(summary: pd.DataFrame) -> pd.DataFrame:
    apex = summary.loc[summary["Architecture_ablation"].eq("apex")].copy()
    default = apex.loc[apex["Loss_mode"].eq("default")].set_index("Dataset")
    mse = apex.loc[apex["Loss_mode"].eq("mse")].set_index("Dataset")
    rows: list[dict[str, object]] = []
    for dataset in sorted(set(default.index) & set(mse.index)):
        rows.append(
            {
                "Dataset": dataset,
                "Default_MAE_mean": default.loc[dataset, "MAE_mean"],
                "MSE_only_MAE_mean": mse.loc[dataset, "MAE_mean"],
                "Delta_MAE_mse_minus_default": round(float(mse.loc[dataset, "MAE_mean"] - default.loc[dataset, "MAE_mean"]), 6),
                "Default_MSE_mean": default.loc[dataset, "MSE_mean"],
                "MSE_only_MSE_mean": mse.loc[dataset, "MSE_mean"],
                "Delta_MSE_mse_minus_default": round(float(mse.loc[dataset, "MSE_mean"] - default.loc[dataset, "MSE_mean"]), 6),
                "Default_seeds": int(default.loc[dataset, "Seeds"]),
                "MSE_only_seeds": int(mse.loc[dataset, "Seeds"]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.root / "objective_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_runs = load_summaries(args.root)
    all_runs.to_csv(output_dir / "adept_objective_runs.csv", index=False)
    summary = aggregate(all_runs)
    summary.to_csv(output_dir / "adept_objective_summary.csv", index=False)
    default_vs_mse(summary).to_csv(output_dir / "adept_objective_default_vs_mse.csv", index=False)
    print(f"Objective summaries written to {output_dir}")


if __name__ == "__main__":
    main()
