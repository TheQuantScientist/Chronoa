#!/usr/bin/env python3
"""Summarize ADEPT gate logs produced by run_chronoa_forecasting.py."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adept.experiments.chronoa_forecasting import GATE_NAMES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ADEPT gate logs.")
    parser.add_argument("--root", type=Path, required=True, help="Directory containing *GateLog.csv files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for summary CSVs.")
    return parser.parse_args()


def load_gate_logs(root: Path) -> pd.DataFrame:
    files = sorted(root.rglob("*GateLog.csv"))
    if not files:
        raise FileNotFoundError(f"No *GateLog.csv files found under {root}")
    frames: list[pd.DataFrame] = []
    for path in files:
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["Source_file"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError(f"All gate logs under {root} were empty.")
    return pd.concat(frames, ignore_index=True)


def add_grouping_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    gap = frame["Query_gap_normalized"].abs()
    frame["Gap_bin"] = pd.cut(
        gap,
        bins=[-float("inf"), 0.05, 0.25, 0.50, 1.00, float("inf")],
        labels=["<=0.05", "0.05-0.25", "0.25-0.50", "0.50-1.00", ">1.00"],
        include_lowest=True,
    )
    for name in GATE_NAMES:
        frame[f"Top_{name}"] = frame["Gate_top"].eq(name).astype(float)
    return frame


def summarize(frame: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    gate_cols = [f"Gate_{name}" for name in GATE_NAMES]
    top_cols = [f"Top_{name}" for name in GATE_NAMES]
    agg_spec = {
        "Rows": ("Gate_top", "size"),
        "MAE_scaled": ("AE_scaled", "mean"),
        "MSE_scaled": ("SE_scaled", "mean"),
        "Same_variable_available_rate": ("Same_variable_available", "mean"),
        "Query_gap_normalized": ("Query_gap_normalized", "mean"),
        "Gate_entropy": ("Gate_entropy", "mean"),
    }
    agg_spec.update({column: (column, "mean") for column in gate_cols + top_cols})
    summary = frame.groupby(group_cols, dropna=False, observed=False).agg(**agg_spec).reset_index()
    numeric_cols = summary.select_dtypes(include="number").columns
    summary[numeric_cols] = summary[numeric_cols].round(6)
    return summary


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.root / "gate_summaries"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame = add_grouping_columns(load_gate_logs(args.root))
    frame.to_csv(output_dir / "adept_gate_log_all.csv", index=False)
    summarize(frame, ["Dataset"]).to_csv(output_dir / "adept_gate_by_dataset.csv", index=False)
    summarize(frame, ["Dataset", "Same_variable_available"]).to_csv(
        output_dir / "adept_gate_by_same_variable_availability.csv",
        index=False,
    )
    summarize(frame, ["Dataset", "Gap_bin"]).to_csv(output_dir / "adept_gate_by_gap_bin.csv", index=False)
    summarize(frame, ["Dataset", "Variable"]).to_csv(output_dir / "adept_gate_by_variable.csv", index=False)
    print(f"Gate summaries written to {output_dir}")


if __name__ == "__main__":
    main()
