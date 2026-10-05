"""Compute benchmark-style global metrics from a Chronoa detail log."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute global masked MAE/MSE from a Chronoa detail CSV."
    )
    parser.add_argument(
        "detail_csv",
        type=Path,
        help="Path to a *_DetailLog.csv file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detail = pd.read_csv(args.detail_csv)

    required = {"Variable", "Actual_scaled", "Predicted_scaled"}
    missing = required - set(detail.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    valid = detail[["Actual_scaled", "Predicted_scaled"]].notna().all(axis=1)
    detail = detail.loc[valid].copy()
    detail["AE_scaled"] = (detail["Actual_scaled"] - detail["Predicted_scaled"]).abs()
    detail["SE_scaled"] = (detail["Actual_scaled"] - detail["Predicted_scaled"]) ** 2

    global_mae = float(detail["AE_scaled"].mean())
    global_mse = float(detail["SE_scaled"].mean())

    per_variable = (
        detail.groupby("Variable", sort=False)
        .agg(
            Count=("AE_scaled", "size"),
            MAE_scaled=("AE_scaled", "mean"),
            MSE_scaled=("SE_scaled", "mean"),
        )
        .reset_index()
    )

    variable_avg_mae = float(per_variable["MAE_scaled"].mean())
    variable_avg_mse = float(per_variable["MSE_scaled"].mean())

    print(f"Detail log: {args.detail_csv}")
    if "Dataset" in detail.columns:
        print(f"Dataset: {detail['Dataset'].iloc[0]}")
    print(f"Valid target points: {len(detail):,}")
    print()
    print("benchmark-style global masked metrics:")
    print(f"  MAE_scaled = {global_mae:.6f}")
    print(f"  MSE_scaled = {global_mse:.6f}")
    print()
    print("Equal-weight average across variables:")
    print(f"  MAE_scaled = {variable_avg_mae:.6f}")
    print(f"  MSE_scaled = {variable_avg_mse:.6f}")
    print()
    print("Per-variable metrics:")
    print(per_variable.to_string(index=False))


if __name__ == "__main__":
    main()
