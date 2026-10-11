#!/usr/bin/env python3
"""Build compact ADEPT architecture, perturbation, and sweep tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "adept_experiments" / "tables"
METRIC_COLUMNS = {
    "global": ("MAE_scaled", "MSE_scaled"),
    "equal-variable": ("Equal_variable_MAE_scaled", "Equal_variable_MSE_scaled"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ADEPT experiment tables from a summary CSV.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--kind", choices=["architecture", "temporal", "sweep"], required=True)
    parser.add_argument("--metric-mode", choices=sorted(METRIC_COLUMNS), default="global")
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {sorted(missing)}")


def load_summary(summary_path: Path, metric_mode: str) -> pd.DataFrame:
    mae_col, mse_col = METRIC_COLUMNS[metric_mode]
    frame = pd.read_csv(summary_path)
    require_columns(frame, {"Dataset", "Method", mae_col, mse_col, "Predictions"})
    frame = frame.copy()
    frame["MAE"] = pd.to_numeric(frame[mae_col], errors="coerce")
    frame["MSE"] = pd.to_numeric(frame[mse_col], errors="coerce")
    frame["Predictions"] = pd.to_numeric(frame["Predictions"], errors="coerce")
    frame = frame.dropna(subset=["MAE", "MSE", "Predictions"])
    frame = frame.loc[frame["Predictions"] > 0].copy()
    defaults = {
        "Run_tag": "full",
        "Variant": "default",
        "Seq_len": np.nan,
        "Pred_len": np.nan,
        "Architecture_ablation": "full",
        "History_perturbation": "original",
        "History_keep_fraction": 1.0,
    }
    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default
    frame["Run_tag"] = frame["Run_tag"].fillna(frame["Variant"]).replace("", "full")
    frame["Architecture_ablation"] = frame["Architecture_ablation"].fillna("full").replace("", "full")
    frame["History_perturbation"] = frame["History_perturbation"].fillna("original")
    return frame


def baseline_lookup(frame: pd.DataFrame, kind: str) -> dict[str, pd.Series]:
    baselines: dict[str, pd.Series] = {}
    for dataset, group in frame.groupby("Dataset", sort=False):
        if kind == "architecture":
            mask = group["Architecture_ablation"].eq("full") | group["Run_tag"].isin(["full", "default", "standard"])
        elif kind == "temporal":
            mask = group["History_perturbation"].eq("original") & pd.to_numeric(group["History_keep_fraction"], errors="coerce").fillna(1.0).eq(1.0)
        else:
            defaults = group.loc[group["Run_tag"].astype(str).str.contains("standard|sl", regex=True, na=False)]
            mask = group.index.isin(defaults.index[:1])
            if not mask.any():
                mask = group.index == group.index[0]
        if mask.any():
            baselines[str(dataset)] = group.loc[mask].iloc[0]
    return baselines


def label_row(row: pd.Series, kind: str) -> str:
    if kind == "architecture":
        ablation = str(row.get("Architecture_ablation", "full") or "full")
        if ablation == "full":
            return "full ADEPT"
        return ablation.replace("_", " ")
    if kind == "temporal":
        keep_fraction = float(row.get("History_keep_fraction", 1.0) or 1.0)
        if keep_fraction < 1.0 and str(row.get("History_perturbation", "original")) == "original":
            return f"keep {int(round(keep_fraction * 100))}% history"
        return str(row.get("History_perturbation", "original"))
    return f"seq={int(row['Seq_len'])}, pred={int(row['Pred_len'])}"


def experiment_table(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    baselines = baseline_lookup(frame, kind)
    rows: list[dict[str, object]] = []
    for _, row in frame.iterrows():
        dataset = str(row["Dataset"])
        baseline = baselines.get(dataset)
        if baseline is None:
            continue
        rows.append(
            {
                "Dataset": dataset,
                "Condition": label_row(row, kind),
                "Run_tag": row.get("Run_tag", ""),
                "Seq_len": row.get("Seq_len", np.nan),
                "Pred_len": row.get("Pred_len", np.nan),
                "MAE": float(row["MAE"]),
                "MSE": float(row["MSE"]),
                "Delta_MAE_vs_full": float(row["MAE"] - baseline["MAE"]),
                "Delta_MSE_vs_full": float(row["MSE"] - baseline["MSE"]),
                "Predictions": int(round(float(row["Predictions"]))),
            }
        )
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame, kind: str, metric_mode: str) -> str:
    title = {
        "architecture": "ADEPT Neural Architecture Ablation",
        "temporal": "ADEPT Temporal-Information Test",
        "sweep": "ADEPT Lookback/Horizon Sweep",
    }[kind]
    lines = [
        f"# {title}",
        "",
        f"Metric mode: `{metric_mode}`. Lower is better. Deltas are row metric minus the full/default ADEPT baseline.",
        "",
        "| Dataset | Condition | MAE | Delta MAE | MSE | Delta MSE | Predictions |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            f"{row['Dataset']} | {row['Condition']} | "
            f"{row['MAE']:.4f} | {row['Delta_MAE_vs_full']:+.4f} | "
            f"{row['MSE']:.4f} | {row['Delta_MSE_vs_full']:+.4f} | "
            f"{int(row['Predictions'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    frame = experiment_table(load_summary(args.summary, args.metric_mode), args.kind)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"adept_{args.kind}_{args.metric_mode.replace('-', '_')}"
    csv_path = args.output_dir / f"{stem}.csv"
    md_path = args.output_dir / f"{stem}.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(markdown_table(frame, args.kind, args.metric_mode), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
