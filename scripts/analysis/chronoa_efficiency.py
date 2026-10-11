#!/usr/bin/env python3
"""Generate ADEPT efficiency tables from completed summary CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "adept_experiments" / "tables"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ADEPT throughput tables from a summary CSV.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def efficiency_frame(summary_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_path)
    required = {"Dataset", "Method", "Predictions", "Fallback", "Time_s"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns in summary CSV: {sorted(missing)}")
    frame = frame.copy()
    frame["Predictions"] = pd.to_numeric(frame["Predictions"], errors="coerce")
    frame["Fallback"] = pd.to_numeric(frame["Fallback"], errors="coerce")
    frame["Wall_Total_s"] = pd.to_numeric(frame["Time_s"], errors="coerce")
    frame = frame.dropna(subset=["Predictions", "Wall_Total_s"])
    frame = frame.loc[frame["Predictions"] > 0].copy()
    frame["ms_per_pred"] = 1000.0 * frame["Wall_Total_s"] / frame["Predictions"]
    frame["pred_per_s"] = frame["Predictions"] / frame["Wall_Total_s"]
    return frame.sort_values(["Dataset", "Method"])


def markdown_efficiency(frame: pd.DataFrame) -> str:
    lines = [
        "# ADEPT Efficiency",
        "",
        "Wall time includes calibration plus forecasting for each completed run.",
        "",
        "| Dataset | Method | Run tag | Predictions | Fallback | Wall total (s) | ms/pred | pred/s |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in frame.iterrows():
        lines.append(
            "| "
            f"{row['Dataset']} | {row['Method']} | {row.get('Run_tag', '')} | "
            f"{int(round(row['Predictions']))} | {int(round(row['Fallback']))} | "
            f"{row['Wall_Total_s']:.4f} | {row['ms_per_pred']:.4f} | {row['pred_per_s']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    frame = efficiency_frame(args.summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "adept_efficiency.csv"
    md_path = args.output_dir / "adept_efficiency.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(markdown_efficiency(frame), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
