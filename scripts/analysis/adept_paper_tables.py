#!/usr/bin/env python3
"""Generate ADEPT paper tables from retained result exports.

The script intentionally uses only the Python standard library because the
paper workspace may not have NumPy/Pandas installed.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path


DATASET_ORDER = ["P12", "MIMIC", "USHCN", "HumanActivity"]
DATASET_SLUG = {
    "p12": "P12",
    "mimic": "MIMIC",
    "ushcn": "USHCN",
    "humanactivity": "HumanActivity",
}
TEMPORAL_LABELS = {
    "original": "Original",
    "shuffle_timestamps": "Shuffle",
    "swap_halves": "Swap",
    "keep_time_gaps": "Gaps",
}
BASELINE_CONDITIONS = {
    "original": "original",
    "shuffle_timestamps": "shuffle timestamps",
    "swap_halves": "swap halves",
    "keep_time_gaps": "keep time gaps",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adept-root",
        type=Path,
        default=Path("Chronoa/results/adept_seeded_parallel12"),
    )
    parser.add_argument(
        "--baseline-temporal",
        type=Path,
        default=Path("ChronoLM/cross_baseline_results/ablation/cross_baseline_temporal.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Chronoa/results/adept_seeded_parallel12/tables"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261005)
    return parser.parse_args()


def first_float(value: object) -> float:
    match = re.search(r"[-+]?\d+(?:\.\d+)?", str(value))
    if match is None:
        return float("nan")
    return float(match.group(0))


def fmt(value: float, digits: int = 4, signed: bool = False) -> str:
    sign = "+" if signed else ""
    return f"{value:{sign}.{digits}f}"


def ci_text(point: float, low: float, high: float) -> str:
    return f"{point:.4f} [{low:.4f}, {high:.4f}]"


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def bootstrap_ratio(
    numerator: list[float],
    denominator: list[float],
    rng: random.Random,
    reps: int,
) -> tuple[float, float, float]:
    point = float(sum(numerator) / sum(denominator))
    n_units = len(numerator)
    indices = range(n_units)
    draws: list[float] = []
    for _ in range(reps):
        num = 0.0
        den = 0.0
        for sampled in rng.choices(indices, k=n_units):
            num += numerator[sampled]
            den += denominator[sampled]
        draws.append(num / den)
    return point, percentile(draws, 0.025), percentile(draws, 0.975)


def detail_paths(root: Path, suite: str) -> list[Path]:
    return sorted(root.glob(f"seed*_{suite}_*/*/*DetailLog.csv"))


def parse_run_dir(path: Path) -> tuple[int, str, str]:
    run_dir = ""
    for parent in [path.parent, *path.parents]:
        if re.match(r"seed\d+_", parent.name):
            run_dir = parent.name
            break
    match = re.match(r"seed(\d+)_(standard|arch|temporal|window)_([^_]+)(?:_(.*))?", run_dir)
    if match is None:
        raise ValueError(f"Cannot parse run directory: {run_dir}")
    seed = int(match.group(1))
    dataset = DATASET_SLUG[match.group(3)]
    suffix = match.group(4) or ""
    return seed, dataset, suffix


def read_single_row_csv(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        return next(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def entity_bootstrap_table(root: Path, rng: random.Random, reps: int) -> list[dict[str, object]]:
    units: dict[tuple[str, int, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    for path in detail_paths(root, "standard"):
        seed, dataset, _ = parse_run_dir(path)
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                unit = units[(dataset, seed, row["Entity"])]
                unit[0] += float(row["AE_scaled"])
                unit[1] += float(row["SE_scaled"])
                unit[2] += 1.0
    table_rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        group = [value for key, value in units.items() if key[0] == dataset]
        mae = bootstrap_ratio([row[0] for row in group], [row[2] for row in group], rng, reps)
        mse = bootstrap_ratio([row[1] for row in group], [row[2] for row in group], rng, reps)
        table_rows.append(
            {
                "Dataset": dataset,
                "Seed_entity_units": len(group),
                "Predictions": int(sum(row[2] for row in group)),
                "MAE": mae[0],
                "MAE_CI_low": mae[1],
                "MAE_CI_high": mae[2],
                "MSE": mse[0],
                "MSE_CI_low": mse[1],
                "MSE_CI_high": mse[2],
            }
        )
    return table_rows


def read_error_by_key(path: Path, key_cols: list[str]) -> dict[tuple[str, ...], tuple[str, float, float]]:
    rows: dict[tuple[str, ...], tuple[str, float, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[column] for column in key_cols)
            rows[key] = (row["Entity"], float(row["AE_scaled"]), float(row["SE_scaled"]))
    return rows


def paired_decoder_table(root: Path, rng: random.Random, reps: int) -> list[dict[str, object]]:
    table_rows: list[dict[str, object]] = []
    key_cols = ["Entity", "Variable", "Step", "Target_time"]
    for slug, dataset in DATASET_SLUG.items():
        units: dict[tuple[int, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
        total_adept = 0
        total_matched = 0
        for seed in [2024, 2025, 2026]:
            adept_paths = list(root.glob(f"seed{seed}_arch_{slug}_apex/*/*DetailLog.csv"))
            no_expert_paths = list(root.glob(f"seed{seed}_arch_{slug}_full/*/*DetailLog.csv"))
            if not adept_paths or not no_expert_paths:
                continue
            adept = read_error_by_key(adept_paths[0], key_cols)
            no_expert = read_error_by_key(no_expert_paths[0], key_cols)
            total_adept += len(adept)
            for key, (_, adept_ae, adept_se) in adept.items():
                if key not in no_expert:
                    continue
                entity, no_expert_ae, no_expert_se = no_expert[key]
                unit = units[(seed, entity)]
                unit[0] += no_expert_ae - adept_ae
                unit[1] += no_expert_se - adept_se
                unit[2] += 1.0
                total_matched += 1
        group = list(units.values())
        mae = bootstrap_ratio([row[0] for row in group], [row[2] for row in group], rng, reps)
        mse = bootstrap_ratio([row[1] for row in group], [row[2] for row in group], rng, reps)
        entity_mae_delta = [row[0] / row[2] for row in group]
        table_rows.append(
            {
                "Dataset": dataset,
                "Paired_entity_units": len(group),
                "Coordinate_coverage": float(total_matched / total_adept),
                "Delta_MAE": mae[0],
                "Delta_MAE_CI_low": mae[1],
                "Delta_MAE_CI_high": mae[2],
                "Delta_MSE": mse[0],
                "Delta_MSE_CI_low": mse[1],
                "Delta_MSE_CI_high": mse[2],
                "MAE_entity_win_pct": 100.0 * sum(value > 0 for value in entity_mae_delta) / len(entity_mae_delta),
            }
        )
    order = {dataset: idx for idx, dataset in enumerate(DATASET_ORDER)}
    return sorted(table_rows, key=lambda row: order[row["Dataset"]])


def adept_temporal_summary(root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(root.glob("seed*_temporal_*/*summary.csv")):
        seed, dataset, suffix = parse_run_dir(path)
        if suffix not in TEMPORAL_LABELS:
            continue
        row = read_single_row_csv(path)
        rows.append(
            {
                "Dataset": dataset,
                "Model": "ADEPT",
                "Condition": suffix,
                "Seed": seed,
                "MSE": float(row["MSE_scaled"]),
                "MAE": float(row["MAE_scaled"]),
            }
        )
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["Dataset"], row["Model"], row["Condition"])].append(row)
    return [
        {
            "Dataset": key[0],
            "Model": key[1],
            "Condition": key[2],
            "MSE": sum(float(row["MSE"]) for row in group) / len(group),
            "MAE": sum(float(row["MAE"]) for row in group) / len(group),
            "Runs": len({row["Seed"] for row in group}),
        }
        for key, group in grouped.items()
    ]


def cross_model_temporal_table(root: Path, baseline_csv: Path) -> list[dict[str, object]]:
    reverse_conditions = {value: key for key, value in BASELINE_CONDITIONS.items()}
    baseline: list[dict[str, object]] = []
    with baseline_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Model"] not in {"APN", "GraFITi", "tPatchGNN"}:
                continue
            condition = reverse_conditions.get(row["ConditionLabel"])
            if condition is None:
                continue
            baseline.append(
                {
                    "Dataset": row["Dataset"],
                    "Model": row["Model"],
                    "Condition": condition,
                    "MSE": first_float(row["MSE"]),
                    "MAE": first_float(row["MAE"]),
                    "Runs": int(row["Runs"]),
                }
            )
    combined = adept_temporal_summary(root) + baseline

    table_rows: list[dict[str, object]] = []
    for dataset in DATASET_ORDER:
        for model in ["ADEPT", "APN", "GraFITi", "tPatchGNN"]:
            group = [row for row in combined if row["Dataset"] == dataset and row["Model"] == model]
            if not group:
                continue
            by_condition = {row["Condition"]: row for row in group}
            original = by_condition["original"]
            row = {
                "Dataset": dataset,
                "Model": model,
                "Orig_MSE": float(original["MSE"]),
                "Orig_MAE": float(original["MAE"]),
                "Runs": int(original["Runs"]),
            }
            for condition in ["shuffle_timestamps", "swap_halves", "keep_time_gaps"]:
                current = by_condition[condition]
                row[f"{condition}_Delta_MSE"] = float(current["MSE"] - original["MSE"])
                row[f"{condition}_Delta_MAE"] = float(current["MAE"] - original["MAE"])
            table_rows.append(row)
    return table_rows


def write_entity_bootstrap_tex(frame: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{Entity-bootstrap uncertainty for \method{} at the standard benchmark windows. Point estimates are target-weighted global scaled errors; 95\% intervals resample seed--entity units with replacement.}",
        r"\label{tab:entity-bootstrap}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{l|r|r|c|c}",
        r"\toprule",
        r"Dataset & Seed--entity units & Predictions & MAE [95\% CI] & MSE [95\% CI] \\",
        r"\midrule",
    ]
    for row in frame:
        lines.append(
            f"{row['Dataset']} & {int(row['Seed_entity_units']):,} & "
            f"{int(row['Predictions']):,} & "
            f"{ci_text(row['MAE'], row['MAE_CI_low'], row['MAE_CI_high'])} & "
            f"{ci_text(row['MSE'], row['MSE_CI_low'], row['MSE_CI_high'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_paired_decoder_tex(frame: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table}[!htbp]",
        r"\caption{Paired entity-level comparison between \method{} and the no-adaptive-expert decoder diagnostic. Positive deltas mean the no-adaptive-expert variant has higher error than \method{}. Confidence intervals resample paired seed--entity units.}",
        r"\label{tab:paired-decoder-entity}",
        r"\centering",
        r"\scriptsize",
        r"\begin{tabular}{l|r|c|c|c|c}",
        r"\toprule",
        r"Dataset & Paired units & Coverage & $\Delta$MAE [95\% CI] & $\Delta$MSE [95\% CI] & MAE win \% \\",
        r"\midrule",
    ]
    for row in frame:
        lines.append(
            f"{row['Dataset']} & {int(row['Paired_entity_units']):,} & "
            f"{100.0 * row['Coordinate_coverage']:.1f}\\% & "
            f"{ci_text(row['Delta_MAE'], row['Delta_MAE_CI_low'], row['Delta_MAE_CI_high'])} & "
            f"{ci_text(row['Delta_MSE'], row['Delta_MSE_CI_low'], row['Delta_MSE_CI_high'])} & "
            f"{row['MAE_entity_win_pct']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def write_cross_model_temporal_tex(frame: list[dict[str, object]]) -> str:
    lines = [
        r"\begin{table*}[!htbp]",
        r"\caption{Cross-model temporal perturbation summary at the standard benchmark windows. Deltas are relative to each model's original-history evaluation under the same window and are aggregate, not paired, comparisons.}",
        r"\label{tab:cross-model-temporal}",
        r"\centering",
        r"\scriptsize",
        r"\resizebox{\textwidth}{!}{",
        r"\begin{tabular}{l|l|r|r|r|r|r|r}",
        r"\toprule",
        r"Dataset & Model & Orig. MSE & Shuffle $\Delta$MSE & Swap $\Delta$MSE & Gaps $\Delta$MSE & Orig. MAE & Gaps $\Delta$MAE \\",
        r"\midrule",
    ]
    for dataset_idx, dataset in enumerate(DATASET_ORDER):
        if dataset_idx:
            lines.append(r"\midrule")
        for row in [row for row in frame if row["Dataset"] == dataset]:
            model = r"\method{}" if row["Model"] == "ADEPT" else row["Model"]
            lines.append(
                f"{row['Dataset']} & {model} & {fmt(row['Orig_MSE'])} & "
                f"{fmt(row['shuffle_timestamps_Delta_MSE'], signed=True)} & "
                f"{fmt(row['swap_halves_Delta_MSE'], signed=True)} & "
                f"{fmt(row['keep_time_gaps_Delta_MSE'], signed=True)} & "
                f"{fmt(row['Orig_MAE'])} & "
                f"{fmt(row['keep_time_gaps_Delta_MAE'], signed=True)} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    entity_bootstrap = entity_bootstrap_table(args.adept_root, rng, args.bootstrap_reps)
    paired_decoder = paired_decoder_table(args.adept_root, rng, args.bootstrap_reps)
    cross_model_temporal = cross_model_temporal_table(args.adept_root, args.baseline_temporal)

    outputs = [
        ("adept_entity_bootstrap", entity_bootstrap, write_entity_bootstrap_tex(entity_bootstrap)),
        ("adept_paired_decoder_entity", paired_decoder, write_paired_decoder_tex(paired_decoder)),
        ("adept_cross_model_temporal", cross_model_temporal, write_cross_model_temporal_tex(cross_model_temporal)),
    ]
    for stem, frame, tex in outputs:
        write_csv(args.output_dir / f"{stem}.csv", frame)
        (args.output_dir / f"{stem}.tex").write_text(tex, encoding="utf-8")
        print(f"Wrote {args.output_dir / f'{stem}.csv'}")
        print(f"Wrote {args.output_dir / f'{stem}.tex'}")


if __name__ == "__main__":
    main()
