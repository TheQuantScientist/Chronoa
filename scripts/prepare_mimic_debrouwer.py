#!/usr/bin/env python3
"""Build the benchmark/tsdm MIMIC-III De Brouwer complete_tensor.csv.

This runner executes the official GRU-ODE-Bayes MIMIC preprocessing notebooks
headlessly, with two local patches:

1. Notebook paths are redirected to a local work directory and your raw
   MIMIC-III v1.4 dump.
2. Old pandas ``DataFrame.append`` calls are shimmed for pandas >= 2.

The benchmark loader expects the modified 30-minute binning used by tsdm benchmark, so the
DataMerging aggregation is run with ``bin_k=2`` and the resulting CSV is
validated before being copied into ~/.tsdm/rawdata/MIMIC_III_DeBrouwer2019.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable


NOTEBOOK_STEPS: tuple[tuple[str, int | None], ...] = (
    ("Admissions.ipynb", 44),
    ("Outputs.ipynb", 19),
    ("LabEvents.ipynb", 14),
    ("Prescriptions.ipynb", 15),
)

DATA_MERGING_CELLS: tuple[int, ...] = (
    1,
    2,
    3,
    5,
    6,
    7,
    9,
    10,
    12,
    22,
    26,
    28,
    29,
    30,
    32,
)

EXPECTED_SHAPE = (3_082_224, 7)
EXPECTED_COLUMNS = {
    "UNIQUE_ID",
    "TIME_STAMP",
    "LABEL_CODE",
    "VALUENUM",
    "VALUENORM",
    "MEAN",
    "STD",
}


def parse_args() -> argparse.Namespace:
    project = Path(__file__).resolve().parents[1]
    qa_root = project.parent
    return argparse.ArgumentParser(description=__doc__).parse_args(
        namespace=SimpleNamespace(
            repo=qa_root / "gru_ode_bayes",
            raw_dir=project / "vendor" / "upstream_benchmark" / "data" / "physionet.org" / "files" / "mimiciii" / "1.4",
            work_dir=qa_root / "mimic_debrouwer_work",
            target_dir=Path.home() / ".tsdm" / "rawdata" / "MIMIC_III_DeBrouwer2019",
            seed=432,
            force=False,
            allow_shape_mismatch=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    project = Path(__file__).resolve().parents[1]
    qa_root = project.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=qa_root / "gru_ode_bayes")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=project / "vendor" / "upstream_benchmark" / "data" / "physionet.org" / "files" / "mimiciii" / "1.4",
    )
    parser.add_argument("--work-dir", type=Path, default=qa_root / "mimic_debrouwer_work")
    parser.add_argument(
        "--target-dir",
        type=Path,
        default=Path.home() / ".tsdm" / "rawdata" / "MIMIC_III_DeBrouwer2019",
    )
    parser.add_argument("--seed", type=int, default=432)
    parser.add_argument("--force", action="store_true", help="Rebuild even when processed intermediates exist.")
    parser.add_argument(
        "--allow-shape-mismatch",
        action="store_true",
        help="Copy the CSV even if it does not match the benchmark/tsdm expected shape.",
    )
    return parser


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def install_pandas_compat(pd) -> None:
    if hasattr(pd.DataFrame, "append"):
        return

    def append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        frames = [self]
        if isinstance(other, list):
            frames.extend(other)
        else:
            frames.append(other)
        return pd.concat(
            frames,
            ignore_index=ignore_index,
            verify_integrity=verify_integrity,
            sort=sort,
        )

    pd.DataFrame.append = append


def notebook_code_cells(path: Path) -> list[tuple[int, str]]:
    with path.open("r", encoding="utf-8") as handle:
        notebook = json.load(handle)
    cells: list[tuple[int, str]] = []
    for idx, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            if source.strip():
                cells.append((idx, source))
    return cells


def patch_source(source: str, work_dir: Path, clean_dir: Path, *, seed: int) -> str:
    work = f"{work_dir.as_posix().rstrip('/')}/"
    clean = f"{clean_dir.as_posix().rstrip('/')}/"
    source = source.replace('file_path="~/Documents/Data/Full_MIMIC/"', f'file_path="{work}"')
    source = source.replace('outfile_path="~/Documents/Data/Full_MIMIC/Clean_data/"', f'outfile_path="{clean}"')
    source = source.replace(
        'pd.to_datetime(patients_df["DOB"], format=\'%Y-%m-%d\')',
        'pd.to_datetime(patients_df["DOB"], format=\'%Y-%m-%d %H:%M:%S\')',
    )
    source = source.replace(
        'adm_2_15=adm_2.loc[((adm_2["ADMITTIME"]-adm_2["DOBTIME"]).dt.days/365)>15].copy()',
        'age_years = adm_2["ADMITTIME"].dt.year - adm_2["DOBTIME"].dt.year\n'
        'age_years -= ((adm_2["ADMITTIME"].dt.month < adm_2["DOBTIME"].dt.month) | '
        '((adm_2["ADMITTIME"].dt.month == adm_2["DOBTIME"].dt.month) & '
        '(adm_2["ADMITTIME"].dt.day < adm_2["DOBTIME"].dt.day))).astype(int)\n'
        'adm_2_15=adm_2.loc[age_years > 15].copy()',
    )
    source = source.replace(
        'pd.to_datetime(merged_df["CHARTTIME"], format=\'%Y-%m-%d %H:%M:%S\')',
        'pd.to_datetime(merged_df["CHARTTIME"], format=\'mixed\')',
    )
    source = source.replace(
        'death_tags_s=admissions.groupby("HADM_ID")["DEATHTAG"].unique().astype(int).to_frame().reset_index()',
        'death_tags_s=admissions.groupby("HADM_ID")["DEATHTAG"].first().astype(int).to_frame().reset_index()',
    )
    source = source.replace(
        'complete_df10=complete_df10.drop(complete_df10.loc[complete_df10["HADM_ID"].isin(id_list)].index).copy()',
        'complete_df10=complete_df10.drop(complete_df10.loc[complete_df10["HADM_ID"].isin(id_list)].index).copy()\ncomplete_df=complete_df10',
    )
    source = source.replace("np.random.shuffle(unique_ids)", f"np.random.seed({seed})\nnp.random.shuffle(unique_ids)")
    return source


def make_namespace(raw_dir: Path, work_dir: Path, clean_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    install_pandas_compat(pd)
    original_read_csv = pd.read_csv

    def read_csv_compat(path, *args, **kwargs):
        candidate = Path(os.path.expanduser(str(path)))
        if candidate.exists():
            return original_read_csv(candidate, *args, **kwargs)

        raw_candidate = raw_dir / candidate.name
        if raw_candidate.exists():
            return original_read_csv(raw_candidate, *args, **kwargs)

        gz_candidate = raw_dir / f"{candidate.name}.gz"
        if gz_candidate.exists():
            return original_read_csv(gz_candidate, *args, **kwargs)

        raise FileNotFoundError(f"{candidate} (also tried {raw_candidate} and {gz_candidate})")

    pd.read_csv = read_csv_compat
    plt.show = lambda *args, **kwargs: None

    ns = {
        "__name__": "__mimic_debrouwer_notebook__",
        "pd": pd,
        "plt": plt,
        "file_path": f"{work_dir.as_posix().rstrip('/')}/",
        "outfile_path": f"{clean_dir.as_posix().rstrip('/')}/",
    }
    return ns


def run_cells(
    notebook_path: Path,
    ns: dict,
    work_dir: Path,
    clean_dir: Path,
    *,
    seed: int,
    keep_indices: Iterable[int] | None = None,
    max_index: int | None = None,
) -> None:
    keep = set(keep_indices) if keep_indices is not None else None
    for idx, source in notebook_code_cells(notebook_path):
        if keep is not None and idx not in keep:
            continue
        if max_index is not None and idx > max_index:
            continue
        patched = patch_source(source, work_dir, clean_dir, seed=seed)
        log(f"running {notebook_path.name} cell {idx}")
        try:
            exec(compile(patched, f"{notebook_path}:{idx}", "exec"), ns)
        except Exception as exc:
            log(f"FAILED in {notebook_path.name} cell {idx}: {exc}")
            raise


def ensure_inputs(raw_dir: Path) -> None:
    required = [
        "ADMISSIONS.csv.gz",
        "PATIENTS.csv.gz",
        "INPUTEVENTS_MV.csv.gz",
        "D_ITEMS.csv.gz",
        "OUTPUTEVENTS.csv.gz",
        "LABEVENTS.csv.gz",
        "D_LABITEMS.csv.gz",
        "DIAGNOSES_ICD.csv.gz",
        "PRESCRIPTIONS.csv.gz",
    ]
    missing = [name for name in required if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing raw MIMIC files in {raw_dir}: {', '.join(missing)}")


def maybe_run_processing(args: argparse.Namespace, ns: dict, mimic_dir: Path, clean_dir: Path) -> None:
    outputs = [
        args.work_dir / "Admissions_processed.csv",
        args.work_dir / "INPUTS_processed.csv",
        args.work_dir / "OUTPUTS_processed.csv",
        args.work_dir / "LAB_processed.csv",
        args.work_dir / "PRESCRIPTIONS_processed.csv",
    ]
    if not args.force and all(path.exists() for path in outputs):
        log("processed intermediates already exist; use --force to rebuild them")
        return

    for notebook_name, max_index in NOTEBOOK_STEPS:
        run_cells(
            mimic_dir / notebook_name,
            ns,
            args.work_dir,
            clean_dir,
            seed=args.seed,
            max_index=max_index,
        )


def run_data_merging(args: argparse.Namespace, ns: dict, mimic_dir: Path, clean_dir: Path) -> Path:
    complete_path = clean_dir / "complete_tensor.csv"
    if args.force and complete_path.exists():
        complete_path.unlink()

    run_cells(
        mimic_dir / "DataMerging.ipynb",
        ns,
        args.work_dir,
        clean_dir,
        seed=args.seed,
        keep_indices=DATA_MERGING_CELLS,
    )

    import pandas as pd

    complete_tensor = ns["complete_tensor_nocov"]
    complete_tensor.to_csv(complete_path)
    log(f"wrote {complete_path}")

    if "complete_tensor" in ns and "hot_encodings" in ns:
        covariates = ns["complete_tensor"].groupby("UNIQUE_ID").nth(0)[list(ns["hot_encodings"].columns)]
        covariates.to_csv(clean_dir / "complete_covariates.csv")
        log(f"wrote {clean_dir / 'complete_covariates.csv'}")

    if "d" in ns:
        unique_id_df = pd.DataFrame(
            {"HADM_ID": list(ns["d"].keys()), "UNIQUE_ID": list(ns["d"].values())}
        )
        unique_id_df.to_csv(clean_dir / "UNIQUE_ID_dict.csv", index=False)
        log(f"wrote {clean_dir / 'UNIQUE_ID_dict.csv'}")

    return complete_path


def validate_and_install(path: Path, target_dir: Path, *, allow_shape_mismatch: bool) -> None:
    import pandas as pd

    df = pd.read_csv(path, index_col=0)
    columns = set(df.columns)
    log(f"complete_tensor shape: {df.shape}")
    log(f"complete_tensor columns: {', '.join(df.columns)}")
    missing = sorted(EXPECTED_COLUMNS - columns)
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")

    if df.shape != EXPECTED_SHAPE and not allow_shape_mismatch:
        raise ValueError(
            f"Shape {df.shape} does not match benchmark/tsdm expected {EXPECTED_SHAPE}. "
            "Not installing into ~/.tsdm. Re-run with --allow-shape-mismatch only for debugging."
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "complete_tensor.csv"
    shutil.copy2(path, target)
    log(f"installed {target}")


def main() -> int:
    raise SystemExit(
        "Deprecated: use scripts/run_pyomnits_mimic_preprocess.sh instead. "
        "The maintained PyOmniTS preprocessing script includes checksum validation "
        "for benchmark/tsdm MIMIC_III_DeBrouwer2019."
    )


def _old_main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    args.repo = args.repo.expanduser().resolve()
    args.raw_dir = args.raw_dir.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.target_dir = args.target_dir.expanduser().resolve()

    mimic_dir = args.repo / "data_preproc" / "MIMIC"
    clean_dir = args.work_dir / "Clean_data"
    if not mimic_dir.exists():
        parser.error(f"MIMIC preprocessing notebooks not found: {mimic_dir}")

    ensure_inputs(args.raw_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)

    ns = make_namespace(args.raw_dir, args.work_dir, clean_dir)
    maybe_run_processing(args, ns, mimic_dir, clean_dir)
    complete_path = run_data_merging(args, ns, mimic_dir, clean_dir)
    validate_and_install(
        complete_path,
        args.target_dir,
        allow_shape_mismatch=args.allow_shape_mismatch,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
