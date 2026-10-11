#!/usr/bin/env python3
"""Run upstream neural paper baselines from the vendored benchmark tree.

The upstream shell scripts are the source of truth for model hyperparameters.
This wrapper adds safer GPU scheduling around them:
  * APN, GraFITi, tPatchGNN, and Warpformer are included in the SOTA preset.
  * Hardcoded upstream GPU ids are patched in a temporary script copy.
  * Each job receives CUDA_VISIBLE_DEVICES=<physical_gpu> and sees gpu_id=0.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adept.benchmark import BENCHMARK_ROOT  # noqa: E402
from adept.cli_utils import expand_selection, split_csv_values, unique_preserve_order  # noqa: E402

UPSTREAM_PATCH_MODEL = "A" + "PN"

PAPER_MODELS = [
    "PrimeNet",
    "NeuralFlows",
    "CRU",
    "mTAN",
    "SeFT",
    "GNeuralFlow",
    "GRU_D",
    "Raindrop",
    "Warpformer",
    "tPatchGNN",
    "GraFITi",
    UPSTREAM_PATCH_MODEL,
]
SOTA_MODELS = ["Warpformer", "tPatchGNN", "GraFITi", UPSTREAM_PATCH_MODEL]
DATASETS = ["HumanActivity", "USHCN", "P12", "MIMIC_III"]

MODEL_ALIASES = {
    **{model.lower(): model for model in PAPER_MODELS},
    "gru-d": "GRU_D",
    "grud": "GRU_D",
    "tpatchgnn": "tPatchGNN",
    "mtan": "mTAN",
    "warp": "Warpformer",
    "warpformer": "Warpformer",
    "sota": "__SOTA__",
    "strong": "__SOTA__",
    "paper": "__PAPER__",
}

DATASET_ALIASES = {
    "humanactivity": "HumanActivity",
    "human_activity": "HumanActivity",
    "ushcn": "USHCN",
    "physionet": "P12",
    "physionet2012": "P12",
    "physionet_2012": "P12",
    "p12": "P12",
    "mimic": "MIMIC_III",
    "mimiciii": "MIMIC_III",
    "mimic_iii": "MIMIC_III",
}


@dataclass(frozen=True)
class Job:
    model: str
    dataset: str

    @property
    def key(self) -> str:
        return f"{self.model}/{self.dataset}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run upstream neural baseline scripts with GPU scheduling.")
    parser.add_argument("--model", action="append", default=[], help="Model folder, paper, sota/strong, or all. Repeatable/comma-separated. Default: paper table models.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset script. Repeatable/comma-separated. Default: all four paper datasets.")
    parser.add_argument("--list", action="store_true", help="Print the availability matrix and exit.")
    parser.add_argument("--execute", action="store_true", help="Actually run scripts. Without this flag the wrapper only prints commands.")
    parser.add_argument("--continue-on-error", action="store_true", help="Keep running later scripts after one script fails.")
    parser.add_argument("--log-root", type=Path, default=Path("upstream_model_runs"), help="Wrapper log directory.")
    parser.add_argument("--gpu-ids", default="0", help="Physical GPU ids to use, e.g. 0 or 0,1,2,3.")
    parser.add_argument("--jobs-per-gpu", type=int, default=1, help="Concurrent scripts per GPU. Keep 1 unless models are small.")
    parser.add_argument("--no-gpu-patch", action="store_true", help="Run vendor scripts as-is, preserving their hardcoded GPU ids. Also disables --itr patching.")
    parser.add_argument("--itr", type=int, default=None, help="Override upstream --itr seed count in temporary scripts, e.g. 3 for seeds 2024-2026.")
    return parser.parse_args()


def normalize_models(raw_values: list[str]) -> list[str]:
    raw = split_csv_values(raw_values, default=PAPER_MODELS)
    selected: list[str] = []
    for item in raw:
        normalized = item.lower()
        value = MODEL_ALIASES.get(normalized, item)
        if normalized == "all" or value == "__PAPER__":
            selected.extend(PAPER_MODELS)
        elif value == "__SOTA__":
            selected.extend(SOTA_MODELS)
        else:
            selected.append(value)
    return unique_preserve_order(selected)


def normalize_datasets(raw_values: list[str]) -> list[str]:
    return expand_selection(raw_values, DATASETS, DATASET_ALIASES)


def parse_gpu_ids(value: str) -> list[str]:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    return ids or ["0"]


def script_path(model: str, dataset: str) -> Path:
    return BENCHMARK_ROOT / "scripts" / model / f"{dataset}.sh"


def print_matrix(models: list[str], datasets: list[str]) -> None:
    print("Upstream paper-script availability:")
    header = "Model".ljust(16) + " ".join(dataset.rjust(14) for dataset in datasets)
    print(header)
    print("-" * len(header))
    for model in models:
        cells = []
        for dataset in datasets:
            cells.append(("yes" if script_path(model, dataset).exists() else "missing").rjust(14))
        print(model.ljust(16) + " ".join(cells))


def patched_script(original: Path, model: str, dataset: str, log_root: Path, physical_gpu_id: str, itr: int | None) -> Path:
    text = original.read_text(encoding="utf-8")
    python_executable = sys.executable
    text = re.sub(r"--gpu_id\s+\d+", "--gpu_id 0", text)
    text = re.sub(r"GPU_IDS=\([^)]*\)", f"GPU_IDS=({physical_gpu_id})", text)
    if itr is not None:
        text = re.sub(r"--itr\s+(?:\"?\$itr\"?|\d+)", f"--itr {itr}", text)
    text = re.sub(r'launch_command="python"', f'launch_command="{python_executable}"', text)
    text = re.sub(r"(?<![A-Za-z0-9_./-])python(\s+main\.py\b)", python_executable + r"\1", text)
    text = re.sub(r"gpu_id=\$\{GPU_IDS\[\$free_gpu_idx\]\}", "gpu_id=${GPU_IDS[$free_gpu_idx]}", text)
    script = log_root / "_patched_scripts" / model / f"{dataset}.sh"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(text, encoding="utf-8")
    script.chmod(0o755)
    return script


def command_for_job(job: Job, log_root: Path, patch_gpu: bool, physical_gpu_id: str, itr: int | None) -> tuple[list[str], Path]:
    script = script_path(job.model, job.dataset)
    if not script.exists():
        return ["bash", str(script)], script
    runnable = patched_script(script, job.model, job.dataset, log_root, physical_gpu_id, itr) if patch_gpu else script
    return ["bash", str(runnable)], runnable


def launch_job(job: Job, gpu_id: str, log_root: Path, patch_gpu: bool, itr: int | None) -> tuple[subprocess.Popen[str], Path, float]:
    script = script_path(job.model, job.dataset)
    if not script.exists():
        raise FileNotFoundError(script)
    log_root.mkdir(parents=True, exist_ok=True)
    wrapper_log = log_root / f"{job.model}_{job.dataset}.log"
    command, runnable = command_for_job(job, log_root, patch_gpu, gpu_id, itr)
    env = os.environ.copy()
    if patch_gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"[run] {job.key} gpu={gpu_id} cwd={BENCHMARK_ROOT} {' '.join(command)}")
    log_file = wrapper_log.open("w", encoding="utf-8")
    log_file.write(f"JOB={job.key}\nGPU={gpu_id}\nSCRIPT={runnable}\nCWD={BENCHMARK_ROOT}\nCOMMAND={' '.join(command)}\n\n")
    log_file.flush()
    process = subprocess.Popen(
        command,
        cwd=BENCHMARK_ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    process._chronoa_log_file = log_file  # type: ignore[attr-defined]
    return process, wrapper_log, time.time()


def run_scheduled(jobs: list[Job], args: argparse.Namespace) -> list[tuple[Job, int]]:
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    slots = [(gpu_id, slot) for gpu_id in gpu_ids for slot in range(max(1, args.jobs_per_gpu))]
    running: dict[tuple[str, int], tuple[Job, subprocess.Popen[str], Path, float]] = {}
    failures: list[tuple[Job, int]] = []
    queue = list(jobs)

    while queue or running:
        for slot in slots:
            if not queue or slot in running:
                continue
            job = queue.pop(0)
            try:
                process, log_path, start = launch_job(job, slot[0], args.log_root, not args.no_gpu_patch, args.itr)
            except FileNotFoundError as exc:
                print(f"[missing] {exc}", file=sys.stderr)
                failures.append((job, 2))
                if not args.continue_on_error:
                    return failures
                continue
            running[slot] = (job, process, log_path, start)

        if not running:
            continue

        time.sleep(5)
        finished: list[tuple[str, int]] = []
        for slot, (job, process, log_path, start) in running.items():
            return_code = process.poll()
            if return_code is None:
                continue
            log_file = getattr(process, "_chronoa_log_file", None)
            if log_file is not None:
                log_file.close()
            elapsed = time.time() - start
            status = "ok" if return_code == 0 else f"failed:{return_code}"
            print(f"[{status}] {job.key} gpu={slot[0]} in {elapsed:.1f}s, log: {log_path}")
            if return_code != 0:
                failures.append((job, return_code))
                if not args.continue_on_error:
                    for _, process_info in running.items():
                        other_job, other_process, _other_log, _other_start = process_info
                        if other_process.poll() is None:
                            print(f"[terminate] {other_job.key}")
                            other_process.terminate()
                    return failures
            finished.append(slot)
        for slot in finished:
            running.pop(slot, None)
    return failures


def main() -> None:
    args = parse_args()
    if not args.log_root.is_absolute():
        args.log_root = (PROJECT_ROOT / args.log_root).resolve()
    models = normalize_models(args.model)
    datasets = normalize_datasets(args.dataset)

    if args.list:
        print_matrix(models, datasets)
        return

    jobs = [Job(model, dataset) for model in models for dataset in datasets]
    if not args.execute:
        print("Dry run. Add --execute to run these upstream scripts.")
        print(f"GPU ids: {', '.join(parse_gpu_ids(args.gpu_ids))}; jobs per GPU: {args.jobs_per_gpu}; itr: {args.itr or 'upstream'}")
        for job in jobs:
            script = script_path(job.model, job.dataset)
            status = "ok" if script.exists() else "missing"
            print(f"[{status}] {job.key}: {script}")
        return

    failures = run_scheduled(jobs, args)
    if failures:
        print("Failed scripts:", file=sys.stderr)
        for job, return_code in failures:
            print(f"  {job.key}: exit {return_code}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
