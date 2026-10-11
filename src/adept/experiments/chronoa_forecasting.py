"""Chronoa neural forecasting on irregular time-series benchmark splits.

This module intentionally implements Chronoa as an end-to-end neural event/query
forecaster. The model consumes asynchronous observation events, builds latent
memory with a transformer encoder, mixes variables through an adaptive graph,
and decodes requested future (time, variable) queries with cross-attention plus
learned continuous-time delta kernels.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import math
import os
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from chronoa.benchmark import BENCHMARK_ROOT, add_benchmark_to_path

add_benchmark_to_path()

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - handled with a clear runtime error.
    torch = None
    nn = None


DATASET_DEFAULTS = {
    "P12": {"seq_len": 36, "pred_len": 3, "display_name": "PhysioNet P12"},
    "MIMIC": {"seq_len": 72, "pred_len": 3, "display_name": "MIMIC-III"},
    "USHCN": {"seq_len": 150, "pred_len": 3, "display_name": "USHCN"},
    "HumanActivity": {
        "seq_len": 3000,
        "pred_len": 300,
        "display_name": "HumanActivity",
    },
}

APN_BASELINES = {
    "HumanActivity": {"MAE": 0.1159, "MSE": 0.0421},
    "MIMIC": {"MAE": 0.4016, "MSE": 0.4292},
    "P12": {"MAE": 0.3650, "MSE": 0.3093},
    "USHCN": {"MAE": 0.2611, "MSE": 0.1590},
}

MODEL_NAME = "Chronoa-EDGE"
RUN_NAME_PREFIX = "chronoa"
HISTORY_PERTURBATIONS = (
    "original",
    "shuffle_timestamps",
    "swap_halves",
    "drop_early_history",
    "keep_last_obs",
    "keep_time_gaps",
)
ARCHITECTURE_ABLATIONS = (
    "apex",
    "full",
    "no_delta_kernel",
    "no_graph_mixer",
    "no_transformer",
    "no_time_encoding",
    "direct_decoder",
)
LOSS_MODES = (
    "default",
    "mse",
)
GATE_NAMES = ("neural", "same_last", "same_ema", "trend", "graph_ema", "global_ema")
SCALAR_EXPERT_NAMES = ("same_last", "same_ema", "trend", "graph_ema", "global_ema")


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") & 0xFFFFFFFF


def canonical_dataset_name(dataset_name: str) -> str:
    normalized = dataset_name.strip().replace("-", "_").upper()
    aliases = {
        "PHYSIONET": "P12",
        "PHYSIONET2012": "P12",
        "PHYSIONET_2012": "P12",
        "P12": "P12",
        "MIMIC": "MIMIC",
        "MIMICIII": "MIMIC",
        "MIMIC_III": "MIMIC",
        "MIMIC3": "MIMIC",
        "MIMIC_3": "MIMIC",
        "USHCN": "USHCN",
        "USHCN_DEBROUWER2019": "USHCN",
        "HUMANACTIVITY": "HumanActivity",
        "HUMAN_ACTIVITY": "HumanActivity",
        "PERSONACTIVITY": "HumanActivity",
        "PERSON_ACTIVITY": "HumanActivity",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_DEFAULTS))
        raise ValueError(f"Unsupported dataset {dataset_name!r}. Use one of: {supported}") from exc


def canonical_history_perturbation(name: str) -> str:
    normalized = name.strip().replace("-", "_").lower()
    if normalized in HISTORY_PERTURBATIONS:
        return normalized
    supported = ", ".join(HISTORY_PERTURBATIONS)
    raise ValueError(f"Unsupported history perturbation {name!r}. Use one of: {supported}")


def canonical_architecture_ablation(name: str) -> str:
    normalized = name.strip().replace("-", "_").replace(" ", "_").lower()
    aliases = {
        "apex": "apex",
        "chronoa_apex": "apex",
        "adaptive_expert": "apex",
        "adaptive_experts": "apex",
        "default": "full",
        "chronoa": "full",
        "full": "full",
        "edge": "full",
        "chronoa_edge": "full",
        "no_delta": "no_delta_kernel",
        "no_delta_kernel": "no_delta_kernel",
        "without_delta_kernel": "no_delta_kernel",
        "no_graph": "no_graph_mixer",
        "no_graph_mixer": "no_graph_mixer",
        "without_graph": "no_graph_mixer",
        "no_transformer": "no_transformer",
        "without_transformer": "no_transformer",
        "no_time": "no_time_encoding",
        "no_time_encoding": "no_time_encoding",
        "without_time_encoding": "no_time_encoding",
        "direct": "direct_decoder",
        "direct_decoder": "direct_decoder",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(ARCHITECTURE_ABLATIONS)
        raise ValueError(f"Unsupported architecture ablation {name!r}. Use one of: {supported}") from exc


def canonical_loss_mode(name: str) -> str:
    normalized = name.strip().replace("-", "_").replace(" ", "_").lower()
    aliases = {
        "default": "default",
        "mixed": "default",
        "released": "default",
        "paper": "default",
        "mse": "mse",
        "mse_only": "mse",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        supported = ", ".join(LOSS_MODES)
        raise ValueError(f"Unsupported loss mode {name!r}. Use one of: {supported}") from exc


@dataclass(frozen=True)
class ChronoaConfig:
    """Configuration for one Chronoa-EDGE neural benchmark run."""

    dataset_name: str = "P12"
    seq_len: int | None = None
    pred_len: int | None = None
    max_test_samples: int | None = None
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    architecture_ablation: str = "full"
    loss_mode: str = "default"
    history_perturbation: str = "original"
    history_keep_fraction: float = 1.0
    random_seed: int = 1729
    overwrite: bool = False
    run_name: str | None = None
    output_dir: Path | None = None
    output_csv: Path | None = None
    checkpoint_csv: Path | None = None
    detail_log_csv: Path | None = None
    training_log_csv: Path | None = None
    gate_log_csv: Path | None = None
    debug_log: Path | None = None
    save_gate_log: bool = False
    max_gate_log_rows: int | None = None
    trace_every: int = 0
    progress_every: int = 100
    trace_history_points: int = 5
    max_events: int = 1536
    max_queries_per_sample: int = 512
    batch_size: int = 16
    epochs: int = 80
    patience: int = 12
    learning_rate: float = 8e-4
    weight_decay: float = 1e-4
    d_model: int = 96
    layers: int = 2
    heads: int = 4
    dropout: float = 0.10
    delta_kernels: int = 4
    device: str = "auto"

    def __post_init__(self) -> None:
        dataset_name = canonical_dataset_name(self.dataset_name)
        architecture_ablation = canonical_architecture_ablation(self.architecture_ablation)
        loss_mode = canonical_loss_mode(self.loss_mode)
        history_perturbation = canonical_history_perturbation(self.history_perturbation)
        if not (0.0 < float(self.history_keep_fraction) <= 1.0):
            raise ValueError(f"history_keep_fraction must be in (0, 1], got {self.history_keep_fraction}")
        defaults = DATASET_DEFAULTS[dataset_name]
        seq_len = int(self.seq_len or defaults["seq_len"])
        pred_len = int(self.pred_len or defaults["pred_len"])
        run_name = self.run_name or f"{RUN_NAME_PREFIX}_{dataset_name.lower()}"
        output_dir = self.output_dir or Path("chronoa_results") / dataset_name.lower()
        output_dir.mkdir(parents=True, exist_ok=True)

        object.__setattr__(self, "dataset_name", dataset_name)
        object.__setattr__(self, "architecture_ablation", architecture_ablation)
        object.__setattr__(self, "loss_mode", loss_mode)
        object.__setattr__(self, "history_perturbation", history_perturbation)
        object.__setattr__(self, "seq_len", seq_len)
        object.__setattr__(self, "pred_len", pred_len)
        object.__setattr__(self, "run_name", run_name)
        object.__setattr__(self, "output_dir", output_dir)

        def default_path(path: Path | None, suffix: str) -> Path:
            return path or output_dir / f"{run_name}_{dataset_name}_{suffix}"

        object.__setattr__(self, "output_csv", default_path(self.output_csv, "Results.csv"))
        object.__setattr__(self, "checkpoint_csv", default_path(self.checkpoint_csv, "Checkpoint.csv"))
        object.__setattr__(self, "detail_log_csv", default_path(self.detail_log_csv, "DetailLog.csv"))
        object.__setattr__(self, "training_log_csv", default_path(self.training_log_csv, "TrainingLog.csv"))
        object.__setattr__(self, "gate_log_csv", default_path(self.gate_log_csv, "GateLog.csv"))
        object.__setattr__(self, "debug_log", default_path(self.debug_log, "debug.log"))


@dataclass(frozen=True)
class BenchmarkData:
    name: str
    display_name: str
    test_dataset: object
    fit_dataset: object
    selection_dataset: object
    lookback_span: float
    columns: list[str]
    encoder_index_by_column: dict[str, int]
    scaled_to_raw: Callable[[float, int], float]
    raw_to_scaled: Callable[[float, int], float]
    time_to_prompt: Callable[[float], float]
    format_timestamp: Callable[[float], str]
    format_tail_time: Callable[[float], str]
    time_unit: str
    entity_label: str


@dataclass(frozen=True)
class ForecastSample:
    key: str | int
    inputs: tuple[object, object, object]
    targets: object


@dataclass
class NeuralSample:
    key: str | int
    event_times: np.ndarray
    event_vars: np.ndarray
    event_values: np.ndarray
    query_times: np.ndarray
    query_vars: np.ndarray
    query_targets: np.ndarray
    query_steps: np.ndarray
    history_by_var: dict[int, tuple[list[float], list[float]]]


@dataclass
class Batch:
    event_times: object
    event_vars: object
    event_values: object
    event_mask: object
    query_times: object
    query_vars: object
    query_targets: object
    query_mask: object
    query_lengths: list[int]


def configure_logging(log_path: Path) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger(__name__)


def hours_to_timestamp(hour_value: float) -> str:
    hours = int(hour_value)
    minutes = int((hour_value - hours) * 60)
    seconds = int(((hour_value - hours) * 60 - minutes) * 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def format_values(values: Iterable[float], digits: int = 3) -> str:
    return "[" + ", ".join(f"{value:.{digits}f}" for value in values) + "]"


def materialize_dataset(dataset: object) -> list[object]:
    return [dataset[index] for index in range(len(dataset))]


def build_human_activity_samples(
    records: Iterable[tuple[object, object, object, object]],
    seq_len: int,
    pred_len: int,
) -> list[ForecastSample]:
    if torch is None:
        raise RuntimeError("PyTorch is required for HumanActivity preprocessing.")

    samples: list[ForecastSample] = []
    for record_id, tt, vals, mask in records:
        t_max = int(tt.max())
        for start_time in range(0, t_max - seq_len, 4000):
            end_history = start_time + seq_len
            end_target = start_time + seq_len + pred_len

            if end_history >= t_max:
                history_indices = torch.where((tt >= start_time) & (tt <= end_history))[0]
            else:
                history_indices = torch.where((tt >= start_time) & (tt < end_history))[0]

            if end_target >= t_max:
                target_indices = torch.where((tt >= end_history) & (tt <= end_target))[0]
            else:
                target_indices = torch.where((tt >= end_history) & (tt < end_target))[0]

            if len(history_indices) == 0:
                continue

            x_values = vals[history_indices].clone().float()
            x_mask = mask[history_indices].bool()
            y_values = vals[target_indices].clone().float()
            y_mask = mask[target_indices].bool()

            x_values[~x_mask] = torch.nan
            y_values[~y_mask] = torch.nan

            t_input = (tt[history_indices] - start_time).float()
            t_target = (tt[target_indices] - start_time).float()
            sample_key = f"{record_id}_{start_time // pred_len}"

            samples.append(
                ForecastSample(
                    key=sample_key,
                    inputs=(t_input, x_values, t_target),
                    targets=y_values,
                )
            )

    return samples


def load_benchmark_data(config: ChronoaConfig) -> BenchmarkData:
    if config.dataset_name == "P12":
        from data.dependencies.tsdm.tasks.P12 import Physionet2012

        task = Physionet2012(seq_len=float(config.seq_len) - 0.5, pred_len=config.pred_len)
        encoder = task.encoder.column_encoders
        columns = list(task.dataset.columns)
        encoder_columns = list(task.encoder.columns)
        encoder_index_by_column = {
            column_name: column_index for column_index, column_name in enumerate(encoder_columns)
        }

        def scaled_to_raw(scaled_value: float, encoder_index: int) -> float:
            return float(scaled_value * encoder.stdv[encoder_index] + encoder.mean[encoder_index])

        def raw_to_scaled(raw_value: float, encoder_index: int) -> float:
            return float((raw_value - encoder.mean[encoder_index]) / encoder.stdv[encoder_index])

        return BenchmarkData(
            name="P12",
            display_name="PhysioNet P12",
            test_dataset=task.get_dataset((0, "test")),
            fit_dataset=materialize_dataset(task.get_dataset((0, "train"))),
            selection_dataset=materialize_dataset(task.get_dataset((0, "val"))),
            lookback_span=float(config.seq_len),
            columns=columns,
            encoder_index_by_column=encoder_index_by_column,
            scaled_to_raw=scaled_to_raw,
            raw_to_scaled=raw_to_scaled,
            time_to_prompt=lambda value: value * 48,
            format_timestamp=hours_to_timestamp,
            format_tail_time=lambda value: f"{value:.2f}h",
            time_unit="hours after ICU admission",
            entity_label="patient",
        )

    if config.dataset_name == "MIMIC":
        from data.dependencies.tsdm.datasets import MIMIC_III_DeBrouwer2019 as MIMIC_III_Dataset
        from data.dependencies.tsdm.tasks.mimic_iii_debrouwer2019 import MIMIC_III_DeBrouwer2019

        task = MIMIC_III_DeBrouwer2019(seq_len=float(config.seq_len) - 0.5, pred_len=config.pred_len)
        columns = list(task.dataset.columns)
        metadata = MIMIC_III_Dataset()["metadata"].set_index("LABEL_CODE")
        label_codes = [int(column_name) for column_name in columns]
        missing_codes = [code for code in label_codes if code not in metadata.index]
        if missing_codes:
            raise ValueError(
                "De Brouwer MIMIC metadata is missing label codes: "
                + ", ".join(str(code) for code in missing_codes)
            )

        offset = metadata.loc[label_codes, "MEANS"].to_numpy(dtype=np.float64)
        scale = metadata.loc[label_codes, "STDVS"].to_numpy(dtype=np.float64)
        invalid_scale = (~np.isfinite(scale)) | (scale <= 1e-8)
        scale[invalid_scale] = 1.0

        return BenchmarkData(
            name="MIMIC",
            display_name="MIMIC-III",
            test_dataset=task.get_dataset((0, "test")),
            fit_dataset=materialize_dataset(task.get_dataset((0, "train"))),
            selection_dataset=materialize_dataset(task.get_dataset((0, "val"))),
            lookback_span=float(config.seq_len),
            columns=columns,
            encoder_index_by_column={column_name: column_index for column_index, column_name in enumerate(columns)},
            scaled_to_raw=lambda value, index: float(value * scale[index] + offset[index]),
            raw_to_scaled=lambda value, index: float((value - offset[index]) / scale[index]),
            time_to_prompt=lambda value: value * 96.0,
            format_timestamp=lambda value: hours_to_timestamp(value / 2.0),
            format_tail_time=lambda value: f"{value / 2.0:.2f}h",
            time_unit="hours after ICU admission",
            entity_label="patient",
        )

    if config.dataset_name == "USHCN":
        from data.dependencies.tsdm.tasks.ushcn_debrouwer2019 import USHCN_DeBrouwer2019

        task = USHCN_DeBrouwer2019(normalize_time=False, seq_len=config.seq_len - 0.5, pred_len=config.pred_len)
        columns = list(task.dataset.columns)
        return BenchmarkData(
            name="USHCN",
            display_name="USHCN",
            test_dataset=task.get_dataset((0, "test")),
            fit_dataset=materialize_dataset(task.get_dataset((0, "train"))),
            selection_dataset=materialize_dataset(task.get_dataset((0, "val"))),
            lookback_span=float(config.seq_len),
            columns=columns,
            encoder_index_by_column={column_name: column_index for column_index, column_name in enumerate(columns)},
            scaled_to_raw=lambda value, _index: float(value),
            raw_to_scaled=lambda value, _index: float(value),
            time_to_prompt=lambda value: value,
            format_timestamp=lambda value: f"day {value:.2f}",
            format_tail_time=lambda value: f"{value:.2f}d",
            time_unit="days in the 4-year USHCN window",
            entity_label="station",
        )

    if config.dataset_name == "HumanActivity":
        from data.dependencies.HumanActivity.HumanActivity import HumanActivity
        from sklearn import model_selection

        dataset_root = Path(
            os.getenv(
                "CHRONOLM_HUMANACTIVITY_ROOT",
                str(BENCHMARK_ROOT / "storage" / "datasets" / "HumanActivity"),
            )
        )
        processed_path = dataset_root / "processed" / "data.pt"
        human_activity = HumanActivity(root=str(dataset_root), download=not processed_path.exists())
        seen_records, test_records = model_selection.train_test_split(
            human_activity, train_size=0.9, random_state=42, shuffle=False
        )
        train_records, validation_records = model_selection.train_test_split(
            seen_records, train_size=0.9, random_state=42, shuffle=False
        )

        tag_names = ["ANKLE_LEFT", "ANKLE_RIGHT", "CHEST", "BELT"]
        axis_names = ["x", "y", "z"]
        columns = [f"{tag}_{axis}" for tag in tag_names for axis in axis_names]

        return BenchmarkData(
            name="HumanActivity",
            display_name="HumanActivity",
            test_dataset=build_human_activity_samples(test_records, seq_len=config.seq_len, pred_len=config.pred_len),
            fit_dataset=build_human_activity_samples(train_records, seq_len=config.seq_len, pred_len=config.pred_len),
            selection_dataset=build_human_activity_samples(validation_records, seq_len=config.seq_len, pred_len=config.pred_len),
            lookback_span=float(config.seq_len),
            columns=columns,
            encoder_index_by_column={column_name: column_index for column_index, column_name in enumerate(columns)},
            scaled_to_raw=lambda value, _index: float(value),
            raw_to_scaled=lambda value, _index: float(value),
            time_to_prompt=lambda value: value,
            format_timestamp=lambda value: f"{value:.0f} ms",
            format_tail_time=lambda value: f"{value:.0f}ms",
            time_unit="milliseconds inside the benchmark HumanActivity activity window",
            entity_label="activity_window",
        )

    raise ValueError(f"Unsupported dataset: {config.dataset_name}")


def tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def history_diagnostics(history_times: list[float], history_scaled: list[float]) -> dict[str, float]:
    n_history = len(history_scaled)
    if n_history == 0:
        return {
            "History_n": 0.0,
            "History_span": 0.0,
            "History_density": 0.0,
            "History_last_gap": float("nan"),
            "History_median_gap": float("nan"),
            "History_volatility": float("nan"),
            "History_mode_fraction": float("nan"),
        }

    times = np.asarray(history_times, dtype=float)
    values = np.asarray(history_scaled, dtype=float)
    finite_values = values[np.isfinite(values)]
    span = float(np.nanmax(times) - np.nanmin(times)) if len(times) else 0.0
    sorted_times = np.sort(times[np.isfinite(times)])
    gaps = np.diff(sorted_times) if len(sorted_times) > 1 else np.asarray([], dtype=float)
    if len(finite_values):
        rounded = [round(float(value), 4) for value in finite_values]
        mode_fraction = Counter(rounded).most_common(1)[0][1] / len(rounded)
        volatility = float(np.std(finite_values))
    else:
        mode_fraction = float("nan")
        volatility = float("nan")

    return {
        "History_n": float(n_history),
        "History_span": span,
        "History_density": float(n_history / span) if span > 1e-12 else float(n_history),
        "History_last_gap": float(gaps[-1]) if len(gaps) else float("nan"),
        "History_median_gap": float(np.median(gaps)) if len(gaps) else float("nan"),
        "History_volatility": volatility,
        "History_mode_fraction": float(mode_fraction),
    }


def rebuild_history_by_var(
    event_times: np.ndarray,
    event_vars: np.ndarray,
    event_values: np.ndarray,
    n_vars: int,
) -> dict[int, tuple[list[float], list[float]]]:
    history: dict[int, list[tuple[float, float]]] = {index: [] for index in range(n_vars)}
    for time_value, var_index, value in zip(event_times, event_vars, event_values):
        history[int(var_index)].append((float(time_value), float(value)))
    rebuilt: dict[int, tuple[list[float], list[float]]] = {}
    for var_index, points in history.items():
        points = sorted(points, key=lambda item: item[0])
        rebuilt[var_index] = ([point[0] for point in points], [point[1] for point in points])
    return rebuilt


def perturb_events(
    event_times: np.ndarray,
    event_vars: np.ndarray,
    event_values: np.ndarray,
    config: ChronoaConfig,
    benchmark_name: str,
    sample_index: int,
    n_vars: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(event_values) == 0:
        return event_times, event_vars, event_values
    perturbation = config.history_perturbation
    keep_fraction = float(config.history_keep_fraction)
    rng = np.random.default_rng(stable_seed(config.random_seed, benchmark_name, sample_index, perturbation, keep_fraction))
    times = np.asarray(event_times, dtype=float).copy()
    vars_ = np.asarray(event_vars, dtype=np.int64).copy()
    values = np.asarray(event_values, dtype=float).copy()

    if perturbation == "shuffle_timestamps":
        times = times[rng.permutation(len(times))]
    elif perturbation == "swap_halves":
        order = np.argsort(times)
        midpoint = len(order) // 2
        swapped = np.concatenate([order[midpoint:], order[:midpoint]])
        values = values[swapped]
        vars_ = vars_[swapped]
    elif perturbation == "drop_early_history":
        threshold = float(np.median(times))
        mask = times >= threshold
        if mask.any():
            times, vars_, values = times[mask], vars_[mask], values[mask]
    elif perturbation == "keep_last_obs":
        keep_indices: list[int] = []
        for var_index in range(n_vars):
            var_positions = np.where(vars_ == var_index)[0]
            if len(var_positions):
                keep_indices.append(int(var_positions[np.argmax(times[var_positions])]))
        if keep_indices:
            keep_indices = sorted(keep_indices, key=lambda idx: times[idx])
            times, vars_, values = times[keep_indices], vars_[keep_indices], values[keep_indices]
    elif perturbation == "keep_time_gaps":
        values = np.zeros_like(values)
    elif perturbation != "original":
        raise ValueError(f"Unsupported history perturbation: {perturbation}")

    if keep_fraction < 1.0 and len(values) > 1:
        n_keep = max(1, int(round(len(values) * keep_fraction)))
        keep = sorted(int(index) for index in rng.choice(len(values), size=n_keep, replace=False))
        times, vars_, values = times[keep], vars_[keep], values[keep]

    order = np.argsort(times)
    return times[order], vars_[order], values[order]


def truncate_events(
    event_times: np.ndarray,
    event_vars: np.ndarray,
    event_values: np.ndarray,
    max_events: int,
    training: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(event_values) <= max_events:
        return event_times, event_vars, event_values
    order = np.argsort(event_times)
    if training:
        rng = np.random.default_rng(seed)
        latest_count = max_events // 2
        latest = order[-latest_count:]
        older = order[:-latest_count]
        random_count = max_events - latest_count
        if len(older) > random_count:
            older = rng.choice(older, size=random_count, replace=False)
        keep = np.concatenate([older, latest])
    else:
        keep = order[-max_events:]
    keep = keep[np.argsort(event_times[keep])]
    return event_times[keep], event_vars[keep], event_values[keep]


def sample_to_neural_sample(
    sample: object,
    benchmark: BenchmarkData,
    config: ChronoaConfig,
    sample_index: int,
    training: bool,
    perturb_history: bool,
) -> NeuralSample | None:
    t_input, x_input, t_target = sample.inputs
    y_target = sample.targets
    x = tensor_to_numpy(x_input).astype(np.float32)
    y = tensor_to_numpy(y_target).astype(np.float32)
    t_in = tensor_to_numpy(t_input).astype(np.float32)
    t_out = tensor_to_numpy(t_target).astype(np.float32)

    if x.ndim != 2 or y.ndim != 2:
        return None
    n_vars = len(benchmark.columns)
    valid_history = np.isfinite(x[:, :n_vars])
    event_rows, event_cols = np.where(valid_history)
    if len(event_rows) == 0:
        return None
    event_times = np.asarray([round(benchmark.time_to_prompt(float(t_in[row])), 2) for row in event_rows], dtype=np.float32)
    event_vars = event_cols.astype(np.int64)
    event_values = x[event_rows, event_cols].astype(np.float32)

    if perturb_history:
        event_times, event_vars, event_values = perturb_events(
            event_times=event_times,
            event_vars=event_vars,
            event_values=event_values,
            config=config,
            benchmark_name=benchmark.name,
            sample_index=sample_index,
            n_vars=n_vars,
        )
    event_times, event_vars, event_values = truncate_events(
        event_times=event_times,
        event_vars=event_vars,
        event_values=event_values,
        max_events=int(config.max_events),
        training=training,
        seed=stable_seed(config.random_seed, benchmark.name, sample_index, "event_truncate"),
    )
    if len(event_values) == 0:
        return None

    forecast_limit = min(int(config.pred_len), y.shape[0], len(t_out))
    query_times: list[float] = []
    query_vars: list[int] = []
    query_targets: list[float] = []
    query_steps: list[int] = []
    for step_index in range(forecast_limit):
        valid_targets = np.isfinite(y[step_index, :n_vars])
        if not valid_targets.any():
            continue
        target_time = round(benchmark.time_to_prompt(float(t_out[step_index])), 2)
        for var_index in np.where(valid_targets)[0]:
            query_times.append(target_time)
            query_vars.append(int(var_index))
            query_targets.append(float(y[step_index, var_index]))
            query_steps.append(step_index)

    if not query_targets:
        return None

    if training and len(query_targets) > int(config.max_queries_per_sample):
        rng = np.random.default_rng(stable_seed(config.random_seed, benchmark.name, sample_index, "query_subsample"))
        keep = sorted(int(index) for index in rng.choice(len(query_targets), size=int(config.max_queries_per_sample), replace=False))
        query_times = [query_times[index] for index in keep]
        query_vars = [query_vars[index] for index in keep]
        query_targets = [query_targets[index] for index in keep]
        query_steps = [query_steps[index] for index in keep]

    key = getattr(sample, "key", sample_index)
    return NeuralSample(
        key=key,
        event_times=np.asarray(event_times, dtype=np.float32),
        event_vars=np.asarray(event_vars, dtype=np.int64),
        event_values=np.asarray(event_values, dtype=np.float32),
        query_times=np.asarray(query_times, dtype=np.float32),
        query_vars=np.asarray(query_vars, dtype=np.int64),
        query_targets=np.asarray(query_targets, dtype=np.float32),
        query_steps=np.asarray(query_steps, dtype=np.int64),
        history_by_var=rebuild_history_by_var(event_times, event_vars, event_values, n_vars=n_vars),
    )


def build_neural_samples(
    dataset: object,
    benchmark: BenchmarkData,
    config: ChronoaConfig,
    training: bool,
    perturb_history: bool,
    max_samples: int | None,
    logger: logging.Logger,
    label: str,
) -> list[NeuralSample]:
    total = len(dataset)
    n_samples = min(max_samples or total, total)
    samples: list[NeuralSample] = []
    for sample_index in range(n_samples):
        neural_sample = sample_to_neural_sample(
            sample=dataset[sample_index],
            benchmark=benchmark,
            config=config,
            sample_index=sample_index,
            training=training,
            perturb_history=perturb_history,
        )
        if neural_sample is not None:
            samples.append(neural_sample)
    logger.info("  %s neural samples: %s/%s", label, len(samples), n_samples)
    return samples


def choose_device(config: ChronoaConfig) -> object:
    if torch is None:
        raise RuntimeError("Chronoa-EDGE requires PyTorch. Install torch before running Chronoa.")
    requested = str(config.device).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(requested)


def collate_batch(samples: list[NeuralSample], lookback_span: float, device: object) -> Batch:
    max_events = max(len(sample.event_values) for sample in samples)
    max_queries = max(len(sample.query_targets) for sample in samples)
    batch_size = len(samples)
    event_times = torch.zeros((batch_size, max_events), dtype=torch.float32, device=device)
    event_vars = torch.zeros((batch_size, max_events), dtype=torch.long, device=device)
    event_values = torch.zeros((batch_size, max_events), dtype=torch.float32, device=device)
    event_mask = torch.zeros((batch_size, max_events), dtype=torch.bool, device=device)
    query_times = torch.zeros((batch_size, max_queries), dtype=torch.float32, device=device)
    query_vars = torch.zeros((batch_size, max_queries), dtype=torch.long, device=device)
    query_targets = torch.zeros((batch_size, max_queries), dtype=torch.float32, device=device)
    query_mask = torch.zeros((batch_size, max_queries), dtype=torch.bool, device=device)
    query_lengths: list[int] = []
    scale = max(float(lookback_span), 1e-6)

    for batch_index, sample in enumerate(samples):
        n_events = len(sample.event_values)
        n_queries = len(sample.query_targets)
        event_times[batch_index, :n_events] = torch.as_tensor(sample.event_times / scale, dtype=torch.float32, device=device)
        event_vars[batch_index, :n_events] = torch.as_tensor(sample.event_vars, dtype=torch.long, device=device)
        event_values[batch_index, :n_events] = torch.as_tensor(sample.event_values, dtype=torch.float32, device=device)
        event_mask[batch_index, :n_events] = True
        query_times[batch_index, :n_queries] = torch.as_tensor(sample.query_times / scale, dtype=torch.float32, device=device)
        query_vars[batch_index, :n_queries] = torch.as_tensor(sample.query_vars, dtype=torch.long, device=device)
        query_targets[batch_index, :n_queries] = torch.as_tensor(sample.query_targets, dtype=torch.float32, device=device)
        query_mask[batch_index, :n_queries] = True
        query_lengths.append(n_queries)

    return Batch(
        event_times=event_times,
        event_vars=event_vars,
        event_values=event_values,
        event_mask=event_mask,
        query_times=query_times,
        query_vars=query_vars,
        query_targets=query_targets,
        query_mask=query_mask,
        query_lengths=query_lengths,
    )


def iter_batches(
    samples: list[NeuralSample],
    batch_size: int,
    lookback_span: float,
    device: object,
    shuffle: bool,
    seed: int,
) -> Iterable[tuple[list[NeuralSample], Batch]]:
    indices = np.arange(len(samples))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_samples = [samples[int(index)] for index in batch_indices]
        yield batch_samples, collate_batch(batch_samples, lookback_span=lookback_span, device=device)


if nn is not None:

    class FourierTimeProjection(nn.Module):
        def __init__(self, n_features: int, output_dim: int) -> None:
            super().__init__()
            half = max(1, n_features // 2)
            frequencies = torch.logspace(0.0, 3.0, steps=half)
            self.register_buffer("frequencies", frequencies)
            self.proj = nn.Sequential(
                nn.Linear(2 * half + 2, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )

        def forward(self, times: object) -> object:
            angles = times.unsqueeze(-1) * self.frequencies * (2.0 * math.pi)
            features = torch.cat([
                times.unsqueeze(-1),
                torch.log1p(torch.clamp(times, min=0.0)).unsqueeze(-1),
                torch.sin(angles),
                torch.cos(angles),
            ], dim=-1)
            return self.proj(features)


    class ChronoaEventGraphNet(nn.Module):
        """Query-conditioned event graph network for irregular forecasting."""

        def __init__(
            self,
            n_variables: int,
            d_model: int,
            n_heads: int,
            n_layers: int,
            dropout: float,
            delta_kernels: int,
            architecture_ablation: str,
        ) -> None:
            super().__init__()
            self.n_variables = int(n_variables)
            self.d_model = int(d_model)
            self.delta_kernels = int(delta_kernels)
            self.architecture_ablation = architecture_ablation
            self.apex_decoder = architecture_ablation == "apex"
            self.use_delta_kernel = architecture_ablation not in {"no_delta_kernel", "direct_decoder"}
            self.use_graph_mixer = architecture_ablation not in {"no_graph_mixer", "direct_decoder"}
            self.use_transformer = architecture_ablation not in {"no_transformer", "direct_decoder"}
            self.use_time_encoding = architecture_ablation != "no_time_encoding"
            self.direct_decoder = architecture_ablation == "direct_decoder"

            self.variable_embedding = nn.Embedding(n_variables, d_model)
            self.value_projection = nn.Sequential(
                nn.Linear(1, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.event_time_projection = FourierTimeProjection(8, d_model)
            self.query_time_projection = FourierTimeProjection(8, d_model)
            self.event_norm = nn.LayerNorm(d_model)
            self.query_norm = nn.LayerNorm(d_model)

            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.cross_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
            self.graph_projection = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.log_kernel_scales = nn.Parameter(torch.linspace(math.log(0.05), math.log(2.0), steps=max(delta_kernels, 1)))
            self.same_variable_bias = nn.Parameter(torch.tensor(1.0))
            self.graph_bias_scale = nn.Parameter(torch.tensor(0.5))
            self.kernel_projection = nn.Sequential(
                nn.Linear(max(delta_kernels, 1) * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            self.gap_projection = nn.Sequential(nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model))
            self.decoder_backbone = nn.Sequential(
                nn.LayerNorm(5 * d_model),
                nn.Linear(5 * d_model, 3 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(3 * d_model, d_model),
                nn.GELU(),
            )
            self.decoder_head = nn.Linear(d_model, 1)
            self.scalar_log_scales = nn.Parameter(torch.log(torch.tensor([0.03, 0.15, 0.60], dtype=torch.float32)))
            scalar_experts = 5
            gate_input_dim = d_model + scalar_experts + 2
            self.expert_gate = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, scalar_experts + 1),
            )
            self.expert_residual = nn.Sequential(
                nn.LayerNorm(gate_input_dim),
                nn.Linear(gate_input_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )

        def graph_adjacency(self) -> object:
            embeddings = torch.nn.functional.normalize(self.variable_embedding.weight, dim=-1)
            logits = embeddings @ embeddings.T / math.sqrt(float(self.d_model))
            return torch.softmax(logits, dim=-1)

        def graph_context(self, query_vars: object) -> object:
            if not self.use_graph_mixer:
                return torch.zeros((*query_vars.shape, self.d_model), device=query_vars.device, dtype=torch.float32)
            adjacency = self.graph_adjacency()
            mixed = adjacency[query_vars] @ self.variable_embedding.weight
            return self.graph_projection(mixed)

        def encode_events(self, event_times: object, event_vars: object, event_values: object, event_mask: object) -> object:
            event_tokens = self.variable_embedding(event_vars) + self.value_projection(event_values.unsqueeze(-1))
            if self.use_time_encoding:
                event_tokens = event_tokens + self.event_time_projection(event_times)
            event_tokens = self.event_norm(event_tokens)
            event_tokens = event_tokens.masked_fill(~event_mask.unsqueeze(-1), 0.0)
            if self.use_transformer:
                memory = self.encoder(event_tokens, src_key_padding_mask=~event_mask)
                return memory.masked_fill(~event_mask.unsqueeze(-1), 0.0)
            return event_tokens

        @staticmethod
        def masked_softmax(logits: object, mask: object) -> object:
            weights = torch.softmax(logits.masked_fill(~mask, -1e9), dim=-1)
            weights = weights * mask.to(weights.dtype)
            return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        def build_queries(self, query_times: object, query_vars: object) -> object:
            query_tokens = self.variable_embedding(query_vars)
            if self.use_time_encoding:
                query_tokens = query_tokens + self.query_time_projection(query_times)
            query_tokens = query_tokens + self.graph_context(query_vars)
            return self.query_norm(query_tokens)

        def delta_kernel_context(
            self,
            memory: object,
            event_times: object,
            event_vars: object,
            event_mask: object,
            query_times: object,
            query_vars: object,
        ) -> object:
            if not self.use_delta_kernel:
                return torch.zeros((*query_vars.shape, self.d_model), device=memory.device, dtype=memory.dtype)
            delta = torch.abs(query_times.unsqueeze(-1) - event_times.unsqueeze(1))
            valid = event_mask.unsqueeze(1)
            same = (query_vars.unsqueeze(-1) == event_vars.unsqueeze(1)).to(memory.dtype)
            adjacency = self.graph_adjacency()
            graph_strength = adjacency[query_vars.unsqueeze(-1), event_vars.unsqueeze(1)].clamp_min(1e-6)
            contexts: list[object] = []
            for scale in torch.nn.functional.softplus(self.log_kernel_scales).clamp_min(1e-4):
                logits = -delta / scale
                logits = logits + same * self.same_variable_bias
                logits = logits + torch.log(graph_strength) * torch.nn.functional.softplus(self.graph_bias_scale)
                logits = logits.masked_fill(~valid, -1e9)
                weights = torch.softmax(logits, dim=-1)
                contexts.append(weights @ memory)
            return self.kernel_projection(torch.cat(contexts, dim=-1))

        def scalar_experts(self, batch: Batch) -> tuple[object, object]:
            """Cheap query-level temporal experts used by the APEX gate."""
            event_times = batch.event_times
            event_values = batch.event_values
            event_vars = batch.event_vars
            query_times = batch.query_times
            query_vars = batch.query_vars
            valid = batch.event_mask.unsqueeze(1)
            same_valid = valid & (query_vars.unsqueeze(-1) == event_vars.unsqueeze(1))
            same_has = same_valid.any(dim=-1, keepdim=True).to(event_values.dtype)

            delta = torch.abs(query_times.unsqueeze(-1) - event_times.unsqueeze(1))
            values = event_values.unsqueeze(1)
            scales = torch.exp(self.scalar_log_scales).clamp(1e-3, 4.0)

            last_logits = event_times.unsqueeze(1) * 64.0
            last_weights = self.masked_softmax(last_logits, same_valid)
            same_last = torch.sum(last_weights * values, dim=-1)
            last_time = torch.sum(last_weights * event_times.unsqueeze(1), dim=-1)

            same_short = self.masked_softmax(-delta / scales[0], same_valid)
            same_mid = self.masked_softmax(-delta / scales[1], same_valid)
            same_long = self.masked_softmax(-delta / scales[2], same_valid)
            same_ema = torch.sum(same_short * values, dim=-1)

            if self.use_graph_mixer:
                adjacency = self.graph_adjacency()
                graph_strength = adjacency[query_vars.unsqueeze(-1), event_vars.unsqueeze(1)].clamp_min(1e-6)
            else:
                graph_strength = torch.ones_like(delta)
            graph_logits = -delta / scales[1] + torch.log(graph_strength)
            graph_weights = self.masked_softmax(graph_logits, valid)
            graph_ema = torch.sum(graph_weights * values, dim=-1)

            global_weights = self.masked_softmax(-delta / scales[2], valid)
            global_ema = torch.sum(global_weights * values, dim=-1)

            mean_time = torch.sum(same_long * event_times.unsqueeze(1), dim=-1)
            mean_value = torch.sum(same_long * values, dim=-1)
            centered_time = event_times.unsqueeze(1) - mean_time.unsqueeze(-1)
            centered_value = values - mean_value.unsqueeze(-1)
            time_var = torch.sum(same_long * centered_time.pow(2), dim=-1)
            covariance = torch.sum(same_long * centered_time * centered_value, dim=-1)
            slope = torch.clamp(covariance / time_var.clamp_min(1e-4), min=-8.0, max=8.0)
            trend = same_last + slope * (query_times - last_time)

            has_same = same_has.squeeze(-1).bool()
            same_last = torch.where(has_same, same_last, graph_ema)
            same_ema = torch.where(has_same, same_ema, graph_ema)
            trend = torch.where(has_same, trend, graph_ema)

            experts = torch.stack([same_last, same_ema, trend, graph_ema, global_ema], dim=-1)
            return torch.clamp(experts, min=-12.0, max=12.0), same_has

        def forward(self, batch: Batch, return_diagnostics: bool = False) -> object:
            memory = self.encode_events(batch.event_times, batch.event_vars, batch.event_values, batch.event_mask)
            query_tokens = self.build_queries(batch.query_times, batch.query_vars)
            if self.direct_decoder:
                attention_context = torch.zeros_like(query_tokens)
                kernel_context = torch.zeros_like(query_tokens)
            else:
                attention_context, _ = self.cross_attention(
                    query_tokens,
                    memory,
                    memory,
                    key_padding_mask=~batch.event_mask,
                    need_weights=False,
                )
                kernel_context = self.delta_kernel_context(
                    memory=memory,
                    event_times=batch.event_times,
                    event_vars=batch.event_vars,
                    event_mask=batch.event_mask,
                    query_times=batch.query_times,
                    query_vars=batch.query_vars,
                )
            last_time = torch.max(batch.event_times.masked_fill(~batch.event_mask, -1e9), dim=1).values
            gap = (batch.query_times - last_time.unsqueeze(1)).unsqueeze(-1)
            gap_context = self.gap_projection(gap)
            graph_context = self.graph_context(batch.query_vars)
            decoder_input = torch.cat([query_tokens, attention_context, kernel_context, graph_context, gap_context], dim=-1)
            decoder_hidden = self.decoder_backbone(decoder_input)
            neural_decoded = self.decoder_head(decoder_hidden).squeeze(-1)
            diagnostics: dict[str, object] = {}
            if self.apex_decoder:
                experts, same_has = self.scalar_experts(batch)
                gate_input = torch.cat([decoder_hidden, experts, same_has, gap], dim=-1)
                gate_weights = torch.softmax(self.expert_gate(gate_input), dim=-1)
                expert_stack = torch.cat([neural_decoded.unsqueeze(-1), experts], dim=-1)
                residual = 0.05 * torch.tanh(self.expert_residual(gate_input)).squeeze(-1)
                decoded = torch.sum(gate_weights * expert_stack, dim=-1) + residual
                if return_diagnostics:
                    diagnostics = {
                        "gate_weights": gate_weights,
                        "experts": experts,
                        "neural_decoded": neural_decoded,
                        "residual": residual,
                        "same_has": same_has,
                        "gap": gap,
                    }
            else:
                decoded = neural_decoded
            decoded = decoded.masked_fill(~batch.query_mask, 0.0)
            if return_diagnostics:
                return decoded, diagnostics
            return decoded


def masked_loss(
    prediction: object,
    target: object,
    mask: object,
    architecture: str = "full",
    loss_mode: str = "default",
) -> object:
    error = prediction[mask] - target[mask]
    mse = torch.mean(error**2)
    if canonical_loss_mode(loss_mode) == "mse":
        return mse
    huber = torch.nn.functional.smooth_l1_loss(prediction[mask], target[mask], beta=0.5)
    mae = torch.mean(torch.abs(error))
    if architecture == "apex":
        return 0.50 * mse + 0.20 * huber + 0.30 * mae
    return 0.60 * mse + 0.30 * huber + 0.10 * mae


def evaluate_neural_model(
    model: object,
    samples: list[NeuralSample],
    config: ChronoaConfig,
    lookback_span: float,
    device: object,
) -> dict[str, float]:
    model.eval()
    abs_error = 0.0
    sq_error = 0.0
    count = 0
    with torch.no_grad():
        for _, batch in iter_batches(samples, config.batch_size, lookback_span, device, shuffle=False, seed=config.random_seed):
            prediction = model(batch)
            error = prediction[batch.query_mask] - batch.query_targets[batch.query_mask]
            abs_error += float(torch.sum(torch.abs(error)).detach().cpu())
            sq_error += float(torch.sum(error**2).detach().cpu())
            count += int(batch.query_mask.sum().detach().cpu())
    if count == 0:
        return {"MAE_scaled": float("nan"), "MSE_scaled": float("nan"), "Predictions": 0.0}
    return {"MAE_scaled": abs_error / count, "MSE_scaled": sq_error / count, "Predictions": float(count)}


def train_neural_model(
    train_samples: list[NeuralSample],
    validation_samples: list[NeuralSample],
    benchmark: BenchmarkData,
    config: ChronoaConfig,
    device: object,
    logger: logging.Logger,
) -> tuple[object, pd.DataFrame]:
    if nn is None or torch is None:
        raise RuntimeError("Chronoa-EDGE requires PyTorch.")
    torch.manual_seed(int(config.random_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.random_seed))

    model = ChronoaEventGraphNet(
        n_variables=len(benchmark.columns),
        d_model=int(config.d_model),
        n_heads=int(config.heads),
        n_layers=int(config.layers),
        dropout=float(config.dropout),
        delta_kernels=int(config.delta_kernels),
        architecture_ablation=config.architecture_ablation,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config.learning_rate), weight_decay=float(config.weight_decay))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(int(config.epochs), 1), eta_min=float(config.learning_rate) * 0.1)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("inf")
    stale_epochs = 0
    rows: list[dict[str, float | int | str]] = []

    logger.info("  Training %s (%s, loss=%s) on %s", MODEL_NAME, config.architecture_ablation, config.loss_mode, device)
    logger.info("  Train samples=%s, validation samples=%s", len(train_samples), len(validation_samples))
    for epoch in range(1, int(config.epochs) + 1):
        model.train()
        running = 0.0
        seen = 0
        for _, batch in iter_batches(
            train_samples,
            int(config.batch_size),
            benchmark.lookback_span,
            device,
            shuffle=True,
            seed=stable_seed(config.random_seed, epoch),
        ):
            prediction = model(batch)
            loss = masked_loss(
                prediction,
                batch.query_targets,
                batch.query_mask,
                architecture=config.architecture_ablation,
                loss_mode=config.loss_mode,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n_targets = int(batch.query_mask.sum().detach().cpu())
            running += float(loss.detach().cpu()) * max(n_targets, 1)
            seen += n_targets
        scheduler.step()
        train_loss = running / max(seen, 1)
        val_metrics = evaluate_neural_model(model, validation_samples, config, benchmark.lookback_span, device)
        mae_score_weight = 0.0 if config.loss_mode == "mse" else (0.25 if config.architecture_ablation == "apex" else 0.05)
        score = val_metrics["MSE_scaled"] + mae_score_weight * val_metrics["MAE_scaled"]
        rows.append(
            {
                "Epoch": epoch,
                "Loss_mode": config.loss_mode,
                "Train_loss": train_loss,
                "Validation_MAE_scaled": val_metrics["MAE_scaled"],
                "Validation_MSE_scaled": val_metrics["MSE_scaled"],
                "Validation_score": score,
                "Validation_predictions": val_metrics["Predictions"],
                "Learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        logger.info(
            "  epoch=%03d train_loss=%.6f val_mae=%.6f val_mse=%.6f targets=%s",
            epoch,
            train_loss,
            val_metrics["MAE_scaled"],
            val_metrics["MSE_scaled"],
            int(val_metrics["Predictions"]),
        )
        if score + 1e-7 < best_score:
            best_score = score
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= int(config.patience):
            logger.info("  Early stopping after epoch %s (best score %.6f)", epoch, best_score)
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict_samples(
    model: object,
    samples: list[NeuralSample],
    config: ChronoaConfig,
    lookback_span: float,
    device: object,
) -> list[np.ndarray]:
    model.eval()
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for batch_samples, batch in iter_batches(samples, config.batch_size, lookback_span, device, shuffle=False, seed=config.random_seed):
            batch_prediction = model(batch).detach().cpu().numpy()
            for row_index, sample in enumerate(batch_samples):
                n_queries = len(sample.query_targets)
                predictions.append(batch_prediction[row_index, :n_queries].astype(float))
    return predictions


def predict_samples_with_gate_diagnostics(
    model: object,
    samples: list[NeuralSample],
    config: ChronoaConfig,
    lookback_span: float,
    device: object,
) -> tuple[list[np.ndarray], list[dict[str, np.ndarray]]]:
    model.eval()
    predictions: list[np.ndarray] = []
    diagnostics_by_sample: list[dict[str, np.ndarray]] = []
    with torch.no_grad():
        for batch_samples, batch in iter_batches(samples, config.batch_size, lookback_span, device, shuffle=False, seed=config.random_seed):
            batch_prediction, batch_diagnostics = model(batch, return_diagnostics=True)
            prediction_array = batch_prediction.detach().cpu().numpy()
            diagnostic_arrays = {
                name: value.detach().cpu().numpy()
                for name, value in batch_diagnostics.items()
            }
            for row_index, sample in enumerate(batch_samples):
                n_queries = len(sample.query_targets)
                predictions.append(prediction_array[row_index, :n_queries].astype(float))
                diagnostics_by_sample.append(
                    {
                        name: array[row_index, :n_queries].astype(float)
                        for name, array in diagnostic_arrays.items()
                    }
                )
    return predictions, diagnostics_by_sample


def compute_global_scaled_metrics(detail_log_csv: Path) -> dict[str, float] | None:
    if not detail_log_csv.exists():
        return None
    detail = pd.read_csv(detail_log_csv)
    required = {"Actual_scaled", "Predicted_scaled"}
    if not required <= set(detail.columns):
        return None
    valid = detail[["Actual_scaled", "Predicted_scaled"]].notna().all(axis=1)
    detail = detail.loc[valid].copy()
    if detail.empty:
        return None
    error = detail["Actual_scaled"] - detail["Predicted_scaled"]
    detail["AE_scaled"] = error.abs()
    detail["SE_scaled"] = error**2
    per_variable = (
        detail.groupby("Variable", sort=False)
        .agg(MAE_scaled=("AE_scaled", "mean"), MSE_scaled=("SE_scaled", "mean"))
        .reset_index()
    )
    return {
        "MAE_scaled": float(detail["AE_scaled"].mean()),
        "MSE_scaled": float(detail["SE_scaled"].mean()),
        "Variable_avg_MAE_scaled": float(per_variable["MAE_scaled"].mean()),
        "Variable_avg_MSE_scaled": float(per_variable["MSE_scaled"].mean()),
        "Predictions": float(len(detail)),
    }


def experiment_variant(config: ChronoaConfig) -> str:
    parts: list[str] = []
    if config.architecture_ablation != "full":
        parts.append(config.architecture_ablation)
    if config.loss_mode != "default":
        parts.append(f"loss_{config.loss_mode}")
    if config.history_perturbation != "original":
        parts.append(config.history_perturbation)
    if float(config.history_keep_fraction) < 1.0:
        parts.append(f"keep{int(round(float(config.history_keep_fraction) * 100)):03d}")
    defaults = DATASET_DEFAULTS[config.dataset_name]
    if config.seq_len != defaults["seq_len"] or config.pred_len != defaults["pred_len"]:
        parts.append(f"sl{config.seq_len}_pl{config.pred_len}")
    return "+".join(parts) if parts else "default"


def experiment_method_label(config: ChronoaConfig) -> str:
    variant = experiment_variant(config)
    return MODEL_NAME if variant == "default" else f"{MODEL_NAME}:{variant}"


def variable_result_rows(detail_rows: list[dict], columns: list[str], benchmark_name: str, method_label: str, variant: str, config: ChronoaConfig) -> list[dict]:
    frame = pd.DataFrame(detail_rows)
    rows: list[dict] = []
    if frame.empty:
        return rows
    for variable_name in columns:
        group = frame.loc[frame["Variable"].eq(variable_name)]
        if group.empty:
            continue
        rows.append(
            {
                "Dataset": benchmark_name,
                "Variable": variable_name,
                "Method": method_label,
                "Base_method": MODEL_NAME,
                "Variant": variant,
                "Seq_len": config.seq_len,
                "Pred_len": config.pred_len,
                "Architecture_ablation": config.architecture_ablation,
                "Loss_mode": config.loss_mode,
                "History_perturbation": config.history_perturbation,
                "MAE_scaled": round(float(group["AE_scaled"].mean()), 6),
                "MSE_scaled": round(float(group["SE_scaled"].mean()), 6),
                "MAE_raw": round(float(group["AE_raw"].mean()), 4),
                "Predictions": int(len(group)),
                "Fallback": 0,
                "Time_s": float("nan"),
            }
        )
    return rows


def should_trace_sample(valid_sample_count: int, config: ChronoaConfig) -> bool:
    return config.trace_every > 0 and valid_sample_count % config.trace_every == 0


def run(config: ChronoaConfig) -> dict[str, float | str]:
    warnings.filterwarnings("ignore")
    if config.overwrite:
        for artifact_path in (
            config.output_csv,
            config.checkpoint_csv,
            config.detail_log_csv,
            config.training_log_csv,
            config.gate_log_csv,
            config.debug_log,
        ):
            artifact_path.unlink(missing_ok=True)
    logger = configure_logging(config.debug_log)
    method_label = experiment_method_label(config)
    variant = experiment_variant(config)

    logger.info("Run name: %s", config.run_name)
    logger.info("Dataset: %s", config.dataset_name)
    logger.info("Model: %s", MODEL_NAME)
    logger.info("Variant: %s", variant)
    logger.info("Architecture ablation: %s", config.architecture_ablation)
    logger.info("Loss mode: %s", config.loss_mode)
    logger.info("History perturbation for evaluation: %s", config.history_perturbation)
    logger.info("Output directory: %s", config.output_dir)
    logger.info("Loading %s data through the benchmark pipeline...", config.dataset_name)
    load_start = time.time()
    benchmark = load_benchmark_data(config)
    logger.info("  Data loaded in %.1fs", time.time() - load_start)
    logger.info("  Variables: %s", len(benchmark.columns))

    device = choose_device(config)
    train_samples = build_neural_samples(
        benchmark.fit_dataset,
        benchmark=benchmark,
        config=config,
        training=True,
        perturb_history=False,
        max_samples=config.max_train_samples,
        logger=logger,
        label="train",
    )
    validation_samples = build_neural_samples(
        benchmark.selection_dataset,
        benchmark=benchmark,
        config=config,
        training=False,
        perturb_history=False,
        max_samples=config.max_validation_samples,
        logger=logger,
        label="validation",
    )
    if not train_samples or not validation_samples:
        raise RuntimeError("Chronoa-EDGE needs non-empty train and validation splits.")

    total_start = time.time()
    model, training_log = train_neural_model(train_samples, validation_samples, benchmark, config, device, logger)
    training_log.to_csv(config.training_log_csv, index=False)
    logger.info("Training log CSV: %s", config.training_log_csv)

    test_samples = build_neural_samples(
        benchmark.test_dataset,
        benchmark=benchmark,
        config=config,
        training=False,
        perturb_history=True,
        max_samples=config.max_test_samples,
        logger=logger,
        label="test",
    )
    if not test_samples:
        raise RuntimeError("No valid test samples were produced.")

    if config.save_gate_log and config.architecture_ablation == "apex":
        predictions, gate_diagnostics = predict_samples_with_gate_diagnostics(
            model,
            test_samples,
            config,
            benchmark.lookback_span,
            device,
        )
    else:
        if config.save_gate_log:
            logger.warning("Gate logging was requested but only the apex architecture has gate diagnostics.")
        predictions = predict_samples(model, test_samples, config, benchmark.lookback_span, device)
        gate_diagnostics = [{} for _ in predictions]
    detail_rows: list[dict] = []
    gate_rows: list[dict] = []
    valid_sample_count = 0
    eval_start = time.time()
    for sample, predicted_values, gate_diag in zip(test_samples, predictions, gate_diagnostics):
        valid_sample_count += 1
        trace_sample = should_trace_sample(valid_sample_count, config)
        if trace_sample:
            logger.info(
                "    [INPUT] sample=%s %s=%s events=%s queries=%s",
                valid_sample_count,
                benchmark.entity_label,
                sample.key,
                len(sample.event_values),
                len(sample.query_targets),
            )
        for query_index, predicted_scaled in enumerate(predicted_values):
            var_index = int(sample.query_vars[query_index])
            variable_name = benchmark.columns[var_index]
            encoder_index = benchmark.encoder_index_by_column[variable_name]
            actual_scaled = float(sample.query_targets[query_index])
            predicted_scaled = float(predicted_scaled)
            actual_raw = benchmark.scaled_to_raw(actual_scaled, encoder_index)
            predicted_raw = benchmark.scaled_to_raw(predicted_scaled, encoder_index)
            ae_scaled = abs(actual_scaled - predicted_scaled)
            se_scaled = (actual_scaled - predicted_scaled) ** 2
            ae_raw = abs(actual_raw - predicted_raw)
            history_times, history_values = sample.history_by_var.get(var_index, ([], []))
            history_diag = history_diagnostics(history_times, history_values)
            detail_base = {
                "Dataset": benchmark.name,
                "Method": method_label,
                "Base_method": MODEL_NAME,
                "Variant": variant,
                "Seq_len": config.seq_len,
                "Pred_len": config.pred_len,
                "Architecture_ablation": config.architecture_ablation,
                "Loss_mode": config.loss_mode,
                "History_perturbation": config.history_perturbation,
                "History_keep_fraction": config.history_keep_fraction,
                "Variable": variable_name,
                "Entity": sample.key,
                "Step": int(sample.query_steps[query_index]),
                "Target_time": round(float(sample.query_times[query_index]), 2),
                "Actual_scaled": round(actual_scaled, 6),
                "Predicted_scaled": round(predicted_scaled, 6),
            }
            detail_rows.append(
                {
                    **detail_base,
                    "Actual_raw": round(actual_raw, 4),
                    "Predicted_raw": round(predicted_raw, 4),
                    "AE_scaled": round(ae_scaled, 6),
                    "SE_scaled": round(se_scaled, 6),
                    "AE_raw": round(ae_raw, 6),
                    "History_n": round(float(history_diag.get("History_n", float("nan"))), 6),
                    "History_span": round(float(history_diag.get("History_span", float("nan"))), 6),
                    "History_density": round(float(history_diag.get("History_density", float("nan"))), 6),
                    "History_last_gap": round(float(history_diag.get("History_last_gap", float("nan"))), 6),
                    "History_median_gap": round(float(history_diag.get("History_median_gap", float("nan"))), 6),
                    "History_volatility": round(float(history_diag.get("History_volatility", float("nan"))), 6),
                    "History_mode_fraction": round(float(history_diag.get("History_mode_fraction", float("nan"))), 6),
                }
            )
            if gate_diag and (
                config.max_gate_log_rows is None
                or len(gate_rows) < int(config.max_gate_log_rows)
            ):
                gate_weights = gate_diag["gate_weights"][query_index]
                experts = gate_diag["experts"][query_index]
                neural_decoded = float(gate_diag["neural_decoded"][query_index])
                residual = float(gate_diag["residual"][query_index])
                same_has = float(np.squeeze(gate_diag["same_has"][query_index]))
                query_gap = float(np.squeeze(gate_diag["gap"][query_index]))
                expert_stack = np.concatenate([[neural_decoded], experts])
                gate_entropy = -float(np.sum(gate_weights * np.log(np.clip(gate_weights, 1e-12, 1.0))))
                gate_row = {
                    **detail_base,
                    "AE_scaled": round(ae_scaled, 6),
                    "SE_scaled": round(se_scaled, 6),
                    "Same_variable_available": int(same_has > 0.5),
                    "Query_gap_normalized": round(query_gap, 6),
                    "Neural_decoded": round(neural_decoded, 6),
                    "Residual": round(residual, 6),
                    "Expert_mixture_scaled": round(float(np.sum(gate_weights * expert_stack)), 6),
                    "Gate_entropy": round(gate_entropy, 6),
                    "Gate_top": GATE_NAMES[int(np.argmax(gate_weights))],
                }
                gate_row.update(
                    {
                        f"Gate_{name}": round(float(gate_weights[index]), 6)
                        for index, name in enumerate(GATE_NAMES)
                    }
                )
                gate_row.update(
                    {
                        f"Expert_{name}": round(float(experts[index]), 6)
                        for index, name in enumerate(SCALAR_EXPERT_NAMES)
                    }
                )
                gate_rows.append(gate_row)
        if trace_sample:
            logger.info(
                "    [PRED] actual=%s predicted=%s",
                format_values(sample.query_targets[: min(5, len(sample.query_targets))]),
                format_values(predicted_values[: min(5, len(predicted_values))]),
            )
        if valid_sample_count % int(config.progress_every) == 0 or valid_sample_count == len(test_samples):
            elapsed = time.time() - eval_start
            speed = valid_sample_count / elapsed if elapsed > 0 else 0.0
            remaining = (len(test_samples) - valid_sample_count) / speed if speed > 0 else 0.0
            logger.info(
                "  [eval] samples=%s/%s predictions=%s (%.0fs elapsed, ~%.0fs remaining)",
                valid_sample_count,
                len(test_samples),
                len(detail_rows),
                elapsed,
                remaining,
            )

    pd.DataFrame(detail_rows).to_csv(config.detail_log_csv, index=False)
    if config.save_gate_log:
        pd.DataFrame(gate_rows).to_csv(config.gate_log_csv, index=False)
        logger.info("Gate log CSV: %s (%s rows)", config.gate_log_csv, len(gate_rows))
    result_rows = variable_result_rows(detail_rows, benchmark.columns, benchmark.name, method_label, variant, config)
    results = pd.DataFrame(result_rows)
    results.to_csv(config.output_csv, index=False)
    results.to_csv(config.checkpoint_csv, index=False)

    total_elapsed = time.time() - total_start
    avg_mae = float(results["MAE_scaled"].mean()) if not results.empty else float("nan")
    avg_mse = float(results["MSE_scaled"].mean()) if not results.empty else float("nan")
    global_metrics = compute_global_scaled_metrics(config.detail_log_csv)
    total_predictions = int(global_metrics["Predictions"] if global_metrics else len(detail_rows))

    logger.info("\n%s", "=" * 70)
    logger.info("  %s - %s results", MODEL_NAME, benchmark.display_name)
    logger.info("%s", "=" * 70)
    if not results.empty:
        logger.info(
            "\n%s",
            results[["Variable", "MAE_scaled", "MSE_scaled", "MAE_raw", "Predictions"]].to_string(index=False),
        )
    logger.info("Equal-variable MAE_scaled: %.6f", avg_mae)
    logger.info("Equal-variable MSE_scaled: %.6f", avg_mse)
    if global_metrics is not None:
        logger.info("benchmark-style global MAE_scaled: %.6f", global_metrics["MAE_scaled"])
        logger.info("benchmark-style global MSE_scaled: %.6f", global_metrics["MSE_scaled"])
    logger.info("Total predictions: %s", total_predictions)
    logger.info("Total time: %.1fs", total_elapsed)
    cpu_ms_per_pred = 1000.0 * total_elapsed / total_predictions if total_predictions else float("nan")
    cpu_pred_per_s = total_predictions / total_elapsed if total_elapsed > 0 else float("nan")
    logger.info("CPU ms/pred: %.4f", cpu_ms_per_pred)
    logger.info("CPU pred/s: %.1f", cpu_pred_per_s)

    baseline = APN_BASELINES.get(benchmark.name)
    if baseline is not None and global_metrics is not None:
        mae_delta = baseline["MAE"] - global_metrics["MAE_scaled"]
        mse_delta = baseline["MSE"] - global_metrics["MSE_scaled"]
        logger.info("APN reference: MAE=%.4f, MSE=%.4f", baseline["MAE"], baseline["MSE"])
        logger.info("%s result: MAE=%.4f, MSE=%.4f", MODEL_NAME, global_metrics["MAE_scaled"], global_metrics["MSE_scaled"])
        logger.info("%s delta vs APN reference: MAE=%+.4f, MSE=%+.4f", benchmark.name, mae_delta, mse_delta)

    logger.info("Output CSV: %s", config.output_csv)
    logger.info("Detail log CSV: %s", config.detail_log_csv)
    logger.info("Debug log: %s", config.debug_log)

    summary: dict[str, float | str] = {
        "Dataset": benchmark.name,
        "Method": method_label,
        "Base_method": MODEL_NAME,
        "Variant": variant,
        "Seq_len": config.seq_len,
        "Pred_len": config.pred_len,
        "Architecture_ablation": config.architecture_ablation,
        "Loss_mode": config.loss_mode,
        "History_perturbation": config.history_perturbation,
        "History_keep_fraction": float(config.history_keep_fraction),
        "Equal_variable_MAE_scaled": avg_mae,
        "Equal_variable_MSE_scaled": avg_mse,
        "Predictions": float(total_predictions),
        "Fallback": 0.0,
        "Time_s": total_elapsed,
        "CPU_Total_s": total_elapsed,
        "CPU_ms_per_pred": cpu_ms_per_pred,
        "CPU_pred_per_s": cpu_pred_per_s,
        "Gate_log_rows": float(len(gate_rows)),
    }
    if global_metrics is not None:
        summary.update({"MAE_scaled": global_metrics["MAE_scaled"], "MSE_scaled": global_metrics["MSE_scaled"]})
    return summary
