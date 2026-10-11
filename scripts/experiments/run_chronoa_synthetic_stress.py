#!/usr/bin/env python3
"""Synthetic stress test for the ADEPT neural event graph model."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from adept.experiments.chronoa_forecasting import (  # noqa: E402
    Batch,
    ChronoaEventGraphNet,
    NeuralSample,
    collate_batch,
    masked_loss,
)

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "adept_experiments" / "synthetic_stress"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a synthetic ADEPT neural stress test.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def make_sample(rng: np.random.Generator, key: str, seq_len: int, pred_len: int) -> NeuralSample:
    times = np.arange(seq_len + pred_len, dtype=np.float32)
    driver = np.sin(2 * np.pi * times / 17.0 + rng.uniform(-1.0, 1.0))
    shock = np.zeros_like(times)
    shock[seq_len - 5 : seq_len + pred_len] = rng.normal(0.0, 0.7)
    series0 = driver + 0.15 * np.cumsum(rng.normal(size=len(times))) / np.sqrt(seq_len)
    series1 = np.roll(driver, 3) + shock + rng.normal(0.0, 0.08, size=len(times))
    series2 = 0.5 * series0 - 0.3 * series1 + rng.normal(0.0, 0.05, size=len(times))
    values = np.stack([series0, series1, series2], axis=1).astype(np.float32)

    event_times: list[float] = []
    event_vars: list[int] = []
    event_values: list[float] = []
    for t_index in range(seq_len):
        for var_index in range(3):
            if rng.random() < (0.85 - 0.12 * var_index):
                event_times.append(float(t_index))
                event_vars.append(var_index)
                event_values.append(float(values[t_index, var_index]))

    query_times: list[float] = []
    query_vars: list[int] = []
    query_targets: list[float] = []
    query_steps: list[int] = []
    for step in range(pred_len):
        t_index = seq_len + step
        for var_index in range(3):
            query_times.append(float(t_index))
            query_vars.append(var_index)
            query_targets.append(float(values[t_index, var_index]))
            query_steps.append(step)

    history_by_var = {idx: ([], []) for idx in range(3)}
    for t_value, v_value, x_value in zip(event_times, event_vars, event_values):
        history_by_var[int(v_value)][0].append(float(t_value))
        history_by_var[int(v_value)][1].append(float(x_value))

    return NeuralSample(
        key=key,
        event_times=np.asarray(event_times, dtype=np.float32),
        event_vars=np.asarray(event_vars, dtype=np.int64),
        event_values=np.asarray(event_values, dtype=np.float32),
        query_times=np.asarray(query_times, dtype=np.float32),
        query_vars=np.asarray(query_vars, dtype=np.int64),
        query_targets=np.asarray(query_targets, dtype=np.float32),
        query_steps=np.asarray(query_steps, dtype=np.int64),
        history_by_var=history_by_var,
    )


def make_samples(count: int, seq_len: int, pred_len: int, seed: int) -> list[NeuralSample]:
    rng = np.random.default_rng(seed)
    return [make_sample(rng, f"synthetic_{index}", seq_len, pred_len) for index in range(count)]


def batches(samples: list[NeuralSample], batch_size: int, lookback_span: float, device: object, shuffle_seed: int | None):
    order = np.arange(len(samples))
    if shuffle_seed is not None:
        rng = np.random.default_rng(shuffle_seed)
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        selected = [samples[int(index)] for index in order[start : start + batch_size]]
        yield collate_batch(selected, lookback_span=lookback_span, device=device)


def score(model: object, samples: list[NeuralSample], batch_size: int, lookback_span: float, device: object) -> tuple[float, float]:
    import torch

    model.eval()
    abs_error = 0.0
    sq_error = 0.0
    count = 0
    with torch.no_grad():
        for batch in batches(samples, batch_size, lookback_span, device, shuffle_seed=None):
            prediction = model(batch)
            error = prediction[batch.query_mask] - batch.query_targets[batch.query_mask]
            abs_error += float(torch.sum(torch.abs(error)).cpu())
            sq_error += float(torch.sum(error**2).cpu())
            count += int(batch.query_mask.sum().cpu())
    return abs_error / count, sq_error / count


def main() -> None:
    import torch

    args = parse_args()
    device = torch.device(args.device)
    train = make_samples(args.n_train, args.seq_len, args.pred_len, args.seed)
    test = make_samples(args.n_test, args.seq_len, args.pred_len, args.seed + 1000)
    model = ChronoaEventGraphNet(
        n_variables=3,
        d_model=64,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        delta_kernels=4,
        architecture_ablation="full",
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    rows: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in batches(train, args.batch_size, args.seq_len, device, shuffle_seed=args.seed + epoch):
            prediction = model(batch)
            loss = masked_loss(prediction, batch.query_targets, batch.query_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        mae, mse = score(model, test, args.batch_size, args.seq_len, device)
        rows.append({"Epoch": epoch, "MAE_scaled": mae, "MSE_scaled": mse, "Predictions": args.n_test * args.pred_len * 3})
        print(f"epoch={epoch:03d} mae={mae:.4f} mse={mse:.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "chronoa_edge_synthetic_stress.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Epoch", "MAE_scaled", "MSE_scaled", "Predictions"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
