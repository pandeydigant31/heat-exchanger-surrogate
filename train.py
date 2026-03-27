#!/usr/bin/env python3
"""Heat Exchanger Neural Surrogate — Training Pipeline

Trains a Fourier Neural Operator (FNO) to predict temperature profiles
in a shell-and-tube heat exchanger, replacing the ε-NTU calculation.

Pipeline:
    1. Generate training data: vary operating conditions → ε-NTU + CoolProp solve
    2. Train FNO to map (inlet temps, flows, geometry) → T(x) profiles
    3. Benchmark: FNO inference vs analytical calculation
    4. Evaluate accuracy on held-out test set

Usage:
    cd heat-exchanger-surrogate
    python train.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from src.physics.hx_model import generate_dataset, solve_hx
from src.models.fno import FNO1d

OUTPUTS = PROJECT_ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
MODELS = OUTPUTS / "models"

N_SAMPLES = 2000
N_POINTS = 50
N_EPOCHS = 200
BATCH_SIZE = 64
LR = 1e-3
PATIENCE = 25


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Heat Exchanger Neural Surrogate (FNO)")
    print("=" * 60)

    # ── Stage 1: Generate data ─────────────────────────────────
    print(f"\n[1/4] Generating {N_SAMPLES} heat exchanger scenarios...")
    data = generate_dataset(n_samples=N_SAMPLES, n_points=N_POINTS, seed=42)
    n_valid = len(data["inputs"])
    print(f"  Valid scenarios: {n_valid}")
    print(f"  Input dims: {data['inputs'].shape[1]} ({', '.join(data['input_names'])})")
    print(f"  Profile points: {N_POINTS}")
    print(f"  Output: T_hot(x) + T_cold(x) profiles")

    # Split 70/15/15
    n_train = int(0.7 * n_valid)
    n_val = int(0.15 * n_valid)

    X_all = data["inputs"]
    Y_hot = data["T_hot_profiles"]
    Y_cold = data["T_cold_profiles"]
    # Target: both profiles stacked → (n_samples, 2, n_points)
    Y_all = np.stack([Y_hot, Y_cold], axis=1)

    X_train, X_val, X_test = X_all[:n_train], X_all[n_train:n_train + n_val], X_all[n_train + n_val:]
    Y_train, Y_val, Y_test = Y_all[:n_train], Y_all[n_train:n_train + n_val], Y_all[n_train + n_val:]

    # Normalize inputs
    x_scaler = StandardScaler().fit(X_train)
    X_train_n = torch.tensor(x_scaler.transform(X_train), dtype=torch.float32)
    X_val_n = torch.tensor(x_scaler.transform(X_val), dtype=torch.float32)
    X_test_n = torch.tensor(x_scaler.transform(X_test), dtype=torch.float32)

    # Normalize outputs (per-channel)
    y_mean = Y_train.mean(axis=(0, 2), keepdims=True)
    y_std = Y_train.std(axis=(0, 2), keepdims=True) + 1e-6
    Y_train_n = torch.tensor((Y_train - y_mean) / y_std, dtype=torch.float32)
    Y_val_n = torch.tensor((Y_val - y_mean) / y_std, dtype=torch.float32)
    Y_test_n = torch.tensor((Y_test - y_mean) / y_std, dtype=torch.float32)

    print(f"  Split: {len(X_train)} train / {len(X_val)} val / {len(X_test)} test")

    # ── Stage 2: Train FNO ─────────────────────────────────────
    print(f"\n[2/4] Training Fourier Neural Operator...")

    n_input = X_train.shape[1]
    model = FNO1d(n_input=n_input, n_output=2, n_points=N_POINTS,
                  width=64, modes=16, n_layers=4, dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: FNO-1D, 4 Fourier layers, 16 modes, 64 width")
    print(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS, eta_min=1e-5)
    criterion = torch.nn.MSELoss()

    best_val = float("inf")
    best_state = None
    wait = 0
    losses = {"train": [], "val": []}

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(X_train_n))
        epoch_loss = 0
        n_batches = 0

        for start in range(0, len(X_train_n), BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            pred = model(X_train_n[idx])
            loss = criterion(pred, Y_train_n[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        train_loss = epoch_loss / n_batches

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_n)
            val_loss = criterion(val_pred, Y_val_n).item()

        losses["train"].append(train_loss)
        losses["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        if epoch % 20 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{N_EPOCHS}: train={train_loss:.4f} val={val_loss:.4f}")

        if wait >= PATIENCE:
            print(f"    Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)

    # ── Stage 3: Evaluate ──────────────────────────────────────
    print(f"\n[3/4] Evaluating on test set...")

    model.eval()
    with torch.no_grad():
        test_pred_n = model(X_test_n).numpy()

    # Unnormalize
    test_pred = test_pred_n * y_std + y_mean
    test_true = Y_test

    # MAE per channel (in °C)
    mae_hot = np.mean(np.abs(test_pred[:, 0] - test_true[:, 0]))
    mae_cold = np.mean(np.abs(test_pred[:, 1] - test_true[:, 1]))
    mae_overall = (mae_hot + mae_cold) / 2

    # Outlet temperature accuracy (last point)
    T_hot_out_pred = test_pred[:, 0, -1]
    T_hot_out_true = test_true[:, 0, -1]
    T_cold_out_pred = test_pred[:, 1, 0]  # counterflow: cold outlet is at x=0
    T_cold_out_true = test_true[:, 1, 0]

    mae_hot_out = np.mean(np.abs(T_hot_out_pred - T_hot_out_true))
    mae_cold_out = np.mean(np.abs(T_cold_out_pred - T_cold_out_true))

    print(f"  Profile MAE (hot):  {mae_hot:.3f} °C")
    print(f"  Profile MAE (cold): {mae_cold:.3f} °C")
    print(f"  Overall profile MAE: {mae_overall:.3f} °C")
    print(f"  Outlet MAE (hot):  {mae_hot_out:.3f} °C")
    print(f"  Outlet MAE (cold): {mae_cold_out:.3f} °C")

    # Benchmark speed
    print(f"\n  Benchmarking speed...")
    n_runs = 100

    # FNO batch inference
    start = time.perf_counter()
    for _ in range(n_runs):
        with torch.no_grad():
            _ = model(X_test_n[:32])
    fno_time = (time.perf_counter() - start) / n_runs

    # Analytical (single solve)
    start = time.perf_counter()
    for _ in range(min(n_runs, 50)):
        for j in range(32):
            solve_hx(70, 20, 2.0, 2.0, n_points=N_POINTS)
    analytical_time = (time.perf_counter() - start) / min(n_runs, 50)

    speedup = analytical_time / fno_time
    print(f"    Analytical (32 solves): {analytical_time*1000:.1f}ms")
    print(f"    FNO (batch of 32):      {fno_time*1000:.2f}ms")
    print(f"    Speedup: {speedup:.0f}x")

    results = {
        "profile_mae_hot_C": float(mae_hot),
        "profile_mae_cold_C": float(mae_cold),
        "outlet_mae_hot_C": float(mae_hot_out),
        "outlet_mae_cold_C": float(mae_cold_out),
        "analytical_ms": analytical_time * 1000,
        "fno_ms": fno_time * 1000,
        "speedup": speedup,
        "n_params": n_params,
        "n_train": len(X_train),
    }

    # ── Stage 4: Figures ───────────────────────────────────────
    print(f"\n[4/4] Generating figures...")

    # 1. Sample profile comparisons
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    x_pos = np.linspace(0, 1, N_POINTS)

    for col in range(3):
        idx = col * (len(X_test) // 4)
        ax = axes[0, col]
        ax.plot(x_pos, test_true[idx, 0], "r-", lw=2, label="Analytical (hot)")
        ax.plot(x_pos, test_pred[idx, 0], "r--", lw=2, label="FNO (hot)")
        ax.plot(x_pos, test_true[idx, 1], "b-", lw=2, label="Analytical (cold)")
        ax.plot(x_pos, test_pred[idx, 1], "b--", lw=2, label="FNO (cold)")
        ax.set_xlabel("Position x/L")
        ax.set_ylabel("Temperature (°C)")
        T_hi = X_test[idx, 0]
        T_ci = X_test[idx, 1]
        ax.set_title(f"T_h={T_hi:.0f}°C, T_c={T_ci:.0f}°C")
        if col == 0:
            ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        ax = axes[1, col]
        err_hot = np.abs(test_pred[idx, 0] - test_true[idx, 0])
        err_cold = np.abs(test_pred[idx, 1] - test_true[idx, 1])
        ax.plot(x_pos, err_hot, "r-", lw=1.5, label="|Error| hot")
        ax.plot(x_pos, err_cold, "b-", lw=1.5, label="|Error| cold")
        ax.set_xlabel("Position x/L")
        ax.set_ylabel("Absolute Error (°C)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"FNO Heat Exchanger Surrogate — Profile Predictions\n"
                 f"Overall MAE: {mae_overall:.3f}°C", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "profile_comparison.png", dpi=150)
    plt.close()

    # 2. Outlet temperature scatter
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ax.scatter(T_hot_out_true, T_hot_out_pred, alpha=0.3, s=10, c="red")
    lo, hi = T_hot_out_true.min(), T_hot_out_true.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("True T_hot_out (°C)")
    ax.set_ylabel("Predicted T_hot_out (°C)")
    ax.set_title(f"Hot Outlet (MAE={mae_hot_out:.2f}°C)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.scatter(T_cold_out_true, T_cold_out_pred, alpha=0.3, s=10, c="blue")
    lo, hi = T_cold_out_true.min(), T_cold_out_true.max()
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlabel("True T_cold_out (°C)")
    ax.set_ylabel("Predicted T_cold_out (°C)")
    ax.set_title(f"Cold Outlet (MAE={mae_cold_out:.2f}°C)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "outlet_scatter.png", dpi=150)
    plt.close()

    # 3. Training loss
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses["train"], label="Train", lw=1.5)
    ax.plot(losses["val"], label="Validation", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("FNO Training Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "training_loss.png", dpi=150)
    plt.close()

    # 4. Speed benchmark
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["ε-NTU\n(32 solves)", "FNO\n(batch 32)"],
           [analytical_time * 1000, fno_time * 1000],
           color=["#4A90D9", "#E74C3C"])
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Inference Speed: FNO is {speedup:.0f}x faster")
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate([analytical_time * 1000, fno_time * 1000]):
        ax.text(i, v + 0.5, f"{v:.1f}ms", ha="center")
    plt.tight_layout()
    plt.savefig(FIGURES / "speed_benchmark.png", dpi=150)
    plt.close()

    # Save
    with open(MODELS / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    torch.save({
        "model_state": model.state_dict(),
        "x_scaler_mean": x_scaler.mean_.tolist(),
        "x_scaler_scale": x_scaler.scale_.tolist(),
        "y_mean": y_mean.tolist(),
        "y_std": y_std.tolist(),
        "n_input": n_input,
        "n_points": N_POINTS,
    }, MODELS / "fno.pt")

    print(f"\n  Saved:")
    for p in [FIGURES / "profile_comparison.png", FIGURES / "outlet_scatter.png",
              FIGURES / "training_loss.png", FIGURES / "speed_benchmark.png",
              MODELS / "results.json"]:
        print(f"    {p}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
