# -*- coding: utf-8 -*-
"""Calibration demo: AgreementRatio and PlattScaler on synthetic data.

Demonstrates: raw vs calibrated confidence, reliability diagram, ECE.
Expected: ECE decreases after Platt scaling.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

if "--save" in sys.argv or os.environ.get("MPLBACKEND"):
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from nwf import AgreementRatio, Charge, Field, PlattScaler


def ece(confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece_val = 0.0
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            acc_bin = correct[mask].mean()
            conf_bin = confidences[mask].mean()
            ece_val += mask.sum() * np.abs(acc_bin - conf_bin)
    return ece_val / len(confidences)


def reliability_diagram(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10, ax=None
) -> None:
    """Plot reliability diagram."""
    if ax is None:
        ax = plt.gca()
    bins = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        if mask.sum() > 0:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidences[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_accs.append(0)
            bin_confs.append((bins[i] + bins[i + 1]) / 2)
            bin_counts.append(0)
    bin_accs = np.array(bin_accs)
    bin_confs = np.array(bin_confs)
    ax.bar(bin_confs - 0.5 / n_bins, bin_accs, width=0.9 / n_bins, alpha=0.7, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibration demo")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--overlap", type=float, default=0.5, help="Class overlap (std)")
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # 1. Generate overlapping Gaussians (controlled uncertainty)
    n = args.n_train
    X_a = rng.randn(n // 2, 2) * args.overlap + np.array([-0.5, 0.0])
    X_b = rng.randn(n // 2, 2) * args.overlap + np.array([0.5, 0.0])
    X_train = np.vstack([X_a, X_b])
    y_train = np.array([0] * (n // 2) + [1] * (n // 2))

    sigma_fixed = np.full(2, 0.2)
    field = Field()
    for i in range(len(X_train)):
        ch = Charge(z=X_train[i].astype(np.float64), sigma=sigma_fixed.astype(np.float64))
        field.add(ch, labels=[int(y_train[i])], ids=[i])

    ar = AgreementRatio()

    def get_confidences_and_correct(X: np.ndarray, y: np.ndarray, k: int) -> tuple:
        confs, correct_flags = [], []
        for i in range(len(X)):
            q = Charge(z=X[i].astype(np.float64), sigma=sigma_fixed.astype(np.float64))
            _, _, labs = field.search(q, k=k)
            neighbor_labels = labs[0]
            votes = np.bincount(np.array(neighbor_labels).astype(int), minlength=2)
            pred = int(np.argmax(votes))
            conf = ar.predict(neighbor_labels, pred)
            confs.append(conf)
            correct_flags.append(1 if pred == y[i] else 0)
        return np.array(confs), np.array(correct_flags)

    # 2. Validation set for Platt
    n_val = args.n_val
    X_val_a = rng.randn(n_val // 2, 2) * args.overlap + np.array([-0.5, 0.0])
    X_val_b = rng.randn(n_val // 2, 2) * args.overlap + np.array([0.5, 0.0])
    X_val = np.vstack([X_val_a, X_val_b])
    y_val = np.array([0] * (n_val // 2) + [1] * (n_val // 2))

    conf_val, corr_val = get_confidences_and_correct(X_val, y_val, args.k)
    platt = PlattScaler()
    platt.fit(conf_val, corr_val)

    # 3. Test set
    n_test = args.n_test
    X_test_a = rng.randn(n_test // 2, 2) * args.overlap + np.array([-0.5, 0.0])
    X_test_b = rng.randn(n_test // 2, 2) * args.overlap + np.array([0.5, 0.0])
    X_test = np.vstack([X_test_a, X_test_b])
    y_test = np.array([0] * (n_test // 2) + [1] * (n_test // 2))

    conf_test, corr_test = get_confidences_and_correct(X_test, y_test, args.k)
    conf_calibrated = platt.predict(conf_test)

    ece_raw = ece(conf_test, corr_test, args.n_bins)
    ece_cal = ece(conf_calibrated, corr_test, args.n_bins)
    print(f"ECE (raw):        {ece_raw:.4f}")
    print(f"ECE (calibrated): {ece_cal:.4f}")

    # 4. Reliability diagram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    reliability_diagram(conf_test, corr_test, args.n_bins, ax1)
    ax1.set_title(f"Before calibration (ECE={ece_raw:.3f})")
    reliability_diagram(conf_calibrated, corr_test, args.n_bins, ax2)
    ax2.set_title(f"After Platt scaling (ECE={ece_cal:.3f})")
    plt.tight_layout()

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
