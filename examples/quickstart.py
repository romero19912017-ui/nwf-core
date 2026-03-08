# -*- coding: utf-8 -*-
"""Quickstart: minimal NWF workflow with synthetic 2D data.

Demonstrates: Charge, Field, k-NN classification, AgreementRatio confidence.
Expected: classification of 2 classes with ~95%+ accuracy, confidence scores.
"""
from __future__ import annotations

import argparse
import os
import sys

# Non-interactive backend when --save used (no display)
if "--save" in sys.argv or os.environ.get("MPLBACKEND"):
    import matplotlib
    matplotlib.use("Agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nwf import AgreementRatio, Charge, Field


def main() -> None:
    parser = argparse.ArgumentParser(description="NWF quickstart with synthetic 2D data")
    parser.add_argument("--k", type=int, default=5, help="Number of neighbors for k-NN")
    parser.add_argument("--n-train", type=int, default=100, help="Samples per class (train)")
    parser.add_argument("--n-test", type=int, default=50, help="Test samples")
    parser.add_argument("--sigma", type=float, default=0.1, help="Fixed sigma for charges")
    parser.add_argument("--save", type=str, default="", help="Save plot to path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # 1. Generate synthetic data: 2 classes (Gaussians with different centers)
    print("Generating synthetic 2D data...")
    n_a, n_b = args.n_train, args.n_train
    X_a = rng.randn(n_a, 2) * 0.5 + np.array([-1.0, 0.0])
    X_b = rng.randn(n_b, 2) * 0.5 + np.array([1.0, 0.0])
    X_train = np.vstack([X_a, X_b])
    y_train = np.array([0] * n_a + [1] * n_b)

    # 2. Create charges: z = coordinates, sigma = fixed small value
    sigma_fixed = np.full(2, args.sigma)
    field = Field()
    for i in range(len(X_train)):
        ch = Charge(z=X_train[i].astype(np.float64), sigma=sigma_fixed.astype(np.float64))
        field.add(ch, labels=[int(y_train[i])], ids=[i])

    # 3. Generate test point and classify
    X_test = rng.randn(args.n_test, 2) * 0.5
    y_test = (X_test[:, 0] > 0).astype(int)  # true boundary: x=0

    ar = AgreementRatio()
    correct = 0
    confidences = []

    for i in range(len(X_test)):
        q = Charge(z=X_test[i].astype(np.float64), sigma=sigma_fixed.astype(np.float64))
        dist, idx, labs = field.search(q, k=args.k)
        neighbor_labels = labs[0]
        votes = np.bincount(np.array(neighbor_labels).astype(int), minlength=2)
        pred = int(np.argmax(votes))
        conf = ar.predict(neighbor_labels, pred)
        confidences.append(conf)
        if pred == y_test[i]:
            correct += 1

    acc = correct / len(X_test)
    print(f"Accuracy (k={args.k}): {acc:.3f}")
    print(f"Mean confidence: {np.mean(confidences):.3f}")

    # 4. Visualize: train points, test point, neighbors
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X_a[:, 0], X_a[:, 1], c="C0", alpha=0.6, label="Class A (train)")
    ax.scatter(X_b[:, 0], X_b[:, 1], c="C1", alpha=0.6, label="Class B (train)")
    ax.scatter(X_test[:, 0], X_test[:, 1], c="gray", alpha=0.4, s=20, label="Test")
    ax.axvline(x=0, color="k", linestyle="--", alpha=0.3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"NWF Quickstart: 2 classes, accuracy={acc:.2%}")
    ax.legend()
    ax.set_aspect("equal")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
