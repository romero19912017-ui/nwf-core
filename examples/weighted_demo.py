# -*- coding: utf-8 -*-
"""Weighted charges: alpha in potential and weighted k-NN voting.

Demonstrates axiom 4 (superposition with weights). Class A has alpha=2, class B alpha=1.
Shows how weighted voting shifts the decision boundary vs standard voting.
Run: python weighted_demo.py [--save results/weighted.png]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from nwf import Charge, Field, potential, potential_batch

if "--save" in sys.argv or os.environ.get("MPLBACKEND"):
    import matplotlib
    matplotlib.use("Agg")


def weighted_vote(neighbor_labels: np.ndarray, neighbor_alphas: np.ndarray, n_classes: int) -> int:
    """Weighted majority vote: sum alpha for each class, return argmax."""
    votes = np.zeros(n_classes)
    for lab, a in zip(neighbor_labels, neighbor_alphas):
        votes[int(lab)] += a
    return int(np.argmax(votes))


def main() -> None:
    parser = argparse.ArgumentParser(description="Weighted charges demo (axiom 4)")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--alpha-a", type=float, default=2.0)
    parser.add_argument("--alpha-b", type=float, default=1.0)
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    sigma_fixed = np.full(2, 0.1)

    # Class A (alpha_a) and B (alpha_b)
    X_a = rng.randn(args.n_train, 2) * 0.5 + np.array([-1.0, 0.0])
    X_b = rng.randn(args.n_train, 2) * 0.5 + np.array([1.0, 0.0])
    X_train = np.vstack([X_a, X_b])
    y_train = np.array([0] * len(X_a) + [1] * len(X_b))

    field = Field()
    for i in range(len(X_train)):
        alpha = args.alpha_a if y_train[i] == 0 else args.alpha_b
        ch = Charge(z=X_train[i].astype(np.float64), sigma=sigma_fixed.astype(np.float64), alpha=alpha)
        field.add(ch, labels=[int(y_train[i])], ids=[i])

    # Grid for visualization
    xx = np.linspace(-2.5, 2.5, 50)
    yy = np.linspace(-1.5, 1.5, 40)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]
    z_all = np.stack([c.z for c in field.get_charges()], axis=0)
    s_all = np.stack([c.sigma for c in field.get_charges()], axis=0)
    alpha_all = np.array([c.alpha for c in field.get_charges()])
    phi_grid = potential_batch(grid, z_all, s_all, alpha_all=alpha_all).reshape(XX.shape)

    # Test points: classify with standard vs weighted vote
    X_test = rng.randn(200, 2) * 0.6
    y_true = (X_test[:, 0] > 0).astype(int)
    pred_std = []
    pred_weighted = []
    for i in range(len(X_test)):
        q = Charge(z=X_test[i].astype(np.float64), sigma=sigma_fixed.astype(np.float64))
        dist, idx, labs = field.search(q, k=args.k)
        neighbor_labels = np.array(labs[0]).astype(int)
        indices = np.atleast_1d(idx)
        if indices.ndim > 1:
            indices = indices[0]
        neighbor_alphas = np.array([field[int(ii)].alpha for ii in indices])
        votes_std = np.bincount(neighbor_labels, minlength=2)
        pred_std.append(int(np.argmax(votes_std)))
        pred_weighted.append(weighted_vote(neighbor_labels, neighbor_alphas, 2))
    acc_std = np.mean(np.array(pred_std) == y_true)
    acc_weighted = np.mean(np.array(pred_weighted) == y_true)
    print(f"Standard vote accuracy: {acc_std:.3f}")
    print(f"Weighted vote (alpha) accuracy: {acc_weighted:.3f}")

    if args.save:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].contourf(XX, YY, phi_grid, levels=20, cmap="viridis")
        axes[0].scatter(X_a[:, 0], X_a[:, 1], c="C0", alpha=0.6, s=20, label=f"A (alpha={args.alpha_a})")
        axes[0].scatter(X_b[:, 0], X_b[:, 1], c="C1", alpha=0.6, s=20, label=f"B (alpha={args.alpha_b})")
        axes[0].set_xlabel("x1")
        axes[0].set_ylabel("x2")
        axes[0].set_title("Potential (weighted by alpha)")
        axes[0].legend()
        axes[0].set_aspect("equal")

        pred_grid_std = []
        pred_grid_w = []
        for pt in grid:
            q = Charge(z=pt.astype(np.float64), sigma=sigma_fixed.astype(np.float64))
            _, idx, labs = field.search(q, k=args.k)
            nl = np.array(labs[0]).astype(int)
            indices = np.atleast_1d(idx)
            if indices.ndim > 1:
                indices = indices[0]
            na = np.array([field[int(ii)].alpha for ii in indices])
            pred_grid_std.append(int(np.argmax(np.bincount(nl, minlength=2))))
            pred_grid_w.append(weighted_vote(nl, na, 2))
        dec_std = np.array(pred_grid_std).reshape(XX.shape)
        dec_w = np.array(pred_grid_w).reshape(XX.shape)
        axes[1].contourf(XX, YY, dec_w, levels=[-0.5, 0.5, 1.5], colors=["C0", "C1"], alpha=0.3)
        axes[1].contour(XX, YY, dec_std, levels=[0.5], colors="gray", linestyles="--")
        axes[1].scatter(X_a[:, 0], X_a[:, 1], c="C0", alpha=0.5, s=15)
        axes[1].scatter(X_b[:, 0], X_b[:, 1], c="C1", alpha=0.5, s=15)
        axes[1].set_xlabel("x1")
        axes[1].set_ylabel("x2")
        axes[1].set_title("Decision: filled=weighted, dashed=standard")
        axes[1].set_aspect("equal")
        plt.tight_layout()
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {args.save}")
    print("Done.")


if __name__ == "__main__":
    main()
