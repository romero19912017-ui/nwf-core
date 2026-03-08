# -*- coding: utf-8 -*-
"""OOD detection via semantic potential on synthetic 2D data.

Demonstrates: potential(r, charges) for in-distribution vs OOD.
Expected: in-distribution has higher potential; AUC > 0.8.
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
from sklearn.metrics import roc_auc_score, roc_curve

from nwf import Charge, Field, potential, potential_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="OOD detection via potential")
    parser.add_argument("--n-in", type=int, default=200, help="In-distribution samples")
    parser.add_argument("--n-ood", type=int, default=200, help="OOD samples")
    parser.add_argument("--box", type=float, default=4.0, help="OOD uniform box half-width")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    # 1. In-distribution: two Gaussians
    n_in = args.n_in
    X_a = rng.randn(n_in // 2, 2) * 0.5 + np.array([-1.0, 0.0])
    X_b = rng.randn(n_in // 2, 2) * 0.5 + np.array([1.0, 0.0])
    X_in = np.vstack([X_a, X_b])

    # 2. Build field from in-distribution
    sigma_fixed = np.full(2, 0.3)
    charges = []
    for i in range(len(X_in)):
        ch = Charge(z=X_in[i].astype(np.float64), sigma=sigma_fixed.astype(np.float64))
        charges.append(ch)
    z_all = np.stack([c.z for c in charges])
    s_all = np.stack([c.sigma for c in charges])

    # 3. OOD: uniform in large box
    X_ood = rng.uniform(-args.box, args.box, (args.n_ood, 2))

    # 4. Compute potential for each point
    phi_in = potential_batch(X_in, z_all, s_all)
    phi_ood = potential_batch(X_ood, z_all, s_all)

    y_true = np.concatenate([np.ones(len(phi_in)), np.zeros(len(phi_ood))])
    y_score = np.concatenate([phi_in, phi_ood])
    auroc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    print(f"AUROC: {auroc:.3f}")

    # 5. Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Contour of potential
    xx = np.linspace(-args.box, args.box, 50)
    yy = np.linspace(-args.box, args.box, 50)
    XX, YY = np.meshgrid(xx, yy)
    grid = np.c_[XX.ravel(), YY.ravel()]
    phi_grid = potential_batch(grid, z_all, s_all).reshape(XX.shape)
    axes[0].contourf(XX, YY, phi_grid, levels=20, cmap="viridis")
    axes[0].scatter(X_in[:, 0], X_in[:, 1], c="C0", s=10, alpha=0.7, label="In")
    axes[0].scatter(X_ood[:, 0], X_ood[:, 1], c="C1", s=10, alpha=0.5, label="OOD")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_title("Potential field and points")
    axes[0].legend()

    # Histograms
    axes[1].hist(phi_in, bins=30, alpha=0.7, label="In", color="C0")
    axes[1].hist(phi_ood, bins=30, alpha=0.7, label="OOD", color="C1")
    axes[1].set_xlabel("Potential")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Potential distribution")
    axes[1].legend()

    # ROC
    axes[2].plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
    axes[2].plot([0, 1], [0, 1], "k--")
    axes[2].set_xlabel("FPR")
    axes[2].set_ylabel("TPR")
    axes[2].set_title("ROC curve")
    axes[2].legend()
    plt.tight_layout()

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
