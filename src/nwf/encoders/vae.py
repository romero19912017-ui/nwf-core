# -*- coding: utf-8 -*-
"""Simple VAE encoder for NWF."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _ensure_torch() -> None:
    if not HAS_TORCH:
        raise ImportError("torch required. Install: pip install nwf-core[torch]")


class _VAE(nn.Module):
    """MLP VAE for flat vectors (e.g. MNIST 784)."""

    def __init__(
        self, input_dim: int, latent_dim: int, hidden_dims: Tuple[int, ...] = (512, 256)
    ) -> None:
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(d, h), nn.ReLU()])
            d = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(d, latent_dim)
        self.fc_logvar = nn.Linear(d, latent_dim)
        rev = []
        d = latent_dim
        for h in reversed(hidden_dims):
            rev.extend([nn.Linear(d, h), nn.ReLU()])
            d = h
        rev.append(nn.Linear(d, input_dim))
        rev.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*rev)

    def encode(self, x: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor"]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def decode(self, z: "torch.Tensor") -> "torch.Tensor":
        return self.decoder(z)

    def forward(
        self, x: "torch.Tensor"
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        recon = self.decoder(z)
        return recon, mu, logvar


class VAEEncoder:
    """VAE encoder producing (z, sigma) for NWF charges."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 64,
        hidden_dims: Tuple[int, ...] = (512, 256),
        device: Optional[str] = None,
    ) -> None:
        _ensure_torch()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device or "cpu"
        self._model = _VAE(input_dim, latent_dim, hidden_dims).to(self.device)

    def fit(
        self,
        train_data: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        lr: float = 1e-3,
    ) -> "VAEEncoder":
        """Train VAE on data. train_data: (N, D)."""
        _ensure_torch()
        self._model.train()
        opt = torch.optim.Adam(self._model.parameters(), lr=lr)
        X = torch.FloatTensor(train_data).to(self.device)
        n = len(X)
        for _ in range(epochs):
            perm = np.random.permutation(n)
            for i in range(0, n, batch_size):
                batch = X[perm[i : i + batch_size]]
                recon, mu, logvar = self._model(batch)
                bce = nn.functional.binary_cross_entropy(recon, batch, reduction="sum")
                kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = (bce + kl) / batch.size(0)
                opt.zero_grad()
                loss.backward()
                opt.step()
        self._model.eval()
        return self

    def encode(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode to (z, sigma). sigma = exp(0.5 * logvar)."""
        _ensure_torch()
        self._model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(x).to(self.device)
            if t.dim() == 1:
                t = t.unsqueeze(0)
            mu, logvar = self._model.encode(t)
            sigma = torch.exp(0.5 * logvar)
            z = mu.cpu().numpy()
            s = sigma.cpu().numpy()
        return z.squeeze(), s.squeeze()
