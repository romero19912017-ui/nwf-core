# -*- coding: utf-8 -*-
import numpy as np
import pytest

try:
    from nwf.encoders.vae import VAEEncoder

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_vae_encoder() -> None:
    np.random.seed(42)
    X = np.random.rand(200, 32).astype(np.float32)
    enc = VAEEncoder(input_dim=32, latent_dim=8, hidden_dims=(64, 32))
    enc.fit(X, epochs=3, batch_size=64)
    z, sigma = enc.encode(X[:5])
    assert z.shape == (5, 8)
    assert sigma.shape == (5, 8)
    assert np.all(sigma > 0)
