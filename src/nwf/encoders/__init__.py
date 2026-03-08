# -*- coding: utf-8 -*-
try:
    from nwf.encoders.vae import VAEEncoder  # noqa: F401

    __all__ = ["VAEEncoder"]
except ImportError:
    __all__ = []
