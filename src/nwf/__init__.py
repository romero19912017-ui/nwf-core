# -*- coding: utf-8 -*-
"""Neural Weight Fields (NWF) - Core library."""

from nwf.calibration import AgreementRatio, Calibrator, PlattScaler
from nwf.charge import Charge
from nwf.field import Field
from nwf.index import BruteForceIndex, Index
from nwf.metric import cosine, euclidean, mahalanobis_symmetric
from nwf.potential import potential, potential_batch

__version__ = "0.2.5"
__all__ = [
    "Charge",
    "Field",
    "Index",
    "BruteForceIndex",
    "mahalanobis_symmetric",
    "euclidean",
    "cosine",
    "potential",
    "potential_batch",
    "AgreementRatio",
    "PlattScaler",
    "Calibrator",
]

try:
    from nwf.index_faiss import FAISSIndex  # noqa: F401

    __all__.append("FAISSIndex")
except ImportError:
    pass

try:
    from nwf.encoders import VAEEncoder  # noqa: F401

    __all__.append("VAEEncoder")
except ImportError:
    pass
