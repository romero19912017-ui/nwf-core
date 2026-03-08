User Guide
==========

Quick start
-----------

.. code-block:: python

   import numpy as np
   from nwf import Charge, Field, mahalanobis_symmetric

   c1 = Charge(z=np.array([0.0, 0.0]), sigma=np.array([0.1, 0.1]))
   c2 = Charge(z=np.array([1.0, 1.0]), sigma=np.array([0.1, 0.1]))

   field = Field()
   field.add([c1, c2], labels=[0, 1])
   distances, indices, labels = field.search(c1, k=2)

Components
----------

- **Charge** — base object (z, sigma), center and diagonal covariance
- **Field** — container for charges with add/remove/search
- **Metric** — mahalanobis_symmetric, euclidean, cosine
- **Index** — BruteForceIndex (default), FAISSIndex (with optional Mahalanobis rerank)
- **Calibration** — AgreementRatio, PlattScaler
- **Encoders** — VAEEncoder (optional, requires torch)
