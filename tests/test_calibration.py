# -*- coding: utf-8 -*-
import numpy as np

from nwf.calibration import AgreementRatio, PlattScaler


def test_agreement_ratio() -> None:
    ar = AgreementRatio()
    r = ar.predict(neighbor_labels=[0, 0, 1, 0, 0], predicted_label=0)
    assert abs(r - 0.8) < 1e-6


def test_platt_scaler() -> None:
    np.random.seed(42)
    conf = np.random.rand(100)
    labels = (conf + np.random.randn(100) * 0.2 > 0.5).astype(int)
    ps = PlattScaler()
    ps.fit(conf, labels)
    cal = ps.predict(conf[:5])
    assert cal.shape == (5,)
    assert np.all((cal >= 0) & (cal <= 1))
