import numpy as np
from lightgbm import LGBMClassifier


def test_lightgbm_tiny_fit_predict_smoke():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(32, 8))
    y = rng.integers(0, 2, size=32)

    model = LGBMClassifier(n_estimators=10, max_depth=3, random_state=42)
    model.fit(x, y)
    preds = model.predict(x)

    assert preds.shape == (32,)
