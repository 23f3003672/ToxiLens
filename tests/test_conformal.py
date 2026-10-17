import numpy as np
import pytest
from src.toxilens.conformal import BinaryConformalPredictor


def test_conformal_empty_fit_raises():
    cp = BinaryConformalPredictor(alpha=0.10)
    with pytest.raises(ValueError):
        cp.fit(np.array([]), np.array([]))


def test_conformal_coverage_guarantee():
    rng = np.random.default_rng(42)
    # Generate synthetic calibration probabilities
    n_cal = 500
    y_cal_true = rng.choice([0, 1], size=n_cal, p=[0.8, 0.2])
    y_cal_prob = np.where(y_cal_true == 1, rng.uniform(0.6, 0.95, size=n_cal), rng.uniform(0.05, 0.4, size=n_cal))

    cp = BinaryConformalPredictor(alpha=0.10)
    cp.fit(y_cal_true, y_cal_prob)

    # Test set
    n_test = 500
    y_test_true = rng.choice([0, 1], size=n_test, p=[0.8, 0.2])
    y_test_prob = np.where(y_test_true == 1, rng.uniform(0.6, 0.95, size=n_test), rng.uniform(0.05, 0.4, size=n_test))

    stats = cp.evaluate_coverage(y_test_true, y_test_prob)
    assert stats["empirical_coverage"] >= 85.0
    assert stats["average_set_size"] <= 2.0


def test_conformal_prediction_set_values():
    cp = BinaryConformalPredictor(alpha=0.10)
    # Realistic calibration distribution with some difficult / borderline cases
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.35, 0.45, 0.60, 0.40, 0.55, 0.70, 0.85, 0.90])
    cp.fit(y_true, y_prob)

    # Extreme probabilities
    p_safe = cp.predict_set(0.01)
    assert 0 in p_safe

    p_toxic = cp.predict_set(0.99)
    assert 1 in p_toxic

    # Borderline / uncertain probability
    p_mid = cp.predict_set(0.50)
    assert len(p_mid) >= 1
