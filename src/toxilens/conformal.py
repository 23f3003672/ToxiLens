from typing import Dict, List, Optional, Set, Tuple
import numpy as np


class BinaryConformalPredictor:
    """
    Split Conformal Prediction for binary classification written from scratch with NumPy.
    Provides mathematically guaranteed coverage at a chosen significance level (1 - alpha).
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: Desired error rate (default 0.10 for 90% target coverage).
        """
        self.alpha = alpha
        self.quantile: Optional[float] = None
        self.n_cal: int = 0

    def fit(self, y_cal_true: np.ndarray, y_cal_prob: np.ndarray) -> None:
        """
        Calibrate on held-out calibration set.
        Non-conformity score: s_i = 1 - P(true_class)
        """
        y_true = np.asarray(y_cal_true, dtype=int)
        y_prob = np.asarray(y_cal_prob, dtype=float)

        self.n_cal = len(y_true)
        if self.n_cal == 0:
            raise ValueError("Calibration set cannot be empty.")

        # Non-conformity scores: 1 - P(y_true)
        # If true class is 1: 1 - prob
        # If true class is 0: prob (which is 1 - (1 - prob))
        scores = np.where(y_true == 1, 1.0 - y_prob, y_prob)

        # Conformal quantile: ceil((n + 1) * (1 - alpha)) / n
        level = np.ceil((self.n_cal + 1) * (1.0 - self.alpha)) / self.n_cal
        level = min(1.0, max(0.0, level))

        # Standard empirical quantile
        self.quantile = float(np.quantile(scores, level, method="higher"))

    def predict_set(self, y_prob: float) -> Set[int]:
        """
        Generate prediction set {0}, {1}, {0, 1}, or {} for a single probability.
        """
        if self.quantile is None:
            raise RuntimeError("Conformal predictor must be calibrated with .fit() first.")

        p1 = float(y_prob)
        p0 = 1.0 - p1

        prediction_set = set()
        # Include 0 if non-conformity score for class 0 <= quantile
        if (1.0 - p0) <= self.quantile:
            prediction_set.add(0)
        # Include 1 if non-conformity score for class 1 <= quantile
        if (1.0 - p1) <= self.quantile:
            prediction_set.add(1)

        return prediction_set

    def predict_sets(self, y_probs: np.ndarray) -> List[Set[int]]:
        """
        Generate prediction sets for an array of test probabilities.
        """
        return [self.predict_set(p) for p in y_probs]

    def evaluate_coverage(self, y_test_true: np.ndarray, y_test_probs: np.ndarray) -> Dict[str, float]:
        """
        Compute empirical coverage and set size statistics.
        """
        y_true = np.asarray(y_test_true, dtype=int)
        prediction_sets = self.predict_sets(y_test_probs)

        covered = [y in p_set for y, p_set in zip(y_true, prediction_sets)]
        set_sizes = [len(p_set) for p_set in prediction_sets]

        both_count = sum(1 for s in set_sizes if s == 2)
        single_count = sum(1 for s in set_sizes if s == 1)
        empty_count = sum(1 for s in set_sizes if s == 0)

        return {
            "target_coverage": round((1.0 - self.alpha) * 100.0, 1),
            "empirical_coverage": round(float(np.mean(covered)) * 100.0, 2),
            "average_set_size": round(float(np.mean(set_sizes)), 2),
            "uncertain_pct (both)": round((both_count / len(y_true)) * 100.0, 1),
            "singleton_pct": round((single_count / len(y_true)) * 100.0, 1),
            "empty_pct": round((empty_count / len(y_true)) * 100.0, 1),
        }
