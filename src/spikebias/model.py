"""Model of the sorting accuracy of ground truth neurons from their features.

A binomial generalised linear model (logit link) of the accuracy, fitted by coordinate descent
(statsmodels' elastic net, L1 weight 1, penalty alpha) and refitted for one iteration from these weights
to get standard errors. Continuous features are replaced by their percentile rank among the training
neurons (transform="rank") or kept (transform="zscore"); all features are then z-scored with the
training statistics. Performance is McFadden's pseudo R-squared on held-out neurons, from statsmodels' binomial
log-likelihood (as in the paper).
"""

import random
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

CONTINUOUS = ["firing_rate", "distance", "snr", "spatial_extent", "exc_mini_frequency", "dynamics_holding_current"]


def binomial_loglike(y, mu) -> float:
    """Log-likelihood of accuracies y under predicted accuracies mu (statsmodels' binomial family, as in the paper)."""
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, 1 - 1e-12)
    return float(sm.families.Binomial().loglike(np.asarray(y, dtype=float), mu, scale=1.0))


class AccuracyModel:
    """Binomial GLM of the sorting accuracy.

    Args:
        transform: "rank" (percentile rank of the continuous features, then z-score) or "zscore"
        continuous: continuous features (the others, e.g. one-hot columns, are only z-scored)
        alpha: penalty weight of the elastic net (0: no penalty)
        L1_wt: share of the L1 penalty (1: lasso)
        maxiter, cnvrg_tol: convergence settings of the elastic net
    """

    def __init__(self, transform: str = "rank", continuous=CONTINUOUS, alpha: float = 0.0, L1_wt: float = 1.0,
                 maxiter: int = 100, cnvrg_tol: float = 1e-10):
        if transform not in ("rank", "zscore"):
            raise ValueError('transform must be "rank" or "zscore"')
        self.transform = transform
        self.continuous = list(continuous)
        self.alpha, self.L1_wt, self.maxiter, self.cnvrg_tol = alpha, L1_wt, maxiter, cnvrg_tol

    def _fit_scaling(self, X: pd.DataFrame):
        self.features_ = list(X.columns)
        self.ranked_ = [c for c in X.columns if c in self.continuous] if self.transform == "rank" else []
        self.train_values_ = {c: np.sort(X[c].values.astype(float)) for c in self.ranked_}
        Z = self._rank(X)
        self.mean_, self.std_ = Z.mean(), Z.std(ddof=0).replace(0, 1.0)

    def _rank(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = X[self.features_].astype(float).copy()
        for c in self.ranked_:
            values = self.train_values_[c]
            Z[c] = np.searchsorted(values, Z[c].values, side="right") / len(values)
        return Z

    def _exog(self, X: pd.DataFrame) -> pd.DataFrame:
        Z = (self._rank(X) - self.mean_) / self.std_
        return sm.add_constant(Z, has_constant="add")

    def fit(self, X: pd.DataFrame, y):
        """Fit on the features X (one row per neuron) and the accuracies y (between 0 and 1)."""
        y = np.asarray(y, dtype=float)
        if X.isna().any().any() or not np.isfinite(X.values.astype(float)).all() or np.isnan(y).any():
            raise ValueError("drop the neurons with a missing feature (e.g. neurons without spikes) first")
        self._fit_scaling(X)
        exog = self._exog(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            glm = sm.GLM(y, exog, family=sm.families.Binomial())
            start = glm.fit_regularized(method="elastic_net", alpha=self.alpha, L1_wt=self.L1_wt,
                                        maxiter=self.maxiter, cnvrg_tol=self.cnvrg_tol)
            self.result_ = glm.fit(start_params=start.params, maxiter=1)
        self.null_accuracy_ = float(y.mean())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicted sorting accuracy of each neuron."""
        return np.asarray(self.result_.predict(self._exog(X)))

    def score(self, X: pd.DataFrame, y) -> float:
        """McFadden pseudo R-squared: 1 - LL(model) / LL(null), the null predicting the mean training accuracy."""
        ll = binomial_loglike(y, self.predict(X))
        ll_null = binomial_loglike(y, np.full(len(y), self.null_accuracy_))
        if ll_null == 0:
            return np.nan
        return 1 - ll / ll_null

    @property
    def weights(self) -> pd.DataFrame:
        """Weights (log-odds change per standard deviation of the transformed feature), standard errors, p-values."""
        return pd.DataFrame({"weight": self.result_.params, "se": self.result_.bse, "p_value": self.result_.pvalues})


def cross_validate(model: AccuracyModel, X: pd.DataFrame, y, n_splits: int = 100, train_size: float = 0.75,
                   seed: int = 0) -> np.ndarray:
    """Pseudo R-squared on the test neurons of random splits (NaN when the fit fails on a split).

    Split k draws round(train_size * n) training neurons with random.Random(seed + k), as in the paper.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    n_train = int(np.round(train_size * n))
    r2 = []
    for k in range(n_splits):
        train = random.Random(seed + k).sample(range(n), n_train)
        test = sorted(set(range(n)) - set(train))
        try:
            model.fit(X.iloc[train], y[train])
            r2.append(model.score(X.iloc[test], y[test]))
        except Exception:
            r2.append(np.nan)
    return np.array(r2, dtype=float)


def summarize(r2) -> dict:
    """Median pseudo R-squared over the splits that fit, its 95% interval (1.96 s.e.m.) and their number."""
    r2 = np.asarray(r2, dtype=float)
    finite = r2[np.isfinite(r2)]
    return {"median": float(np.median(finite)) if len(finite) else np.nan,
            "ci95": float(1.96 * np.std(finite) / np.sqrt(len(finite))) if len(finite) else np.nan,
            "n_splits": int(len(finite))}
