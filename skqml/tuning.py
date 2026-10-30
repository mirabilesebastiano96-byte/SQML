"""Hyper-parameter selection: cross-validation splitters, scorers, grid / random / successive-halving search.

Every skqml estimator exposes its hyper-parameters through ``get_params`` / ``set_params`` (constructor
arguments, nested with ``__`` as in ``kernel__reps``), so any of them can be tuned:

>>> search = GridSearchCV(VQCClassifier(), {"layers": [1, 2, 3], "lr": [0.01, 0.05]}, cv=3).fit(X, y)
>>> search.best_params_, search.best_score_
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

from .models.base import ClassifierMixin, clone

__all__ = ["KFold", "StratifiedKFold", "TimeSeriesSplit", "HoldOut", "cross_val_score", "get_scorer",
           "GridSearchCV", "RandomSearchCV", "SuccessiveHalvingSearch", "Uniform", "LogUniform", "IntUniform",
           "Choice", "hyperparameters", "ClusterSearch"]


# ============================================================================ splitters
class KFold:
    def __init__(self, n_splits=5, shuffle=True, seed=0):
        self.n_splits, self.shuffle, self.seed = n_splits, shuffle, seed

    def split(self, x, y=None):
        n = len(x)
        idx = np.random.default_rng(self.seed).permutation(n) if self.shuffle else np.arange(n)
        for fold in np.array_split(idx, self.n_splits):
            yield np.setdiff1d(idx, fold, assume_unique=True), fold


class StratifiedKFold(KFold):
    """Each fold keeps the class proportions of ``y``."""

    def split(self, x, y=None):
        y = np.asarray(y)
        rng = np.random.default_rng(self.seed)
        folds = [[] for _ in range(self.n_splits)]
        for c in np.unique(y):
            members = np.where(y == c)[0]
            if self.shuffle:
                members = rng.permutation(members)
            for k, part in enumerate(np.array_split(members, self.n_splits)):
                folds[k].extend(part.tolist())
        all_idx = np.arange(len(y))
        for fold in folds:
            fold = np.array(sorted(fold), dtype=int)
            yield np.setdiff1d(all_idx, fold, assume_unique=True), fold


class TimeSeriesSplit:
    """Expanding window: train on the past, validate on the next block. ``gap`` drops samples in between
    (useful when windows overlap)."""

    def __init__(self, n_splits=5, gap=0, test_size=None):
        self.n_splits, self.gap, self.test_size = n_splits, gap, test_size

    def split(self, x, y=None):
        n = len(x)
        size = self.test_size or n // (self.n_splits + 1)
        for k in range(self.n_splits):
            test_start = n - (self.n_splits - k) * size
            train_end = test_start - self.gap
            if train_end <= 0:
                continue
            yield np.arange(train_end), np.arange(test_start, min(test_start + size, n))


class HoldOut:
    """A single train/validation split; ``shuffle=False`` keeps time order (last part = validation)."""

    def __init__(self, validation_fraction=0.2, shuffle=False, seed=0):
        self.validation_fraction, self.shuffle, self.seed = validation_fraction, shuffle, seed

    def split(self, x, y=None):
        n = len(x)
        idx = np.random.default_rng(self.seed).permutation(n) if self.shuffle else np.arange(n)
        cut = int(n * (1 - self.validation_fraction))
        yield idx[:cut], idx[cut:]


def _resolve_cv(cv, estimator):
    if cv is None:
        cv = 5
    if isinstance(cv, int):
        return StratifiedKFold(cv) if isinstance(estimator, ClassifierMixin) else KFold(cv)
    return cv


def _take(data, idx):
    if data is None:
        return None
    if isinstance(data, np.ndarray):
        return data[idx]
    return [data[i] for i in idx]


# ============================================================================ scoring
def get_scorer(scoring):
    """Name or callable(estimator, X, y) -> float, where larger is better."""
    if callable(scoring):
        return scoring
    if scoring is None:
        return lambda est, x, y: est.score(x, y)

    def pred(est, x):
        return np.asarray(est.predict(x))

    table = {
        "accuracy": lambda est, x, y: float(np.mean(pred(est, x) == np.asarray(y))),
        "r2": lambda est, x, y: float(1 - np.sum((pred(est, x) - y) ** 2) / np.sum((y - np.mean(y)) ** 2)),
        "neg_mse": lambda est, x, y: -float(np.mean((pred(est, x) - y) ** 2)),
        "neg_rmse": lambda est, x, y: -float(np.sqrt(np.mean((pred(est, x) - y) ** 2))),
        "neg_mae": lambda est, x, y: -float(np.mean(np.abs(pred(est, x) - y))),
        "f1_macro": lambda est, x, y: _f1(np.asarray(y), pred(est, x)),
        "log_likelihood": lambda est, x, y: float(np.mean(est.log_likelihood(x, y))),
    }
    if scoring not in table:
        raise ValueError(f"unknown scoring '{scoring}'. Options: {sorted(table)} or a callable")
    return table[scoring]


def _f1(y, p):
    scores = []
    for c in np.unique(y):
        tp, fp, fn = np.sum((p == c) & (y == c)), np.sum((p == c) & (y != c)), np.sum((p != c) & (y == c))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(scores))


def cross_val_score(estimator, x, y, cv=5, scoring=None, fit_params=None):
    """Scores of ``estimator`` (cloned for every fold) on each validation fold."""
    cv, scorer, out = _resolve_cv(cv, estimator), get_scorer(scoring), []
    y = None if y is None else np.asarray(y)
    for tr, va in cv.split(x, y):
        est = clone(estimator).fit(_take(x, tr), _take(y, tr), **(fit_params or {}))
        out.append(scorer(est, _take(x, va), _take(y, va)))
    return np.array(out)


# ============================================================================ distributions
class Uniform:
    def __init__(self, low, high):
        self.low, self.high = low, high

    def sample(self, rng):
        return float(rng.uniform(self.low, self.high))


class LogUniform(Uniform):
    """Uniform in log-space: good for learning rates and regularisation strengths."""

    def sample(self, rng):
        return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))


class IntUniform(Uniform):
    def sample(self, rng):
        return int(rng.integers(self.low, self.high + 1))


class Choice:
    def __init__(self, options):
        self.options = list(options)

    def sample(self, rng):
        return self.options[rng.integers(len(self.options))]


def _sample(space, rng):
    out = {}
    for k, v in space.items():
        out[k] = v.sample(rng) if hasattr(v, "sample") else (v[rng.integers(len(v))] if isinstance(v, (list, tuple)) else v)
    return out


# ============================================================================ searches
class _BaseSearch:
    def __init__(self, estimator, cv=5, scoring=None, refit=True, verbose=False):
        self.estimator, self.cv, self.scoring, self.refit, self.verbose = estimator, cv, scoring, refit, verbose

    def _evaluate(self, params, x, y, fit_params):
        t0 = time.time()
        try:
            scores = cross_val_score(clone(self.estimator).set_params(**params), x, y, self.cv, self.scoring, fit_params)
            error = None
        except Exception as exc:                       # a bad combination should not stop the search
            scores, error = np.array([np.nan]), f"{type(exc).__name__}: {exc}"
        row = {"params": params, "mean_score": float(np.nanmean(scores)) if not np.all(np.isnan(scores)) else -np.inf,
               "std_score": float(np.nanstd(scores)) if not np.all(np.isnan(scores)) else np.nan,
               "fold_scores": scores.tolist(), "fit_time": time.time() - t0, "error": error}
        if self.verbose:
            print(f"{row['mean_score']:+.4f} ± {row['std_score']:.4f}  ({row['fit_time']:.1f}s)  {params}"
                  + (f"  [{error}]" if error else ""))
        return row

    def _finish(self, x, y, fit_params):
        self.cv_results_.sort(key=lambda r: r["mean_score"], reverse=True)
        for rank, r in enumerate(self.cv_results_, 1):
            r["rank"] = rank
        best = self.cv_results_[0]
        self.best_params_, self.best_score_ = best["params"], best["mean_score"]
        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(x, y, **(fit_params or {}))
        return self

    def results(self):
        """cv_results_ as a pandas DataFrame when pandas is installed, else the list of dicts."""
        try:
            import pandas as pd
        except ImportError:
            return self.cv_results_
        rows = [{**r["params"], "mean_score": r["mean_score"], "std_score": r["std_score"], "fit_time": r["fit_time"],
                 "rank": r["rank"]} for r in self.cv_results_]
        return pd.DataFrame(rows).set_index("rank")

    def predict(self, x):
        return self.best_estimator_.predict(x)

    def score(self, x, y):
        return get_scorer(self.scoring)(self.best_estimator_, x, y)


class GridSearchCV(_BaseSearch):
    """Exhaustive search over ``param_grid`` (a dict of lists, or a list of such dicts)."""

    def __init__(self, estimator, param_grid, cv=5, scoring=None, refit=True, verbose=False):
        super().__init__(estimator, cv, scoring, refit, verbose)
        self.param_grid = param_grid

    def candidates(self):
        grids = self.param_grid if isinstance(self.param_grid, list) else [self.param_grid]
        for grid in grids:
            keys = list(grid)
            for values in itertools.product(*(grid[k] for k in keys)):
                yield dict(zip(keys, values))

    def fit(self, x, y=None, **fit_params):
        self.cv_results_ = [self._evaluate(p, x, y, fit_params) for p in self.candidates()]
        return self._finish(x, y, fit_params)


class RandomSearchCV(_BaseSearch):
    """``n_iter`` random combinations from ``param_distributions`` (lists or Uniform/LogUniform/IntUniform/Choice)."""

    def __init__(self, estimator, param_distributions, n_iter=10, cv=5, scoring=None, refit=True, seed=0,
                 verbose=False):
        super().__init__(estimator, cv, scoring, refit, verbose)
        self.param_distributions, self.n_iter, self.seed = param_distributions, n_iter, seed

    def fit(self, x, y=None, **fit_params):
        rng = np.random.default_rng(self.seed)
        self.cv_results_ = [self._evaluate(_sample(self.param_distributions, rng), x, y, fit_params)
                            for _ in range(self.n_iter)]
        return self._finish(x, y, fit_params)


class SuccessiveHalvingSearch(_BaseSearch):
    """Successive halving (Jamieson & Talwalkar 2016): start many random candidates with a small budget
    (``resource``, by default the number of epochs), keep the best 1/eta, multiply the budget by eta, repeat.
    Much cheaper than a full search when every fit is slow, as with simulated quantum circuits."""

    def __init__(self, estimator, param_distributions, n_candidates=27, resource="epochs", min_resource=5,
                 max_resource=None, eta=3, cv=3, scoring=None, refit=True, seed=0, verbose=False):
        super().__init__(estimator, cv, scoring, refit, verbose)
        self.param_distributions, self.n_candidates, self.resource = param_distributions, n_candidates, resource
        self.min_resource, self.max_resource, self.eta, self.seed = min_resource, max_resource, eta, seed

    def fit(self, x, y=None, **fit_params):
        rng = np.random.default_rng(self.seed)
        max_r = self.max_resource or self.estimator.get_params()[self.resource]
        candidates = [_sample(self.param_distributions, rng) for _ in range(self.n_candidates)]
        r, self.rounds_ = self.min_resource, []
        while True:
            if self.verbose:
                print(f"--- round: {len(candidates)} candidates, {self.resource} = {r}")
            rows = [self._evaluate({**c, self.resource: r}, x, y, fit_params) for c in candidates]
            rows.sort(key=lambda row: row["mean_score"], reverse=True)
            self.rounds_.append(rows)
            if len(candidates) == 1 or r >= max_r:
                break
            keep = max(1, math.ceil(len(candidates) / self.eta))
            candidates = [{k: v for k, v in row["params"].items() if k != self.resource} for row in rows[:keep]]
            r = min(int(r * self.eta), max_r)
        self.cv_results_ = list(self.rounds_[-1])
        return self._finish(x, y, fit_params)


def hyperparameters(estimator_or_class) -> dict:
    """The tunable hyper-parameters of a model and their default values."""
    cls = estimator_or_class if isinstance(estimator_or_class, type) else type(estimator_or_class)
    return cls.default_params()


class ClusterSearch:
    """Hyper-parameter search for clustering models (no labels needed).

    ``scoring``: "silhouette" (on standardised inputs, comparable across kernels), "silhouette_kernel" (on the
    model's own Hilbert-space distances), "davies_bouldin" (negated, so larger is better), or with ``y``:
    "ari" (adjusted Rand index) and "nmi". ``param_grid`` is a grid (dict / list of dicts); with ``n_iter`` it is
    treated as a space of lists or distributions and sampled randomly.
    """

    def __init__(self, estimator, param_grid, scoring="silhouette", n_iter=None, seed=0, verbose=False):
        self.estimator, self.param_grid, self.scoring = estimator, param_grid, scoring
        self.n_iter, self.seed, self.verbose = n_iter, seed, verbose

    def _score(self, est, x, y):
        from .models.clustering import (adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score,
                                        silhouette_score)
        labels = est.labels_
        if len(np.unique(labels[labels >= 0])) < 2:
            return -np.inf
        xs = (x - x.mean(0)) / (x.std(0) + 1e-9)
        if self.scoring == "silhouette":
            return silhouette_score(xs, labels)
        if self.scoring == "silhouette_kernel":
            return silhouette_score(labels=labels, distances=est.d_)
        if self.scoring == "davies_bouldin":
            return -davies_bouldin_score(xs, labels)
        if self.scoring == "ari":
            return adjusted_rand_score(y, labels)
        if self.scoring == "nmi":
            return normalized_mutual_info_score(y, labels)
        if callable(self.scoring):
            return self.scoring(est, x, y)
        raise ValueError(self.scoring)

    def fit(self, x, y=None):
        x = np.asarray(x, dtype=float)
        if self.n_iter:
            rng = np.random.default_rng(self.seed)
            candidates = [_sample(self.param_grid, rng) for _ in range(self.n_iter)]
        else:
            candidates = list(GridSearchCV(self.estimator, self.param_grid).candidates())
        self.cv_results_ = []
        for params in candidates:
            t0 = time.time()
            est = clone(self.estimator).set_params(**params).fit(x)
            row = {"params": params, "mean_score": float(self._score(est, x, y)),
                   "n_clusters_found": int(len(np.unique(est.labels_[est.labels_ >= 0]))),
                   "fit_time": time.time() - t0}
            self.cv_results_.append(row)
            if self.verbose:
                print(f"{row['mean_score']:+.4f}  ({row['fit_time']:.1f}s)  {params}")
        self.cv_results_.sort(key=lambda r: r["mean_score"], reverse=True)
        for rank, r in enumerate(self.cv_results_, 1):
            r["rank"] = rank
        self.best_params_, self.best_score_ = self.cv_results_[0]["params"], self.cv_results_[0]["mean_score"]
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(x)
        self.labels_ = self.best_estimator_.labels_
        return self

    def results(self):
        try:
            import pandas as pd
        except ImportError:
            return self.cv_results_
        return pd.DataFrame([{**r["params"], "score": r["mean_score"], "clusters": r["n_clusters_found"],
                              "fit_time": r["fit_time"], "rank": r["rank"]} for r in self.cv_results_]).set_index("rank")
