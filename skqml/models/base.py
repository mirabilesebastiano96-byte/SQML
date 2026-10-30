"""Shared machinery for scikit-learn-style estimators."""
from __future__ import annotations

import copy
import inspect

import numpy as np

from ..autograd import Tensor, no_grad
from ..nn import cross_entropy, mse_loss, train


class Scaler:
    """Standardise features with statistics learned at fit time."""

    def fit(self, x):
        x = np.asarray(x, dtype=float)
        axes = tuple(range(x.ndim - 1))
        self.mean_, self.std_ = x.mean(axes), x.std(axes) + 1e-9
        return self

    def transform(self, x):
        return (np.asarray(x, dtype=float) - self.mean_) / self.std_


class ParamsMixin:
    """Hyper-parameter access in the scikit-learn style.

    The hyper-parameters of an estimator are the arguments of its constructor (following ``**kwargs`` up the
    class hierarchy). ``set_params`` accepts nested names such as ``kernel__reps``.
    """

    @classmethod
    def _param_names(cls):
        names = []
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            sig = inspect.signature(init)
            for name, p in sig.parameters.items():
                if name != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL) and name not in names:
                    names.append(name)
            if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                break
        return names

    @classmethod
    def default_params(cls) -> dict:
        """Constructor defaults, e.g. ``VQCClassifier.default_params()``."""
        out = {}
        for klass in cls.__mro__:
            init = klass.__dict__.get("__init__")
            if init is None:
                continue
            sig = inspect.signature(init)
            for name, p in sig.parameters.items():
                if name in cls._param_names() and name not in out and p.default is not p.empty:
                    out[name] = p.default
            if not any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
                break
        return out

    def get_params(self, deep: bool = False) -> dict:
        params = {name: getattr(self, name) for name in self._param_names() if hasattr(self, name)}
        if deep:
            for name, value in list(params.items()):
                if hasattr(value, "get_params"):
                    params.update({f"{name}__{k}": v for k, v in value.get_params().items()})
        return params

    def set_params(self, **params):
        for key, value in params.items():
            if "__" in key:
                outer, inner = key.split("__", 1)
                getattr(self, outer).set_params(**{inner: value})
            elif key in self._param_names():
                setattr(self, key, value)
            else:
                raise ValueError(f"{type(self).__name__} has no hyper-parameter '{key}'. "
                                 f"Available: {self._param_names()}")
        return self

    def __repr__(self):
        defaults = self.default_params()
        changed = {k: v for k, v in self.get_params().items() if k not in defaults or _differs(v, defaults[k])}
        return f"{type(self).__name__}({', '.join(f'{k}={v!r}' for k, v in changed.items())})"


def _differs(a, b):
    try:
        return bool(a != b)
    except Exception:
        return True


def clone(estimator, **overrides):
    """A fresh, unfitted copy with the same hyper-parameters (nested estimators are cloned too)."""
    params = {}
    for k, v in estimator.get_params().items():
        params[k] = clone(v) if hasattr(v, "get_params") else copy.deepcopy(v)
    params.update(overrides)
    return type(estimator)(**params)


class BaseEstimator(ParamsMixin):
    """fit / predict / score on top of a skqml ``Module`` stored in ``self.net_``.

    Common hyper-parameters: ``epochs``, ``lr``, ``batch_size``, ``optimizer`` ("adam" or "sgd"),
    ``weight_decay``, ``patience`` (early stopping, needs ``validation_split`` > 0), ``scale``, ``seed``.
    """

    def __init__(self, epochs=50, lr=0.02, batch_size=32, optimizer="adam", weight_decay=0.0, seed=0, scale=True,
                 patience=None, validation_split=0.0, verbose=False):
        self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed
        self.optimizer, self.weight_decay = optimizer, weight_decay
        self.scale, self.patience, self.validation_split, self.verbose = scale, patience, validation_split, verbose

    # hooks implemented by subclasses ---------------------------------------
    def _build(self, x, y):
        raise NotImplementedError

    def _loss(self, net, x, y):
        raise NotImplementedError

    # common logic ----------------------------------------------------------
    def _prepare_x(self, x, fit=False):
        x = np.asarray(x, dtype=float)
        if not self.scale:
            return x
        if fit:
            self.scaler_ = Scaler().fit(x)
        return self.scaler_.transform(x)

    def _fit_arrays(self, x, y):
        self.rng_ = np.random.default_rng(self.seed)
        self.net_ = self._build(x, y)
        val = None
        if self.validation_split > 0:
            cut = int(len(x) * (1 - self.validation_split))
            x, y, val = x[:cut], y[:cut], (x[cut:], y[cut:])
        self.history_ = train(self.net_, self._loss, (x, y), self.epochs, self.batch_size, self.lr,
                              val_data=val, patience=self.patience, optimizer=self.optimizer, seed=self.seed,
                              verbose=self.verbose, weight_decay=self.weight_decay)
        return self

    def _forward(self, x):
        self.net_.set_mode(False)
        with no_grad():
            return self.net_(Tensor(x)).data

    def n_parameters(self) -> int:
        return self.net_.n_parameters()


class ClassifierMixin:
    def fit(self, x, y):
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        return self._fit_arrays(self._prepare_x(x, fit=True), y_idx)

    def _loss(self, net, x, y):
        return cross_entropy(net(Tensor(x)), y)

    def predict_proba(self, x):
        logits = self._forward(self._prepare_x(x))
        e = np.exp(logits - logits.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    def predict(self, x):
        return self.classes_[np.argmax(self.predict_proba(x), -1)]

    def score(self, x, y):
        return float(np.mean(self.predict(x) == np.asarray(y)))


class RegressorMixin:
    def fit(self, x, y):
        y = np.asarray(y, dtype=float)
        self.y_mean_, self.y_std_ = y.mean(), y.std() + 1e-9
        return self._fit_arrays(self._prepare_x(x, fit=True), (y - self.y_mean_) / self.y_std_)

    def _loss(self, net, x, y):
        return mse_loss(net(Tensor(x))[..., 0], y)

    def predict(self, x):
        return self._forward(self._prepare_x(x))[..., 0] * self.y_std_ + self.y_mean_

    def score(self, x, y):
        """Coefficient of determination R^2."""
        y = np.asarray(y, dtype=float)
        return float(1 - np.sum((self.predict(x) - y) ** 2) / np.sum((y - y.mean()) ** 2))
