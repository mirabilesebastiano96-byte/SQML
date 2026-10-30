"""Quantum kernel methods.

* QuantumKernel: fidelity kernel k(x, x') = |<phi(x')|phi(x)>|^2 (Havlicek et al. 2019; Schuld & Killoran 2019).
* QSVC: support vector classifier on a quantum kernel, solved with SMO (Platt 1998), one-vs-rest for >2 classes.
* QKernelRidge: kernel ridge regression on a quantum kernel.
"""
from __future__ import annotations

import numpy as np

from ..quantum.circuit import Circuit
from ..quantum.functions import statevector
from ..quantum.templates import angle_encoding, zz_feature_map, zz_features
from .base import ParamsMixin, Scaler

__all__ = ["QuantumKernel", "QSVC", "QKernelRidge"]


class QuantumKernel(ParamsMixin):
    """``feature_map`` in {"zz", "angle", "iqp_angle"}; one qubit per feature."""

    def __init__(self, feature_map: str = "zz", reps: int = 2, bandwidth: float = 1.0):
        self.feature_map, self.reps, self.bandwidth = feature_map, reps, bandwidth

    def _states(self, x):
        x = np.asarray(x, dtype=float) * self.bandwidth
        n = x.shape[1]
        if self.feature_map == "zz":
            return statevector(zz_feature_map(n, self.reps), zz_features(x))
        qc = Circuit(n)
        refs = qc.inputs(n)
        for _ in range(self.reps):
            angle_encoding(qc, refs, "Y", hadamard=self.feature_map == "iqp_angle")
            for q in range(n - 1):
                qc.cz(q, q + 1)
        return statevector(qc, x)

    def __call__(self, x1, x2=None) -> np.ndarray:
        s1 = self._states(x1)
        s2 = s1 if x2 is None else self._states(x2)
        return np.abs(s1 @ s2.conj().T) ** 2


def _smo(k, y, c=1.0, tol=1e-3, max_passes=20, max_iter=10000, seed=0):
    """Simplified SMO for the dual SVM with a precomputed kernel; y in {-1, +1}."""
    rng = np.random.default_rng(seed)
    n = len(y)
    alpha, b = np.zeros(n), 0.0
    passes = it = 0
    while passes < max_passes and it < max_iter:
        changed = 0
        for i in range(n):
            e_i = (alpha * y) @ k[:, i] + b - y[i]
            if (y[i] * e_i < -tol and alpha[i] < c) or (y[i] * e_i > tol and alpha[i] > 0):
                j = rng.integers(n - 1)
                j += j >= i
                e_j = (alpha * y) @ k[:, j] + b - y[j]
                a_i, a_j = alpha[i], alpha[j]
                if y[i] != y[j]:
                    lo, hi = max(0, a_j - a_i), min(c, c + a_j - a_i)
                else:
                    lo, hi = max(0, a_i + a_j - c), min(c, a_i + a_j)
                eta = 2 * k[i, j] - k[i, i] - k[j, j]
                if lo == hi or eta >= 0:
                    continue
                alpha[j] = np.clip(a_j - y[j] * (e_i - e_j) / eta, lo, hi)
                if abs(alpha[j] - a_j) < 1e-6:
                    continue
                alpha[i] = a_i + y[i] * y[j] * (a_j - alpha[j])
                b1 = b - e_i - y[i] * (alpha[i] - a_i) * k[i, i] - y[j] * (alpha[j] - a_j) * k[i, j]
                b2 = b - e_j - y[i] * (alpha[i] - a_i) * k[i, j] - y[j] * (alpha[j] - a_j) * k[j, j]
                b = b1 if 0 < alpha[i] < c else (b2 if 0 < alpha[j] < c else (b1 + b2) / 2)
                changed += 1
        passes = passes + 1 if changed == 0 else 0
        it += 1
    return alpha, b


class QSVC(ParamsMixin):
    def __init__(self, kernel: QuantumKernel | None = None, c: float = 1.0, seed: int = 0, scale: bool = True):
        self.kernel = kernel or QuantumKernel()
        self.c, self.seed, self.scale = c, seed, scale

    def _x(self, x, fit=False):
        if not self.scale:
            return np.asarray(x, dtype=float)
        if fit:
            self.scaler_ = Scaler().fit(x)
        return self.scaler_.transform(x)

    def fit(self, x, y):
        self.x_ = self._x(x, fit=True)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        k = self.kernel(self.x_)
        targets = [1] if len(self.classes_) == 2 else range(len(self.classes_))
        self.models_ = []
        for cls in targets:
            yy = np.where(y_idx == cls, 1.0, -1.0)
            alpha, b = _smo(k, yy, self.c, seed=self.seed)
            self.models_.append((alpha * yy, b))
        return self

    def decision_function(self, x):
        k = self.kernel(self._x(x), self.x_)
        scores = np.stack([k @ ay + b for ay, b in self.models_], 1)
        return scores[:, 0] if len(self.classes_) == 2 else scores

    def predict(self, x):
        s = self.decision_function(x)
        idx = (s > 0).astype(int) if s.ndim == 1 else np.argmax(s, 1)
        return self.classes_[idx]

    def score(self, x, y):
        return float(np.mean(self.predict(x) == np.asarray(y)))


class QKernelRidge(ParamsMixin):
    def __init__(self, kernel: QuantumKernel | None = None, alpha: float = 1e-2, scale: bool = True):
        self.kernel, self.alpha, self.scale = kernel or QuantumKernel(), alpha, scale

    def fit(self, x, y):
        self.scaler_ = Scaler().fit(x) if self.scale else None
        self.x_ = self.scaler_.transform(x) if self.scale else np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        self.y_mean_ = y.mean()
        k = self.kernel(self.x_)
        self.coef_ = np.linalg.solve(k + self.alpha * np.eye(len(k)), y - self.y_mean_)
        return self

    def predict(self, x):
        xs = self.scaler_.transform(x) if self.scale else np.asarray(x, dtype=float)
        return self.kernel(xs, self.x_) @ self.coef_ + self.y_mean_

    def score(self, x, y):
        y = np.asarray(y, dtype=float)
        return float(1 - np.sum((self.predict(x) - y) ** 2) / np.sum((y - y.mean()) ** 2))


# =============================================================================== more kernel methods
from ..autograd import Tensor, no_grad                                    # noqa: E402
from ..nn import Adam, Parameter                                          # noqa: E402
from ..quantum.circuit import Observable                                  # noqa: E402
from ..quantum.functions import expval, probs                             # noqa: E402

__all__ += ["ProjectedQuantumKernel", "TrainableQuantumKernel", "QGaussianProcessRegressor", "QKernelPCA",
            "QKMeans", "QuantumExtremeLearningMachine"]


class ProjectedQuantumKernel(ParamsMixin):
    """Huang et al. (Nat. Commun. 2021): k(x,x') = exp(-gamma * sum_q ||rho_q(x) - rho_q(x')||^2),
    with rho_q the one-qubit reduced states, i.e. the Bloch vectors (<X_q>, <Y_q>, <Z_q>)."""

    def __init__(self, reps: int = 2, gamma: float = 1.0, bandwidth: float = 1.0):
        self.reps, self.gamma, self.bandwidth = reps, gamma, bandwidth

    def features(self, x):
        x = np.asarray(x, dtype=float) * self.bandwidth
        n = x.shape[1]
        qc = Circuit(n)
        refs = qc.inputs(n)
        for _ in range(self.reps):
            angle_encoding(qc, refs, "Z", hadamard=True)
            for q in range(n - 1):
                qc.rzz(q, q + 1, refs[q] * 1.0)
        obs = [Observable(f"{p}{q}") for q in range(n) for p in "XYZ"]
        return expval(qc, obs, x).data

    def __call__(self, x1, x2=None):
        f1 = self.features(x1)
        f2 = f1 if x2 is None else self.features(x2)
        d2 = (f1 ** 2).sum(1)[:, None] + (f2 ** 2).sum(1)[None, :] - 2 * f1 @ f2.T
        return np.exp(-self.gamma * np.maximum(d2, 0))


class TrainableQuantumKernel(ParamsMixin):
    """Quantum kernel alignment (Hubregtsen et al. 2022; Glick et al. 2024).

    Feature map U(x, w): layers of RY(x_q) followed by trainable RZ/RY rotations and a CZ chain. The fidelity
    k(x,x') = |<0|U(x',w)^dagger U(x,w)|0>|^2 is the probability of |0...0> after U(x,w) U(x',w)^dagger, so it
    is differentiable in w. ``fit`` maximises the kernel-target alignment. Use it inside ``QSVC``.
    """

    requires_fit = True

    def __init__(self, layers: int = 2, epochs: int = 40, lr: float = 0.05, batch: int = 16, seed: int = 0):
        self.layers, self.epochs, self.lr, self.batch, self.seed = layers, epochs, lr, batch, seed
        self.fitted_ = False

    def _feature_map(self, n):
        qc = Circuit(n)
        x = qc.inputs(n)
        for _ in range(self.layers):
            for q in range(n):
                qc.ry(q, x[q])
            for q in range(n):
                w = qc.weights(2)
                qc.rz(q, w[0]).ry(q, w[1])
            for q in range(n - 1):
                qc.cz(q, q + 1)
        return qc

    def _kernel_tensor(self, xa, xb):
        pairs = np.concatenate([np.repeat(xa, len(xb), 0), np.tile(xb, (len(xa), 1))], 1)
        p0 = probs(self.fidelity_circuit_, None, pairs, self.weights_)[:, 0]
        return p0.reshape(len(xa), len(xb))

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        n = x.shape[1]
        rng = np.random.default_rng(self.seed)
        self.feature_map_ = self._feature_map(n)
        self.fidelity_circuit_ = Circuit(n).compose(self.feature_map_).compose(self.feature_map_.inverse(), n, 0)
        self.weights_ = Parameter(rng.uniform(-np.pi, np.pi, self.feature_map_.n_weights))
        labels = np.where(np.asarray(y) == np.unique(y)[0], -1.0, 1.0)
        opt = Adam([self.weights_], self.lr)
        self.history_ = []
        for _ in range(self.epochs):
            idx = rng.choice(len(x), min(self.batch, len(x)), replace=False)
            self.weights_.grad = None
            k = self._kernel_tensor(x[idx], x[idx])
            yy = np.outer(labels[idx], labels[idx])
            alignment = (k * yy).sum() / (((k * k).sum()).sqrt() * np.sqrt((yy ** 2).sum()))
            (-alignment).backward()
            opt.step()
            self.history_.append(alignment.item())
        self.fitted_ = True
        return self

    def __call__(self, x1, x2=None):
        s1 = statevector(self.feature_map_, x1, self.weights_.data)
        s2 = s1 if x2 is None else statevector(self.feature_map_, x2, self.weights_.data)
        return np.abs(s1 @ s2.conj().T) ** 2


def _qsvc_fit_with_kernel_training(self, x, y):
    xs = self._x(x, fit=True)
    if getattr(self.kernel, "requires_fit", False) and not getattr(self.kernel, "fitted_", False):
        self.kernel.fit(xs, y)
    self.x_ = xs
    self.classes_, y_idx = np.unique(y, return_inverse=True)
    k = self.kernel(self.x_)
    targets = [1] if len(self.classes_) == 2 else range(len(self.classes_))
    self.models_ = []
    for cls in targets:
        yy = np.where(y_idx == cls, 1.0, -1.0)
        alpha, b = _smo(k, yy, self.c, seed=self.seed)
        self.models_.append((alpha * yy, b))
    return self


QSVC.fit = _qsvc_fit_with_kernel_training


class QGaussianProcessRegressor(ParamsMixin):
    """Gaussian-process regression with a quantum kernel: predictive mean and standard deviation."""

    def __init__(self, kernel=None, noise: float = 1e-2):
        self.kernel, self.noise = kernel or QuantumKernel("angle", reps=1), noise

    def fit(self, x, y):
        self.scaler_ = Scaler().fit(x)
        self.x_ = self.scaler_.transform(x)
        y = np.asarray(y, dtype=float)
        self.y_mean_, self.y_std_ = y.mean(), y.std() + 1e-12
        k = self.kernel(self.x_)
        self.l_ = np.linalg.cholesky(k + self.noise * np.eye(len(k)))
        self.alpha_ = np.linalg.solve(self.l_.T, np.linalg.solve(self.l_, (y - self.y_mean_) / self.y_std_))
        return self

    def predict(self, x, return_std: bool = False):
        xs = self.scaler_.transform(x)
        ks = self.kernel(xs, self.x_)
        mean = ks @ self.alpha_ * self.y_std_ + self.y_mean_
        if not return_std:
            return mean
        v = np.linalg.solve(self.l_, ks.T)
        kss = np.diag(self.kernel(xs)) if not isinstance(self.kernel, QuantumKernel) else np.ones(len(xs))
        return mean, np.sqrt(np.maximum(kss - (v ** 2).sum(0), 1e-12)) * self.y_std_

    def score(self, x, y):
        y = np.asarray(y, dtype=float)
        return float(1 - np.sum((self.predict(x) - y) ** 2) / np.sum((y - y.mean()) ** 2))


class QKernelPCA(ParamsMixin):
    """Kernel PCA in the quantum feature space."""

    def __init__(self, n_components: int = 2, kernel=None):
        self.n_components, self.kernel = n_components, kernel or QuantumKernel()

    def fit(self, x):
        self.scaler_ = Scaler().fit(x)
        self.x_ = self.scaler_.transform(x)
        k = self.kernel(self.x_)
        n = len(k)
        self.k_col_ = k.mean(0)
        self.k_all_ = k.mean()
        kc = k - self.k_col_[None, :] - self.k_col_[:, None] + self.k_all_
        vals, vecs = np.linalg.eigh(kc)
        order = np.argsort(vals)[::-1][:self.n_components]
        self.eigenvalues_ = vals[order]
        self.alphas_ = vecs[:, order] / np.sqrt(np.maximum(vals[order], 1e-12))
        self.explained_variance_ratio_ = vals[order] / vals[vals > 0].sum()
        return self

    def transform(self, x):
        k = self.kernel(self.scaler_.transform(x), self.x_)
        kc = k - k.mean(1, keepdims=True) - self.k_col_[None, :] + self.k_all_
        return kc @ self.alphas_

    def fit_transform(self, x):
        return self.fit(x).transform(x)


class QKMeans(ParamsMixin):
    """Kernel k-means with a quantum kernel (distances computed in the Hilbert space of the feature map)."""

    def __init__(self, n_clusters: int = 2, kernel=None, max_iter: int = 100, n_init: int = 5, seed: int = 0):
        self.n_clusters, self.kernel, self.max_iter, self.n_init, self.seed = n_clusters, kernel or QuantumKernel(), \
            max_iter, n_init, seed

    def _assign(self, k_cross, k_self_diag, k_train, labels):
        d = np.zeros((len(k_cross), self.n_clusters))
        for c in range(self.n_clusters):
            m = labels == c
            if not m.any():
                d[:, c] = np.inf
                continue
            d[:, c] = k_self_diag - 2 * k_cross[:, m].mean(1) + k_train[np.ix_(m, m)].mean()
        return d

    def _prep(self, x, fit=False):
        x = np.asarray(x, dtype=float)
        if getattr(self.kernel, "handles_scaling", False):
            return x
        if fit:
            self.scaler_ = Scaler().fit(x)
        return self.scaler_.transform(x)

    def fit(self, x):
        self.x_ = self._prep(x, fit=True)
        k = self.kernel(self.x_)
        rng = np.random.default_rng(self.seed)
        best = (np.inf, None)
        for _ in range(self.n_init):
            labels = rng.integers(self.n_clusters, size=len(k))
            for _ in range(self.max_iter):
                d = self._assign(k, np.diag(k), k, labels)
                new = np.argmin(d, 1)
                if np.array_equal(new, labels):
                    break
                labels = new
            inertia = d[np.arange(len(k)), labels].sum()
            if inertia < best[0]:
                best = (inertia, labels)
        self.inertia_, self.labels_ = best
        self.k_train_ = k
        return self

    def predict(self, x):
        xs = self._prep(x)
        kc = self.kernel(xs, self.x_)
        diag = np.ones(len(xs)) if isinstance(self.kernel, QuantumKernel) else np.diag(self.kernel(xs))
        return np.argmin(self._assign(kc, diag, self.k_train_, self.labels_), 1)


class QuantumExtremeLearningMachine(ParamsMixin):
    """Quantum extreme learning machine: a fixed random circuit maps x to <Z_q>, <Z_q Z_q+1>, <X_q> features;
    only a ridge read-out is trained (classification via one-hot ridge, regression otherwise)."""

    def __init__(self, depth: int = 3, alpha: float = 1e-2, task: str = "classification", seed: int = 0):
        self.depth, self.alpha, self.task, self.seed = depth, alpha, task, seed

    def _features(self, xs):
        return np.concatenate([expval(self.circuit_, self.obs_, np.arctan(xs) * 2).data, np.ones((len(xs), 1))], 1)

    def fit(self, x, y):
        self.scaler_ = Scaler().fit(x)
        xs = self.scaler_.transform(x)
        n = xs.shape[1]
        rng = np.random.default_rng(self.seed)
        qc = Circuit(n)
        refs = qc.inputs(n)
        for _ in range(self.depth):
            angle_encoding(qc, refs, "Y")
            for q in range(n):
                qc.rx(q, float(rng.uniform(0, 2 * np.pi))).rz(q, float(rng.uniform(0, 2 * np.pi)))
            for q in range(n):
                qc.cnot(q, (q + 1) % n) if n > 1 else None
        self.circuit_ = qc
        self.obs_ = [Observable(f"Z{q}") for q in range(n)] + [Observable(f"X{q}") for q in range(n)] + \
                    [Observable(f"Z{q} Z{(q + 1) % n}") for q in range(n) if n > 1]
        f = self._features(xs)
        if self.task == "classification":
            self.classes_, idx = np.unique(y, return_inverse=True)
            target = np.eye(len(self.classes_))[idx]
        else:
            target = np.asarray(y, dtype=float)
        self.coef_ = np.linalg.solve(f.T @ f + self.alpha * np.eye(f.shape[1]), f.T @ target)
        return self

    def predict(self, x):
        out = self._features(self.scaler_.transform(x)) @ self.coef_
        return self.classes_[np.argmax(out, 1)] if self.task == "classification" else out

    def score(self, x, y):
        y = np.asarray(y)
        if self.task == "classification":
            return float(np.mean(self.predict(x) == y))
        return float(1 - np.sum((self.predict(x) - y) ** 2) / np.sum((y - y.mean()) ** 2))
