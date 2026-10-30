"""Variational quantum classifiers and regressors.

* VQC (Havlicek et al. 2019; Schuld et al. 2020): angle encoding + trainable ansatz + Pauli-Z readout.
* Data re-uploading (Perez-Salinas et al. 2020): the encoding is repeated before every layer.
* Hybrid QNN (Mari et al. 2020): classical layer -> quantum layer -> classical layer.
"""
from __future__ import annotations

import numpy as np

from ..autograd import concat
from ..nn import Linear, Module, Parameter, QuantumLayer
from ..quantum.circuit import Observable
from ..quantum.templates import variational_circuit
from .base import BaseEstimator, ClassifierMixin, RegressorMixin

__all__ = ["VQCClassifier", "VQCRegressor", "DataReuploadingClassifier", "DataReuploadingRegressor",
           "HybridQNNClassifier", "HybridQNNRegressor"]


class _VQCNet(Module):
    def __init__(self, n_qubits, n_outputs, layers, ansatz, reupload, rng, binary, encoding="angle",
                 input_transform="arctan"):
        circuit = variational_circuit(n_qubits, layers, ansatz, encoding, reupload=reupload)
        n_obs = 1 if binary else n_outputs
        if n_obs > n_qubits:
            raise ValueError(f"{n_outputs} outputs need at least {n_outputs} qubits")
        self.q = QuantumLayer(circuit, [Observable(f"Z{k}") for k in range(n_obs)], input_transform, rng=rng)
        self.scale = Parameter(np.full(n_obs, 3.0))
        self.bias = Parameter(np.zeros(n_obs))
        self.binary = binary

    def forward(self, x):
        z = self.q(x) * self.scale + self.bias
        return concat([z, -z], -1) * 0.5 if self.binary else z


def _check_features(x, n_qubits):
    if n_qubits is not None and x.shape[1] != n_qubits:
        raise ValueError(f"VQC uses one qubit per feature: got {x.shape[1]} features and n_qubits={n_qubits}. "
                         "Reduce the features (e.g. PCA) or use HybridQNN.")


class VQCClassifier(ClassifierMixin, BaseEstimator):
    """One qubit per feature, ``layers`` ansatz layers, class scores from <Z_k>.

    Hyper-parameters: ``layers``, ``ansatz`` ("strongly_entangling", "hardware_efficient", "real_amplitudes"),
    ``encoding`` ("angle", "angle_h"), ``input_transform`` ("arctan", "pi", "tanh_pi", None), ``reupload``,
    plus the training ones of :class:`BaseEstimator`.
    """

    def __init__(self, layers=2, ansatz="strongly_entangling", encoding="angle", input_transform="arctan",
                 reupload=False, **kwargs):
        super().__init__(**kwargs)
        self.layers, self.ansatz, self.reupload = layers, ansatz, reupload
        self.encoding, self.input_transform = encoding, input_transform

    def _build(self, x, y):
        k = len(self.classes_)
        return _VQCNet(x.shape[1], k, self.layers, self.ansatz, self.reupload, self.rng_, k == 2, self.encoding,
                       self.input_transform)


class VQCRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, layers=2, ansatz="strongly_entangling", encoding="angle", input_transform="arctan",
                 reupload=False, **kwargs):
        super().__init__(**kwargs)
        self.layers, self.ansatz, self.reupload = layers, ansatz, reupload
        self.encoding, self.input_transform = encoding, input_transform

    def _build(self, x, y):
        return _VQCNet(x.shape[1], 1, self.layers, self.ansatz, self.reupload, self.rng_, False, self.encoding,
                       self.input_transform)


class DataReuploadingClassifier(VQCClassifier):
    def __init__(self, layers=3, **kwargs):
        kwargs.pop("reupload", None)
        super().__init__(layers=layers, reupload=True, **kwargs)


class DataReuploadingRegressor(VQCRegressor):
    def __init__(self, layers=3, **kwargs):
        kwargs.pop("reupload", None)
        super().__init__(layers=layers, reupload=True, **kwargs)


class _HybridNet(Module):
    def __init__(self, n_in, n_out, n_qubits, layers, rng, ansatz="strongly_entangling", input_transform="tanh_pi"):
        self.pre = Linear(n_in, n_qubits, rng=rng)
        circuit = variational_circuit(n_qubits, layers, ansatz)
        self.q = QuantumLayer(circuit, None, input_transform, rng=rng)
        self.post = Linear(n_qubits, n_out, rng=rng)

    def forward(self, x):
        return self.post(self.q(self.pre(x)))


class HybridQNNClassifier(ClassifierMixin, BaseEstimator):
    """Linear(n_features -> n_qubits) -> quantum layer -> Linear(n_qubits -> n_classes)."""

    def __init__(self, n_qubits=4, layers=2, ansatz="strongly_entangling", input_transform="tanh_pi", **kwargs):
        super().__init__(**kwargs)
        self.n_qubits, self.layers, self.ansatz, self.input_transform = n_qubits, layers, ansatz, input_transform

    def _build(self, x, y):
        return _HybridNet(x.shape[1], len(self.classes_), self.n_qubits, self.layers, self.rng_, self.ansatz,
                          self.input_transform)


class HybridQNNRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_qubits=4, layers=2, ansatz="strongly_entangling", input_transform="tanh_pi", **kwargs):
        super().__init__(**kwargs)
        self.n_qubits, self.layers, self.ansatz, self.input_transform = n_qubits, layers, ansatz, input_transform

    def _build(self, x, y):
        return _HybridNet(x.shape[1], 1, self.n_qubits, self.layers, self.rng_, self.ansatz, self.input_transform)


# =============================================================================== QKAN networks
from .recurrent import QKANLayer                                           # noqa: E402

__all__ += ["QKANClassifier", "QKANRegressor"]


class _QKANNet(Module):
    def __init__(self, sizes, rng, reps):
        self.layers = [QKANLayer(a, b, reps, rng) for a, b in zip(sizes[:-1], sizes[1:])]

    def forward(self, x):
        x = x.tanh()
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = x.tanh()
        return x


class QKANClassifier(ClassifierMixin, BaseEstimator):
    """Quantum Kolmogorov-Arnold network (Jiang et al. 2025): every edge is a single-qubit DARUAN circuit."""

    def __init__(self, hidden=(4,), reps=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden, self.reps = tuple(hidden), reps

    def _build(self, x, y):
        return _QKANNet((x.shape[1],) + self.hidden + (len(self.classes_),), self.rng_, self.reps)


class QKANRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, hidden=(4,), reps=2, **kwargs):
        super().__init__(**kwargs)
        self.hidden, self.reps = tuple(hidden), reps

    def _build(self, x, y):
        return _QKANNet((x.shape[1],) + self.hidden + (1,), self.rng_, self.reps)
