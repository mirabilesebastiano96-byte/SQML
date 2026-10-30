"""Quantum convolutional models.

* QCNNClassifier (Cong, Choi & Lukin 2019): alternating convolution (2-qubit blocks on neighbours) and
  pooling (controlled rotations, then the control qubit is discarded) until few qubits remain.
* QuanvolutionalClassifier (Henderson et al. 2020): a fixed random circuit acts as a convolution filter on
  image patches; the resulting feature maps feed a classical classifier.
"""
from __future__ import annotations

import numpy as np

from ..autograd import Tensor
from ..nn import Linear, Module, Parameter, QuantumLayer, Sequential, Activation
from ..quantum.circuit import Circuit, Observable
from ..quantum.functions import expval, probs
from ..quantum.templates import angle_encoding
from .base import BaseEstimator, ClassifierMixin

__all__ = ["qcnn_circuit", "QCNNClassifier", "QuanvolutionFilter", "QuanvolutionalClassifier"]


def _conv(qc, a, b):
    w = qc.weights(6)
    qc.ry(a, w[0]).ry(b, w[1]).cnot(a, b).ry(a, w[2]).ry(b, w[3]).cnot(b, a).rz(a, w[4]).rz(b, w[5])


def _pool(qc, source, sink):
    w = qc.weights(2)
    qc.crz(source, sink, w[0]).x(source).crx(source, sink, w[1]).x(source)


def qcnn_circuit(n_qubits: int, n_keep: int = 1):
    """Build the QCNN; returns (circuit, wires that survive pooling)."""
    qc = Circuit(n_qubits)
    angle_encoding(qc, qc.inputs(n_qubits), "Y", hadamard=True)
    active = list(range(n_qubits))
    while len(active) > n_keep:
        for i in range(0, len(active) - 1, 2):
            _conv(qc, active[i], active[i + 1])
        for i in range(1, len(active) - 1, 2):
            _conv(qc, active[i], active[i + 1])
        n_pool = min(len(active) // 2, len(active) - n_keep)
        survivors, idx, pooled = [], 0, 0
        while idx < len(active):
            if pooled < n_pool and idx + 1 < len(active):
                _pool(qc, active[idx], active[idx + 1])        # information flows into active[idx+1]
                survivors.append(active[idx + 1])
                pooled += 1
                idx += 2
            else:
                survivors.append(active[idx])
                idx += 1
        active = survivors
    return qc, active


class _QCNNNet(Module):
    def __init__(self, n_qubits, n_classes, rng):
        n_keep = max(1, int(np.ceil(np.log2(n_classes))))
        self.circuit, self.readout = qcnn_circuit(n_qubits, n_keep)
        self.readout = self.readout[-n_keep:]
        self.weights = Parameter(rng.uniform(-np.pi, np.pi, self.circuit.n_weights))
        self.n_classes = n_classes

    def forward(self, x):
        p = probs(self.circuit, self.readout, x.arctan() * 2, self.weights)
        p = p[:, :self.n_classes]
        return (p / p.sum(-1, keepdims=True) + 1e-9).log()        # log-probabilities used as logits


class QCNNClassifier(ClassifierMixin, BaseEstimator):
    """One qubit per feature (e.g. 8 PCA components or an 8-pixel downsampled image)."""

    def _build(self, x, y):
        return _QCNNNet(x.shape[1], len(self.classes_), self.rng_)


class QuanvolutionFilter:
    """Fixed random circuit applied to every ``k x k`` patch (stride k). Output: (N, H/k, W/k, k*k)."""

    def __init__(self, kernel_size: int = 2, n_layers: int = 1, seed: int = 0):
        self.k = kernel_size
        n = kernel_size ** 2
        rng = np.random.default_rng(seed)
        qc = Circuit(n)
        angle_encoding(qc, qc.inputs(n), "Y")
        gates = ["RX", "RY", "RZ"]
        for _ in range(n_layers):
            for _ in range(2 * n):
                q = int(rng.integers(n))
                qc.add(gates[rng.integers(3)], q, float(rng.uniform(0, 2 * np.pi)))
            for _ in range(n):
                a, b = rng.choice(n, 2, replace=False)
                qc.cnot(int(a), int(b))
        self.circuit = qc
        self.observables = [Observable(f"Z{q}") for q in range(n)]

    def transform(self, images: np.ndarray) -> np.ndarray:
        """``images``: (N, H, W) with values in [0, 1]."""
        images = np.asarray(images, dtype=float)
        n, h, w = images.shape
        k = self.k
        patches = images[:, :h - h % k, :w - w % k].reshape(n, h // k, k, w // k, k).transpose(0, 1, 3, 2, 4)
        flat = patches.reshape(-1, k * k) * np.pi
        out = expval(self.circuit, self.observables, flat).data
        return out.reshape(n, h // k, w // k, k * k)


class QuanvolutionalClassifier(ClassifierMixin, BaseEstimator):
    """Quanvolution filter (fixed) + small classical head trained with cross-entropy."""

    def __init__(self, kernel_size=2, n_layers=1, hidden=32, **kwargs):
        kwargs.setdefault("scale", False)
        super().__init__(**kwargs)
        self.kernel_size, self.n_layers, self.hidden = kernel_size, n_layers, hidden

    def _prepare_x(self, x, fit=False):
        if fit:
            self.filter_ = QuanvolutionFilter(self.kernel_size, self.n_layers, self.seed)
        feats = self.filter_.transform(x)
        return feats.reshape(len(feats), -1)

    def _build(self, x, y):
        return Sequential(Linear(x.shape[1], self.hidden, rng=self.rng_), Activation("relu"),
                          Linear(self.hidden, len(self.classes_), rng=self.rng_))
