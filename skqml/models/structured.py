"""Structured quantum networks.

* TTNClassifier / MERAClassifier (Grant et al., npj Quantum Inf. 2018): hierarchical tensor-network classifiers.
* OrthogonalLayer / OrthogonalQNNClassifier (Landman et al., Quantum 2022): pyramid circuits of RBS gates that
  implement exactly an orthogonal matrix on unary-encoded vectors.
* QuantumTransformerClassifier / Regressor (Cherrat et al., Quantum 2024): transformer whose value and attention
  maps are orthogonal quantum layers.
"""
from __future__ import annotations

import numpy as np

from ..autograd import Tensor, concat
from ..nn import Activation, Linear, Module, Parameter
from ..quantum.circuit import Circuit
from ..quantum.functions import amplitudes, probs
from ..quantum.templates import angle_encoding
from .base import BaseEstimator, ClassifierMixin, RegressorMixin

__all__ = ["tree_circuit", "TTNClassifier", "MERAClassifier", "OrthogonalLayer", "OrthogonalQNNClassifier",
           "QuantumTransformerClassifier", "QuantumTransformerRegressor"]


# =============================================================================== TTN / MERA
def _block(qc, a, b):
    w = qc.weights(4)
    qc.ry(a, w[0]).ry(b, w[1]).cnot(a, b).ry(a, w[2]).ry(b, w[3])


def tree_circuit(n_qubits: int, n_keep: int = 1, mera: bool = False):
    """Binary tree of 2-qubit blocks; information flows into the second qubit of every pair.
    With ``mera=True`` disentanglers act across neighbouring pairs before each coarse-graining step."""
    qc = Circuit(n_qubits)
    angle_encoding(qc, qc.inputs(n_qubits), "Y")
    active = list(range(n_qubits))
    while len(active) > n_keep:
        if mera:
            for i in range(1, len(active) - 1, 2):
                _block(qc, active[i], active[i + 1])
        n_merge = min(len(active) // 2, len(active) - n_keep)
        new, i, merged = [], 0, 0
        while i < len(active):
            if merged < n_merge and i + 1 < len(active):
                _block(qc, active[i], active[i + 1])
                new.append(active[i + 1])
                merged, i = merged + 1, i + 2
            else:
                new.append(active[i])
                i += 1
        active = new
    return qc, active


class _TreeNet(Module):
    def __init__(self, n_qubits, n_classes, mera, rng):
        n_keep = max(1, int(np.ceil(np.log2(n_classes))))
        self.circuit, self.readout = tree_circuit(n_qubits, n_keep, mera)
        self.weights = Parameter(rng.uniform(-np.pi, np.pi, self.circuit.n_weights))
        self.n_classes = n_classes

    def forward(self, x):
        p = probs(self.circuit, self.readout, x.arctan() * 2, self.weights)[:, :self.n_classes]
        return (p / p.sum(-1, keepdims=True) + 1e-9).log()


class TTNClassifier(ClassifierMixin, BaseEstimator):
    """Tree tensor network classifier: one qubit per feature."""

    def _build(self, x, y):
        return _TreeNet(x.shape[1], len(self.classes_), False, self.rng_)


class MERAClassifier(ClassifierMixin, BaseEstimator):
    """MERA classifier: TTN plus disentanglers."""

    def _build(self, x, y):
        return _TreeNet(x.shape[1], len(self.classes_), True, self.rng_)


# =============================================================================== orthogonal layers
def _brick_rbs(n):
    qc = Circuit(n)
    for depth in range(n):
        for i in range(depth % 2, n - 1, 2):
            qc.rbs(i, i + 1, qc.weights(1)[0])
    return qc


class OrthogonalLayer(Module):
    """y = W x with W orthogonal, realised by a brick-wall of RBS gates acting on the unary subspace.

    In the unary encoding a vector x in R^n is the state sum_i x_i |e_i> on n qubits; RBS gates keep the state in
    that subspace and act as Givens rotations. Column i of W is the circuit applied to |e_i>.
    """

    def __init__(self, n_in, n_out, rng=None, bias=False):
        rng = rng or np.random.default_rng()
        self.n_in, self.n_out, self.n = n_in, n_out, max(n_in, n_out, 2)
        self.bias = Parameter(np.zeros(n_out)) if bias else None
        self.circuit = _brick_rbs(self.n)
        self.weights = Parameter(rng.uniform(-np.pi, np.pi, self.circuit.n_weights))
        self.unary = [1 << (self.n - 1 - i) for i in range(self.n)]
        basis = np.zeros((self.n, 2 ** self.n), dtype=complex)
        basis[np.arange(self.n), self.unary] = 1.0
        self.basis_states = basis

    def matrix(self):
        """W as a Tensor: columns = circuit applied to each unary basis state."""
        cols = amplitudes(self.circuit, self.unary, None, self.weights, self.basis_states)   # (n_in_basis, n_out)
        return cols.T                                                                         # W[out, in]

    def forward(self, x):
        if self.n_in < self.n:
            x = concat([x, Tensor(np.zeros(x.shape[:-1] + (self.n - self.n_in,)))], -1)
        y = (x @ self.matrix().T)[..., :self.n_out]
        return y + self.bias if self.bias is not None else y


class _ONN(Module):
    def __init__(self, n_in, n_out, hidden, rng):
        self.l1 = OrthogonalLayer(n_in, hidden, rng, bias=True)
        self.l2 = OrthogonalLayer(hidden, n_out, rng, bias=True)
        self.scale = Parameter(np.array(3.0))

    def forward(self, x):
        h = self.l1(x).tanh()
        return self.l2(h) * self.scale


class OrthogonalQNNClassifier(ClassifierMixin, BaseEstimator):
    """Two orthogonal quantum layers with a tanh in between."""

    def __init__(self, hidden=6, **kwargs):
        super().__init__(**kwargs)
        self.hidden = hidden

    def _build(self, x, y):
        return _ONN(x.shape[1], len(self.classes_), max(self.hidden, len(self.classes_)), self.rng_)


# =============================================================================== quantum transformer
class _QTransformer(Module):
    def __init__(self, n_features, n_out, dim, rng):
        self.embed = Linear(n_features, dim, rng=rng)
        self.value = OrthogonalLayer(dim, dim, rng)
        self.attn = OrthogonalLayer(dim, dim, rng)
        self.head = Linear(dim, n_out, rng=rng)
        self.dim = dim

    def forward(self, x):
        h = self.embed(x)                                                   # (B, T, d)
        hn = h / ((h ** 2).sum(-1, keepdims=True) + 1e-9).sqrt()           # unit vectors, as in unary loading
        scores = hn @ self.attn(hn).transpose(0, 2, 1)                     # x_i^T W x_j  (B, T, T)
        a = (scores * np.sqrt(self.dim)).softmax(-1)
        h = h + a @ self.value(hn)
        return self.head(h.mean(1))


class QuantumTransformerClassifier(ClassifierMixin, BaseEstimator):
    """Sequences (N, T, D): orthogonal quantum value map and quantum attention scores x_i^T W x_j."""

    def __init__(self, dim=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _build(self, x, y):
        return _QTransformer(x.shape[2], len(self.classes_), self.dim, self.rng_)


class QuantumTransformerRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, dim=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

    def _build(self, x, y):
        return _QTransformer(x.shape[2], 1, self.dim, self.rng_)
