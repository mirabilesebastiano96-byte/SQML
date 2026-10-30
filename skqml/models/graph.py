"""Quantum graph neural network.

QGNNClassifier follows the equivariant, QAOA-inspired graph circuits of Verdon et al. (2019) and
Skolik et al. (2023, "Equivariant quantum circuits for learning on weighted graphs"):
one qubit per node, node features on RY, edge weights scale shared RZZ angles, shared RX mixers.
The read-out is permutation invariant (mean of <Z_i> and of <Z_i Z_j> over edges).
"""
from __future__ import annotations

import numpy as np

from ..autograd import Tensor, concat, no_grad
from ..nn import Linear, Module, Parameter, cross_entropy, mse_loss, train
from ..quantum.circuit import Circuit, Observable, WeightParam
from ..quantum.functions import expval
from .base import ParamsMixin

__all__ = ["QGNNClassifier", "QGNNRegressor"]


def _graph_circuit(adjacency, layers):
    n = len(adjacency)
    qc = Circuit(n)
    x = qc.inputs(n)
    gammas, betas = qc.weights(layers), qc.weights(layers)
    qc.n_weights = 2 * layers
    for q in range(n):
        qc.h(q).ry(q, x[q])
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if adjacency[i][j] != 0]
    for l in range(layers):
        for i, j in edges:
            qc.rzz(i, j, WeightParam(gammas[l].index, float(adjacency[i][j])))
        for q in range(n):
            qc.rx(q, betas[l])
    return qc, edges


class _QGNN(Module):
    def __init__(self, layers, n_out, rng):
        self.layers = layers
        self.weights = Parameter(rng.uniform(0, np.pi, 2 * layers))
        self.head = Linear(2, n_out, rng=rng)
        self._cache = {}

    def embed(self, adjacency, features):
        key = np.asarray(adjacency).tobytes() + bytes(len(adjacency))
        if key not in self._cache:
            qc, edges = _graph_circuit(adjacency, self.layers)
            n = len(adjacency)
            obs_nodes = [Observable(f"Z{q}") for q in range(n)]
            obs_edges = [Observable(f"Z{i} Z{j}") for i, j in edges] or [Observable("Z0")]
            self._cache[key] = (qc, obs_nodes, obs_edges)
        qc, obs_nodes, obs_edges = self._cache[key]
        angles = Tensor(np.arctan(np.asarray(features, dtype=float))[None, :] * 2)
        z = expval(qc, obs_nodes + obs_edges, angles, self.weights)                # (1, n + e)
        n = len(obs_nodes)
        return concat([z[:, :n].mean(1, keepdims=True), z[:, n:].mean(1, keepdims=True)], -1)

    def forward(self, graphs):
        return self.head(concat([self.embed(a, f) for a, f in graphs], 0))


class _QGNNBase(ParamsMixin):
    def __init__(self, layers=2, epochs=40, lr=0.05, batch_size=16, seed=0):
        self.layers, self.epochs, self.lr, self.batch_size, self.seed = layers, epochs, lr, batch_size, seed

    def _graphs(self, adjacencies, features=None):
        if features is None:
            features = [np.asarray(a).sum(1) for a in adjacencies]                 # node degree by default
        return [(np.asarray(a, dtype=float), np.asarray(f, dtype=float)) for a, f in zip(adjacencies, features)]

    def _fit(self, graphs, targets, n_out, loss):
        self.net_ = _QGNN(self.layers, n_out, np.random.default_rng(self.seed))
        index = np.arange(len(graphs))
        self.history_ = train(self.net_, lambda m, idx, t: loss(m([graphs[i] for i in idx]), t),
                              (index, targets), self.epochs, self.batch_size, self.lr, seed=self.seed)
        return self

    def _out(self, graphs):
        with no_grad():
            return self.net_(graphs).data


class QGNNClassifier(_QGNNBase):
    """``fit(adjacencies, y, features=None)``: list of (n x n) weighted adjacency matrices, optional node features."""

    def fit(self, adjacencies, y, features=None):
        self.classes_, idx = np.unique(y, return_inverse=True)
        return self._fit(self._graphs(adjacencies, features), idx, len(self.classes_), cross_entropy)

    def predict(self, adjacencies, features=None):
        return self.classes_[np.argmax(self._out(self._graphs(adjacencies, features)), 1)]

    def score(self, adjacencies, y, features=None):
        return float(np.mean(self.predict(adjacencies, features) == np.asarray(y)))


class QGNNRegressor(_QGNNBase):
    def fit(self, adjacencies, y, features=None):
        y = np.asarray(y, dtype=float)
        self.y_mean_, self.y_std_ = y.mean(), y.std() + 1e-9
        loss = lambda out, t: mse_loss(out[:, 0], t)
        return self._fit(self._graphs(adjacencies, features), (y - self.y_mean_) / self.y_std_, 1, loss)

    def predict(self, adjacencies, features=None):
        return self._out(self._graphs(adjacencies, features))[:, 0] * self.y_std_ + self.y_mean_
