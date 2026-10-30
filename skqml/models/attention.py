"""Quantum self-attention neural network (QSANN, Li et al. 2022).

Queries and keys are single expectation values of two VQCs, values are the expectation vector of a third VQC.
Attention uses a Gaussian projected kernel: alpha_ij = softmax_j( -(q_i - k_j)^2 ).
"""
from __future__ import annotations

import numpy as np

from ..autograd import Tensor
from ..nn import Linear, Module, QuantumLayer
from ..quantum.circuit import Observable
from ..quantum.templates import variational_circuit
from .base import BaseEstimator, ClassifierMixin, RegressorMixin

__all__ = ["QSANNClassifier", "QSANNRegressor", "QuantumSelfAttention"]


class QuantumSelfAttention(Module):
    def __init__(self, n_qubits, layers=1, rng=None):
        make = lambda obs: QuantumLayer(variational_circuit(n_qubits, layers, "hardware_efficient"),
                                        obs, "arctan", rng=rng)
        self.query = make([Observable("Z0")])
        self.key = make([Observable("Z0")])
        self.value = make(None)

    def forward(self, tokens):
        b, t, n = tokens.shape
        flat = tokens.reshape(b * t, n)
        q = self.query(flat).reshape(b, t, 1)
        k = self.key(flat).reshape(b, 1, t)
        v = self.value(flat).reshape(b, t, n)
        alpha = (-((q - k) ** 2)).softmax(-1)               # (B, T, T)
        return tokens + alpha @ v                           # residual connection, as in the paper


class _QSANNNet(Module):
    def __init__(self, n_features, n_out, n_qubits, n_blocks, layers, rng):
        self.embed = Linear(n_features, n_qubits, rng=rng)
        self.blocks = [QuantumSelfAttention(n_qubits, layers, rng) for _ in range(n_blocks)]
        self.head = Linear(n_qubits, n_out, rng=rng)

    def forward(self, x):
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h.mean(1))


class QSANNClassifier(ClassifierMixin, BaseEstimator):
    """Input: sequences (N, T, D) such as token embeddings or time-series windows."""

    def __init__(self, n_qubits=4, n_blocks=1, layers=1, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits, self.n_blocks, self.layers = n_qubits, n_blocks, layers

    def _build(self, x, y):
        return _QSANNNet(x.shape[2], len(self.classes_), self.n_qubits, self.n_blocks, self.layers, self.rng_)


class QSANNRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, n_qubits=4, n_blocks=1, layers=1, **kwargs):
        super().__init__(**kwargs)
        self.n_qubits, self.n_blocks, self.layers = n_qubits, n_blocks, layers

    def _build(self, x, y):
        return _QSANNNet(x.shape[2], 1, self.n_qubits, self.n_blocks, self.layers, self.rng_)
