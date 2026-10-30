"""Quantum Boltzmann machine (Amin et al., Phys. Rev. X 2018).

Fully visible transverse-field Ising model  H = -sum_i b_i Z_i - sum_{i<j} w_ij Z_i Z_j - sum_i g_i X_i.
The model distribution is P(v) = <v| e^{-H} |v> / Tr e^{-H}. Training minimises the exact negative
log-likelihood; the derivative of the matrix exponential uses the Daleckii-Krein formula
d e^{-H}[dH] = V (F o (V^dagger dH V)) V^dagger with F_ij = (e^{-l_i} - e^{-l_j}) / (l_i - l_j).
With g = 0 the model reduces to a classical (fully visible) Boltzmann machine.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..quantum.gates import PAULI
from .base import ParamsMixin

__all__ = ["QuantumBoltzmannMachine"]


def _operator(paulis: dict, n: int) -> np.ndarray:
    m = np.ones((1, 1), dtype=complex)
    for q in range(n):
        m = np.kron(m, PAULI[paulis.get(q, "I")])
    return m


class QuantumBoltzmannMachine(ParamsMixin):
    def __init__(self, n_qubits=4, transverse_field=True, lr=0.05, epochs=200, seed=0):
        self.n, self.transverse, self.lr, self.epochs, self.seed = n_qubits, transverse_field, lr, epochs, seed
        self.n_qubits, self.transverse_field = n_qubits, transverse_field
        n = n_qubits
        self.terms = [{q: "Z"} for q in range(n)] + [{i: "Z", j: "Z"} for i, j in itertools.combinations(range(n), 2)]
        if transverse_field:
            self.terms += [{q: "X"} for q in range(n)]
        self.ops = np.stack([-_operator(t, n) for t in self.terms])        # H = sum_k theta_k ops_k

    def _density(self, theta):
        h = np.tensordot(theta, self.ops, 1)
        lam, v = np.linalg.eigh(h)
        lam_shift = lam - lam.min()
        e = np.exp(-lam_shift)
        rho = (v * e) @ v.conj().T
        return rho / e.sum(), lam_shift, v, e

    def probabilities(self, theta=None):
        rho = self._density(self.theta_ if theta is None else theta)[0]
        return np.real(np.diag(rho))

    def _nll_and_grad(self, theta, q):
        _, lam, v, e = self._density(theta)
        z = e.sum()
        dl = lam[:, None] - lam[None, :]
        same = np.abs(dl) < 1e-10
        f = np.where(same, -e[:, None], (e[:, None] - e[None, :]) / np.where(same, 1, dl))
        p = np.real(np.einsum("ij,j,ij->i", v, e, v.conj())) / z
        mask = q > 0
        nll = -(q[mask] * np.log(p[mask] + 1e-15)).sum()
        grad = np.zeros_like(theta)
        for k, op in enumerate(self.ops):
            m = v.conj().T @ op @ v
            d_exp_diag = np.real(np.einsum("ia,ab,ib->i", v, f * m, v.conj()))       # diag of d e^{-H}
            d_z = -np.real(np.sum(e * np.diag(m)))
            d_p = d_exp_diag / z - p * d_z / z
            grad[k] = -(q[mask] * d_p[mask] / (p[mask] + 1e-15)).sum()
        return nll, grad

    def fit(self, samples, verbose=False):
        """``samples``: integers in [0, 2**n) or binary arrays (N, n)."""
        samples = np.asarray(samples)
        if samples.ndim == 2:
            samples = samples @ (1 << np.arange(self.n)[::-1])
        q = np.bincount(samples.astype(int), minlength=2 ** self.n).astype(float)
        self.target_ = q / q.sum()
        rng = np.random.default_rng(self.seed)
        theta = rng.normal(0, 0.1, len(self.terms))
        m, v = np.zeros_like(theta), np.zeros_like(theta)
        self.history_ = []
        for t in range(1, self.epochs + 1):
            nll, g = self._nll_and_grad(theta, self.target_)
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * g ** 2
            theta -= self.lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
            self.history_.append(nll)
            if verbose and t % 25 == 0:
                print(f"epoch {t:4d}  NLL {nll:.5f}")
        self.theta_ = theta
        return self

    def sample(self, n, seed=None):
        p = self.probabilities()
        return np.random.default_rng(seed).choice(len(p), n, p=p / p.sum())

    def kl_divergence(self):
        p, q = self.probabilities(), self.target_
        m = q > 0
        return float((q[m] * np.log(q[m] / (p[m] + 1e-15))).sum())
