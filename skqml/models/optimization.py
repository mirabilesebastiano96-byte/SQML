"""Variational quantum algorithms for physics and combinatorial optimisation.

* VQE (Peruzzo et al. 2014): minimise <H> over a parameterised ansatz.
* QAOA (Farhi, Goldstone & Gutmann 2014) for MaxCut.
"""
from __future__ import annotations

import numpy as np

from ..autograd import no_grad
from ..nn import Adam, Module, Parameter
from ..quantum.circuit import Circuit, Observable
from ..quantum.functions import expval, probs
from ..quantum.templates import hardware_efficient

__all__ = ["VQE", "QAOAMaxCut"]


class VQE(Module):
    """``hamiltonian`` is an :class:`Observable` (sum of Pauli strings)."""

    def __init__(self, hamiltonian: Observable, n_qubits: int, layers: int = 3, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.h = hamiltonian
        self.circuit = hardware_efficient(Circuit(n_qubits), layers, ("RY", "RZ"), "linear")
        self.weights = Parameter(rng.uniform(-0.1, 0.1, self.circuit.n_weights))

    def energy(self):
        return expval(self.circuit, [self.h], None, self.weights)[0, 0]

    def fit(self, steps=300, lr=0.05, verbose=False):
        opt = Adam(self.parameters(), lr)
        self.history_ = []
        for step in range(steps):
            self.zero_grad()
            e = self.energy()
            e.backward()
            opt.step()
            self.history_.append(e.item())
            if verbose and step % 50 == 0:
                print(f"step {step:4d}  energy {e.item():.6f}")
        self.energy_ = self.history_[-1]
        return self


class QAOAMaxCut(Module):
    """QAOA for MaxCut on a graph given as a list of edges (i, j)."""

    def __init__(self, edges, n_nodes=None, p=2, seed=0):
        rng = np.random.default_rng(seed)
        self.edges = [tuple(e) for e in edges]
        self.n = n_nodes or (max(max(e) for e in self.edges) + 1)
        qc = Circuit(self.n)
        for q in range(self.n):
            qc.h(q)
        for _ in range(p):
            gamma, beta = qc.weights(1)[0], qc.weights(1)[0]
            for i, j in self.edges:
                qc.rzz(i, j, gamma)
            for q in range(self.n):
                qc.rx(q, beta)
        self.circuit = qc
        self.cost = sum((0.5 * Observable(f"Z{i} Z{j}") for i, j in self.edges[1:]), 0.5 * Observable(
            f"Z{self.edges[0][0]} Z{self.edges[0][1]}"))
        self.weights = Parameter(rng.uniform(0, np.pi / 2, qc.n_weights))

    def expected_cut(self):
        # cut = sum_(ij) (1 - Z_i Z_j)/2  ->  maximise  <=>  minimise sum 0.5 <Z_i Z_j>
        return len(self.edges) / 2 - expval(self.circuit, [self.cost], None, self.weights)[0, 0]

    def fit(self, steps=200, lr=0.05):
        opt = Adam(self.parameters(), lr)
        self.history_ = []
        for _ in range(steps):
            self.zero_grad()
            loss = -self.expected_cut()
            loss.backward()
            opt.step()
            self.history_.append(-loss.item())
        return self

    def cut_value(self, bits):
        return sum(bits[i] != bits[j] for i, j in self.edges)

    def best_cut(self, top=1):
        with no_grad():
            p = probs(self.circuit, None, None, self.weights).data[0]
        order = np.argsort(p)[::-1][:top]
        bits = [[int((k >> (self.n - 1 - q)) & 1) for q in range(self.n)] for k in order]
        return [(b, int(self.cut_value(b)), float(p[k])) for b, k in zip(bits, order)]


# =============================================================================== extensions
from ..quantum.functions import statevector                               # noqa: E402

__all__ += ["VQD", "QAOAIsing", "SPSA", "QuantumNaturalGradient"]


class SPSA:
    """Simultaneous perturbation stochastic approximation (Spall 1992): 2 evaluations per step,
    whatever the number of parameters. The standard optimiser on noisy quantum hardware."""

    def __init__(self, a=0.2, c=0.1, alpha=0.602, gamma=0.101, seed=0):
        self.a, self.c, self.alpha, self.gamma, self.rng = a, c, alpha, gamma, np.random.default_rng(seed)

    def minimize(self, f, x0, steps=200):
        x, history = np.array(x0, dtype=float), []
        for k in range(1, steps + 1):
            ak, ck = self.a / (k + 10) ** self.alpha, self.c / k ** self.gamma
            delta = self.rng.choice([-1.0, 1.0], len(x))
            g = (f(x + ck * delta) - f(x - ck * delta)) / (2 * ck) * delta
            x -= ak * g
            history.append(f(x))
        return x, history


class QuantumNaturalGradient:
    """Quantum natural gradient (Stokes et al., Quantum 2020): theta <- theta - eta * G^+ grad, where G is the
    Fubini-Study metric G_ij = Re(<d_i psi|d_j psi> - <d_i psi|psi><psi|d_j psi>), computed from state vectors."""

    def __init__(self, lr=0.1, reg=1e-3, eps=1e-4):
        self.lr, self.reg, self.eps = lr, reg, eps

    def metric(self, circuit, weights):
        psi = statevector(circuit, None, weights)[0]
        d = []
        for k in range(len(weights)):
            e = np.zeros_like(weights)
            e[k] = self.eps
            d.append((statevector(circuit, None, weights + e)[0] - statevector(circuit, None, weights - e)[0]) / (2 * self.eps))
        d = np.array(d)
        overlap = d.conj() @ psi
        return np.real(d.conj() @ d.T - np.outer(overlap, overlap.conj()))

    def step(self, circuit, weights, grad):
        g = self.metric(circuit, weights) + self.reg * np.eye(len(weights))
        return weights - self.lr * np.linalg.solve(g, grad)


def _vqe_fit(self, steps=300, lr=0.05, optimizer="adam", verbose=False):
    """optimizer: "adam" (exact adjoint gradients), "qng" (quantum natural gradient) or "spsa"."""
    self.history_ = []
    if optimizer == "spsa":
        f = lambda w: expval(self.circuit, [self.h], None, w).data[0, 0]
        w, self.history_ = SPSA().minimize(f, self.weights.data, steps)
        self.weights.data = w
    elif optimizer == "qng":
        qng = QuantumNaturalGradient(lr)
        for _ in range(steps):
            self.zero_grad()
            e = self.energy()
            e.backward()
            self.weights.data = qng.step(self.circuit, self.weights.data, self.weights.grad)
            self.history_.append(e.item())
    else:
        opt = Adam(self.parameters(), lr)
        for step in range(steps):
            self.zero_grad()
            e = self.energy()
            e.backward()
            opt.step()
            self.history_.append(e.item())
            if verbose and step % 50 == 0:
                print(f"step {step:4d}  energy {e.item():.6f}")
    self.energy_ = float(self.energy().item())
    return self


VQE.fit = _vqe_fit


class VQD:
    """Variational quantum deflation (Higgott, Wang & Brierley, Quantum 2019): the k lowest energies,
    each found by VQE with a penalty beta * |<psi|psi_j>|^2 on the states already found."""

    def __init__(self, hamiltonian, n_qubits, k=2, layers=3, beta=3.0, seed=0):
        self.h, self.n, self.k, self.layers, self.beta, self.seed = hamiltonian, n_qubits, k, layers, beta, seed

    def fit(self, steps=400, lr=0.05):
        from ..quantum.functions import probs as _probs
        self.energies_, self.states_ = [], []
        for level in range(self.k):
            vqe = VQE(self.h, self.n, self.layers, seed=self.seed + level)
            opt = Adam(vqe.parameters(), lr)
            for _ in range(steps):
                vqe.zero_grad()
                loss = vqe.energy()
                for prev in self.states_:
                    # |<prev|psi>|^2 = probability of |0..0> after U(w) followed by U_prev^dagger
                    loss = loss + self.beta * _probs(prev, None, None, vqe.weights)[0, 0]
                loss.backward()
                opt.step()
            self.energies_.append(float(vqe.energy().item()))
            w = vqe.weights.data.copy()
            overlap = Circuit(self.n).compose(vqe.circuit)
            fixed = vqe.circuit.inverse()
            for name, wires, p in fixed.ops:                         # bake the found angles into U^dagger
                overlap.ops.append((name, wires, None if p is None else p.coeff * w[p.index]))
            self.states_.append(overlap)
        return self


class QAOAIsing(Module):
    """QAOA for an Ising / QUBO cost  C(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j  (z = +-1), minimised."""

    def __init__(self, h, J, p=2, seed=0):
        rng = np.random.default_rng(seed)
        h, J = np.asarray(h, dtype=float), np.asarray(J, dtype=float)
        self.n = len(h)
        qc = Circuit(self.n)
        for q in range(self.n):
            qc.h(q)
        terms = []
        for _ in range(p):
            gamma, beta = qc.weights(1)[0], qc.weights(1)[0]
            for i in range(self.n):
                if h[i]:
                    qc.rz(i, gamma * float(2 * h[i]))
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    if J[i, j]:
                        qc.rzz(i, j, gamma * float(2 * J[i, j]))
            for q in range(self.n):
                qc.rx(q, beta * 2.0)
        cost = [float(h[i]) * Observable(f"Z{i}") for i in range(self.n) if h[i]]
        cost += [float(J[i, j]) * Observable(f"Z{i} Z{j}") for i in range(self.n) for j in range(i + 1, self.n) if J[i, j]]
        self.cost = sum(cost[1:], cost[0])
        self.h, self.J, self.circuit = h, J, qc
        self.weights = Parameter(rng.uniform(0, 0.5, qc.n_weights))

    def energy(self):
        return expval(self.circuit, [self.cost], None, self.weights)[0, 0]

    def fit(self, steps=200, lr=0.05):
        opt = Adam(self.parameters(), lr)
        self.history_ = []
        for _ in range(steps):
            self.zero_grad()
            e = self.energy()
            e.backward()
            opt.step()
            self.history_.append(e.item())
        return self

    def ising_energy(self, z):
        z = np.asarray(z, dtype=float)
        return float(self.h @ z + z @ np.triu(self.J, 1) @ z)

    def best_solutions(self, top=3):
        with no_grad():
            p = probs(self.circuit, None, None, self.weights).data[0]
        order = np.argsort(p)[::-1][:top]
        out = []
        for k in order:
            z = np.array([1 - 2 * ((k >> (self.n - 1 - q)) & 1) for q in range(self.n)])
            out.append((z.tolist(), self.ising_energy(z), float(p[k])))
        return out
