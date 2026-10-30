"""Quantum recurrent models for sequences and time series.

All forecasters take windows ``X`` of shape (N, T, D) and a target ``y`` of shape (N,).

* QRNN       : recurrent cell whose update is a VQC (hybrid quantum RNN).
* QLSTM      : Chen, Yoo & Fang (2020) - the four LSTM gates are VQCs.
* QGRU       : GRU with VQC gates (reset, update, candidate).
* QKANLSTM   : Hsu et al. (2025) - each gate is a KAN layer of single-qubit data re-uploading circuits (DARUAN).
* QuantumReservoirRegressor : quantum reservoir computing (Fujii & Nakajima 2017) with a ridge read-out.
* QMSMoE     : quantum multi-scale memory + mixture of regime experts with probabilistic output.
"""
from __future__ import annotations

import numpy as np

from ..autograd import Tensor, concat, no_grad, stack
from ..nn import Linear, Module, Parameter, QuantumLayer, gaussian_nll, train
from ..quantum.circuit import Circuit, Observable, simulate, apply_matrix
from ..quantum.functions import expval
from ..quantum.templates import variational_circuit
from .base import BaseEstimator, ParamsMixin, RegressorMixin, Scaler

__all__ = ["QRNNRegressor", "QLSTMRegressor", "QGRURegressor", "QKANLSTMRegressor", "QKANLayer",
           "QuantumReservoirRegressor", "QMSMoE", "LSTMRegressor"]


# =============================================================================== cells
class _VQCBlock(Module):
    """Linear(n_in -> n_qubits) -> VQC -> <Z_q> -> Linear(n_qubits -> n_out)."""

    def __init__(self, n_in, n_out, n_qubits, layers, rng, ansatz="hardware_efficient"):
        self.pre = Linear(n_in, n_qubits, rng=rng)
        self.q = QuantumLayer(variational_circuit(n_qubits, layers, ansatz, "angle_h"),
                              None, "arctan", rng=rng)
        self.post = Linear(n_qubits, n_out, rng=rng)

    def forward(self, v):
        return self.post(self.q(self.pre(v)))


class QRNNCell(Module):
    def __init__(self, n_in, hidden, n_qubits=4, layers=2, rng=None, ansatz="hardware_efficient"):
        self.hidden, self.block = hidden, _VQCBlock(n_in + hidden, hidden, n_qubits, layers, rng, ansatz)

    def init_state(self, b):
        return Tensor(np.zeros((b, self.hidden)))

    def forward(self, x, h):
        h = self.block(concat([x, h], -1)).tanh()
        return h, h


class QLSTMCell(Module):
    def __init__(self, n_in, hidden, n_qubits=4, layers=2, rng=None, ansatz="hardware_efficient"):
        self.hidden = hidden
        self.gates = [_VQCBlock(n_in + hidden, hidden, n_qubits, layers, rng, ansatz) for _ in range(4)]
        self.gates[0].post.b.data += 1.0                                     # forget-gate bias

    def init_state(self, b):
        z = np.zeros((b, self.hidden))
        return (Tensor(z), Tensor(z.copy()))

    def forward(self, x, state):
        h, c = state
        v = concat([x, h], -1)
        f, i, g, o = (blk(v) for blk in self.gates)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        return h, (h, c)


class QGRUCell(Module):
    def __init__(self, n_in, hidden, n_qubits=4, layers=2, rng=None, ansatz="hardware_efficient"):
        self.hidden = hidden
        self.r = _VQCBlock(n_in + hidden, hidden, n_qubits, layers, rng, ansatz)
        self.z = _VQCBlock(n_in + hidden, hidden, n_qubits, layers, rng, ansatz)
        self.n = _VQCBlock(n_in + hidden, hidden, n_qubits, layers, rng, ansatz)

    def init_state(self, b):
        return Tensor(np.zeros((b, self.hidden)))

    def forward(self, x, h):
        v = concat([x, h], -1)
        r, z = self.r(v).sigmoid(), self.z(v).sigmoid()
        n = self.n(concat([x, r * h], -1)).tanh()
        h = (1 - z) * n + z * h
        return h, h


def _daruan_circuit(reps):
    """Single-qubit DARUAN: H, [RZ(t) RY(t) RZ(data)] x reps, RZ(t) RY(t). All angles are inputs."""
    qc = Circuit(1)
    a = qc.inputs(3 * reps + 2)
    qc.h(0)
    for l in range(reps):
        qc.rz(0, a[3 * l]).ry(0, a[3 * l + 1]).rz(0, a[3 * l + 2])
    qc.rz(0, a[3 * reps]).ry(0, a[3 * reps + 1])
    return qc


class QKANLayer(Module):
    """KAN layer: out_j = sum_i phi_ij(v_i), each phi a DARUAN circuit with its own parameters."""

    def __init__(self, n_in, n_out, reps=2, rng=None):
        rng = rng or np.random.default_rng()
        self.n_in, self.n_out, self.reps = n_in, n_out, reps
        self.theta = Parameter(rng.uniform(-np.pi, np.pi, (n_in, n_out, reps + 1, 2)))
        self.a = Parameter(np.ones((n_in, n_out, reps)))
        self.b = Parameter(np.zeros((n_in, n_out, reps)))
        self.circuit = _daruan_circuit(reps)
        self.obs = [Observable("Z0")]

    def forward(self, v):
        bsz = v.shape[0]
        zeros = np.zeros((bsz, self.n_in, self.n_out))
        data = v.reshape(bsz, self.n_in, 1, 1) * self.a + self.b                  # (B, in, out, reps)
        cols = []
        for l in range(self.reps):
            cols += [self.theta[:, :, l, 0] + zeros, self.theta[:, :, l, 1] + zeros, data[:, :, :, l]]
        cols += [self.theta[:, :, self.reps, 0] + zeros, self.theta[:, :, self.reps, 1] + zeros]
        angles = stack(cols, -1).reshape(bsz * self.n_in * self.n_out, 3 * self.reps + 2)
        z = expval(self.circuit, self.obs, angles)                                  # (B*in*out, 1)
        return z.reshape(bsz, self.n_in, self.n_out).sum(1)


class QKANLSTMCell(Module):
    def __init__(self, n_in, hidden, reps=2, rng=None):
        self.hidden = hidden
        self.gates = [QKANLayer(n_in + hidden, hidden, reps, rng) for _ in range(4)]

    def init_state(self, b):
        z = np.zeros((b, self.hidden))
        return (Tensor(z), Tensor(z.copy()))

    def forward(self, x, state):
        h, c = state
        v = concat([x, h], -1)
        f, i, g, o = (gate(v) for gate in self.gates)
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        return h, (h, c)


class LSTMCell(Module):
    """Classical LSTM cell, handy as a benchmark."""

    def __init__(self, n_in, hidden, rng=None, **_):
        self.hidden = hidden
        self.lin = Linear(n_in + hidden, 4 * hidden, rng=rng)
        self.lin.b.data[:hidden] = 1.0

    def init_state(self, b):
        z = np.zeros((b, self.hidden))
        return (Tensor(z), Tensor(z.copy()))

    def forward(self, x, state):
        h, c = state
        z = self.lin(concat([x, h], -1))
        H = self.hidden
        f, i, g, o = z[:, :H], z[:, H:2 * H], z[:, 2 * H:3 * H], z[:, 3 * H:]
        c = f.sigmoid() * c + i.sigmoid() * g.tanh()
        h = o.sigmoid() * c.tanh()
        return h, (h, c)


class _SequenceNet(Module):
    def __init__(self, cell, hidden, n_out, rng):
        self.cell = cell
        self.head = Linear(hidden, n_out, rng=rng)

    def forward(self, x):
        state = self.cell.init_state(x.shape[0])
        h = None
        for t in range(x.shape[1]):
            h, state = self.cell(x[:, t, :], state)
        return self.head(h)


class _RecurrentRegressor(RegressorMixin, BaseEstimator):
    cell_class = None

    def __init__(self, hidden=4, n_qubits=4, layers=2, ansatz="hardware_efficient", **kwargs):
        super().__init__(**kwargs)
        self.hidden, self.n_qubits, self.layers, self.ansatz = hidden, n_qubits, layers, ansatz

    def _cell(self, n_in):
        return self.cell_class(n_in, self.hidden, n_qubits=self.n_qubits, layers=self.layers, rng=self.rng_,
                               ansatz=self.ansatz)

    def _build(self, x, y):
        return _SequenceNet(self._cell(x.shape[2]), self.hidden, 1, self.rng_)


class QRNNRegressor(_RecurrentRegressor):
    cell_class = QRNNCell


class QLSTMRegressor(_RecurrentRegressor):
    cell_class = QLSTMCell


class QGRURegressor(_RecurrentRegressor):
    cell_class = QGRUCell


class LSTMRegressor(_RecurrentRegressor):
    cell_class = LSTMCell


class QKANLSTMRegressor(_RecurrentRegressor):
    def __init__(self, hidden=1, reps=2, **kwargs):
        super().__init__(hidden=hidden, **kwargs)
        self.reps = reps

    def _cell(self, n_in):
        return QKANLSTMCell(n_in, self.hidden, self.reps, self.rng_)


# =============================================================================== reservoir
class QuantumReservoirRegressor(ParamsMixin):
    """Fixed random quantum system driven by the inputs; ridge regression on the measured observables.

    At every time step the inputs are written on the first D qubits with RY(pi * tanh(x)), then a fixed random
    unitary scrambles the whole register. <Z_q> and <Z_q Z_q+1> are read after each of the last ``n_read`` steps.
    Only the linear read-out is trained (closed form), so fitting is very fast.
    """

    def __init__(self, n_qubits=6, depth=2, alpha=1e-2, n_read=5, seed=0):
        self.n_qubits, self.depth, self.alpha, self.n_read, self.seed = n_qubits, depth, alpha, n_read, seed

    def _build(self, n_in):
        rng = np.random.default_rng(self.seed)
        n = self.n_qubits
        step = Circuit(n)
        x = step.inputs(n_in)
        for d in range(n_in):
            step.ry(d % n, x[d])
        for _ in range(self.depth):
            for q in range(n):
                step.rx(q, float(rng.uniform(0, 2 * np.pi))).rz(q, float(rng.uniform(0, 2 * np.pi)))
            for q in range(n):
                step.cnot(q, (q + 1) % n)
        self.step_ = step
        self.obs_ = [Observable(f"Z{q}") for q in range(n)] + [Observable(f"Z{q} Z{(q + 1) % n}") for q in range(n)]

    def features(self, x):
        x = self.scaler_.transform(x)
        n, steps, _ = x.shape
        state = None
        feats = []
        for t in range(steps):
            state = simulate(self.step_, np.pi * np.tanh(x[:, t]), None,
                             None if state is None else state.reshape(n, -1))
            if t >= steps - self.n_read:
                feats.append(np.stack([np.real(np.sum(np.conj(state) * o.apply(state), axis=tuple(range(1, state.ndim))))
                                       for o in self.obs_], 1))
        return np.concatenate(feats + [np.ones((n, 1))], 1)

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        self.scaler_ = Scaler().fit(x)
        self._build(x.shape[2])
        f = self.features(x)
        self.coef_ = np.linalg.solve(f.T @ f + self.alpha * np.eye(f.shape[1]), f.T @ np.asarray(y, dtype=float))
        return self

    def predict(self, x):
        return self.features(np.asarray(x, dtype=float)) @ self.coef_

    def score(self, x, y):
        y = np.asarray(y, dtype=float)
        return float(1 - np.sum((self.predict(x) - y) ** 2) / np.sum((y - y.mean()) ** 2))


# =============================================================================== QMS-MoE
def _ema(x, span):
    alpha = 2 / (span + 1)
    u = np.zeros_like(x)
    u[:, 0] = x[:, 0]
    for t in range(1, x.shape[1]):
        u[:, t] = alpha * x[:, t] + (1 - alpha) * u[:, t - 1]
    return u


def _context(win):
    """Six domain-agnostic statistics per variable of a window (N, T, D)."""
    steps = win.shape[1]
    short = max(2, steps // 4)
    tt = np.arange(steps) - (steps - 1) / 2
    slope = np.einsum("t,ntd->nd", tt, win) / (tt ** 2).sum()
    log_sd = np.log(win.std(1) + 1e-3)
    return np.concatenate([win[:, -1], win[:, -short:].mean(1), win.mean(1), log_sd,
                           np.log(win[:, -short:].std(1) + 1e-3) - log_sd, slope], 1)


def memory_circuit(n_qubits: int, steps: int) -> Circuit:
    """Quantum memory unrolled over time: per step RY(data) -> RY(beta) RZ(gamma) -> CNOT ring.
    The state is never reset; beta and gamma are shared by all steps."""
    qc = Circuit(n_qubits)
    beta, gamma = qc.weights(n_qubits), qc.weights(n_qubits)
    for _ in range(steps):
        x = qc.inputs(n_qubits)
        for q in range(n_qubits):
            qc.ry(q, x[q])
        for q in range(n_qubits):
            qc.ry(q, beta[q]).rz(q, gamma[q])
        if n_qubits > 1:
            for q in range(n_qubits if n_qubits > 2 else 1):
                qc.cnot(q, (q + 1) % n_qubits)
    return qc


class _QMSMoENet(Module):
    def __init__(self, n_vars, n_scales, steps, n_context, n_qubits, n_experts, hidden, use_quantum,
                 shared_mean, rng):
        self.use_quantum, self.shared_mean, self.n_qubits, self.m = use_quantum, shared_mean, n_qubits, n_experts
        if use_quantum:
            self.circuit = memory_circuit(n_qubits, steps)
            self.obs = [Observable(f"Z{q}") for q in range(n_qubits)] + \
                       [Observable(f"Z{q} Z{(q + 1) % n_qubits}") for q in range(n_qubits)]
            self.proj = [Linear(n_vars, n_qubits, rng=rng) for _ in range(n_scales)]
            self.mem_w = [Parameter(rng.uniform(-np.pi, np.pi, self.circuit.n_weights)) for _ in range(n_scales)]
        d = (n_scales * 2 * n_qubits if use_quantum else 0) + n_context
        self.gate = Linear(n_context, n_experts, rng=rng)
        self.gate.w.data *= 0.1
        self.experts = [Linear(d, hidden, rng=rng) for _ in range(n_experts)]
        self.sigma_heads = [Linear(hidden, 1, rng=rng) for _ in range(n_experts)]
        for s in self.sigma_heads:
            s.w.data *= 0.1
            s.b.data[:] = np.log(np.e - 1)
        if shared_mean:
            self.mean_head = Linear(d, 1, rng=rng)
            self.mean_head.w.data[:] = 0
        else:
            self.mean_heads = [Linear(hidden, 1, rng=rng) for _ in range(n_experts)]
            for mh in self.mean_heads:
                mh.w.data *= 0.1

    def memory_features(self, u):
        bsz, n_scales, steps, _ = u.shape
        feats = []
        for k in range(n_scales):
            angles = self.proj[k](Tensor(u[:, k])).reshape(bsz, steps * self.n_qubits)
            feats.append(expval(self.circuit, self.obs, angles, self.mem_w[k]))
        return feats

    def forward(self, u, c):
        c = Tensor(c)
        feats = self.memory_features(u) if self.use_quantum else []
        z = concat(feats + [c], -1)
        pi = self.gate(c).softmax(-1)
        hs = [e(z).tanh() for e in self.experts]
        sigma = concat([head(h) for head, h in zip(self.sigma_heads, hs)], -1).softplus() + 0.01
        if self.shared_mean:
            mu = self.mean_head(z) + np.zeros((1, self.m))
        else:
            mu = concat([head(h) for head, h in zip(self.mean_heads, hs)], -1)
        return pi, mu, sigma


class QMSMoE(ParamsMixin):
    """Quantum Multi-Scale memory + Mixture of Experts (probabilistic forecaster for any time series).

    * K quantum memories (one per EMA scale) whose 4-qubit state evolves across the window and is never reset;
    * a gate driven by domain-agnostic context statistics weighs M experts (regimes);
    * the output is a Gaussian mixture  p(y) = sum_m pi_m N(mu_m, sigma_m^2), trained by negative log-likelihood.

    ``shared_mean=True`` makes all experts share one linear mean (a scale mixture: good for financial returns).
    """

    def __init__(self, scales=None, n_qubits=4, n_experts=3, hidden=8, use_quantum=True, shared_mean=False,
                 mean_l2=0.0, epochs=60, lr=5e-3, batch_size=64, patience=10, validation_split=0.15, seed=0,
                 verbose=False):
        self.scales, self.n_qubits, self.n_experts, self.hidden = scales, n_qubits, n_experts, hidden
        self.use_quantum, self.shared_mean, self.mean_l2 = use_quantum, shared_mean, mean_l2
        self.epochs, self.lr, self.batch_size, self.patience = epochs, lr, batch_size, patience
        self.validation_split, self.seed, self.verbose = validation_split, seed, verbose

    def _inputs(self, x, fit=False):
        x = np.asarray(x, dtype=float)
        if fit:
            self.scaler_ = Scaler().fit(x)
            steps = x.shape[1]
            self.scales_ = self.scales or (1, max(2, steps // 4), steps)
        xs = np.clip(self.scaler_.transform(x), -6, 6)
        u = np.stack([np.stack([_ema(xs[:, :, d], s) for d in range(xs.shape[2])], -1) for s in self.scales_], 1)
        c = _context(xs)
        if fit:
            self.c_mean_, self.c_std_ = c.mean(0), c.std(0) + 1e-9
        return u, (c - self.c_mean_) / self.c_std_

    def _nll(self, net, u, c, y):
        pi, mu, sigma = net(u, c)
        logp = (pi + 1e-12).log() - sigma.log() - 0.5 * np.log(2 * np.pi) - 0.5 * ((y[:, None] - mu) / sigma) ** 2
        loss = -logp.logsumexp(-1).mean()
        if self.shared_mean and self.mean_l2:
            loss = loss + 0.5 * self.mean_l2 * (net.mean_head.w ** 2).sum()
        return loss

    def fit(self, x, y):
        y = np.asarray(y, dtype=float)
        self.y_mean_, self.y_std_ = y.mean(), y.std() + 1e-9
        u, c = self._inputs(x, fit=True)
        ys = (y - self.y_mean_) / self.y_std_
        rng = np.random.default_rng(self.seed)
        self.net_ = _QMSMoENet(u.shape[3], u.shape[1], u.shape[2], c.shape[1], self.n_qubits, self.n_experts,
                               self.hidden, self.use_quantum, self.shared_mean, rng)
        cut = int(len(ys) * (1 - self.validation_split))
        val = (u[cut:], c[cut:], ys[cut:]) if self.validation_split > 0 else None
        self.history_ = train(self.net_, self._nll, (u[:cut], c[:cut], ys[:cut]), self.epochs, self.batch_size,
                              self.lr, val_data=val, patience=self.patience, seed=self.seed, verbose=self.verbose)
        return self

    def predict_distribution(self, x):
        """Mixture weights, means and standard deviations in the original units: three (N, M) arrays."""
        u, c = self._inputs(x)
        with no_grad():
            pi, mu, sigma = self.net_(u, c)
        return pi.data, mu.data * self.y_std_ + self.y_mean_, sigma.data * self.y_std_

    def predict(self, x):
        pi, mu, _ = self.predict_distribution(x)
        return (pi * mu).sum(1)

    def predict_std(self, x):
        pi, mu, sigma = self.predict_distribution(x)
        mean = (pi * mu).sum(1, keepdims=True)
        return np.sqrt((pi * (sigma ** 2 + (mu - mean) ** 2)).sum(1))

    def predict_interval(self, x, level=0.9):
        pi, mu, sigma = self.predict_distribution(x)
        lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
        return _mixture_quantile(pi, mu, sigma, lo), _mixture_quantile(pi, mu, sigma, hi)

    def log_likelihood(self, x, y):
        pi, mu, sigma = self.predict_distribution(x)
        y = np.asarray(y, dtype=float)[:, None]
        logp = np.log(pi + 1e-12) - np.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * ((y - mu) / sigma) ** 2
        m = logp.max(1, keepdims=True)
        return (m[:, 0] + np.log(np.exp(logp - m).sum(1)))

    def score(self, x, y):
        y = np.asarray(y, dtype=float)
        return float(1 - np.sum((self.predict(x) - y) ** 2) / np.sum((y - y.mean()) ** 2))

    def n_parameters(self):
        return self.net_.n_parameters()


def _erf(x):
    s, x = np.sign(x), np.abs(x)
    t = 1 / (1 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))))
    return s * (1 - poly * np.exp(-x * x))


def _mixture_quantile(pi, mu, sigma, level, iters=60):
    lo, hi = (mu - 8 * sigma).min(1), (mu + 8 * sigma).max(1)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        cdf = (pi * 0.5 * (1 + _erf((mid[:, None] - mu) / (sigma * np.sqrt(2))))).sum(1)
        lo, hi = np.where(cdf < level, mid, lo), np.where(cdf >= level, mid, hi)
    return 0.5 * (lo + hi)
