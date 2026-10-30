"""Generative and compression models.

* QCBM (Benedetti et al. 2019; Liu & Wang 2018): quantum circuit Born machine trained with the MMD loss.
* QGAN (Lloyd & Weedbrook 2018; Dallaire-Demers & Killoran 2018): quantum generator vs classical discriminator.
* QuantumAutoencoder (Romero, Olson & Aspuru-Guzik 2017): compresses n-qubit states into fewer qubits.
"""
from __future__ import annotations

import numpy as np

from ..autograd import Tensor, no_grad
from ..nn import Activation, Adam, Linear, Module, Parameter, QuantumLayer, Sequential, binary_cross_entropy, train
from ..quantum.circuit import FIXED_GATES, Circuit, Observable, apply_matrix, simulate
from ..quantum.functions import probs
from ..quantum.gates import param_unitary
from .base import ParamsMixin
from ..quantum.templates import amplitude_encode, angle_encoding, hardware_efficient, strongly_entangling

__all__ = ["QCBM", "QGAN", "QuantumAutoencoder"]


class QCBM(Module):
    """Learns a distribution over n-bit strings; samples are integers in [0, 2**n)."""

    def __init__(self, n_qubits=4, layers=3, bandwidths=(0.25, 1.0, 4.0), seed=0):
        rng = np.random.default_rng(seed)
        self.n_qubits, self.seed = n_qubits, seed
        self.circuit = Circuit(n_qubits)
        for q in range(n_qubits):
            self.circuit.h(q)
        strongly_entangling(self.circuit, layers)
        self.weights = Parameter(rng.uniform(-np.pi, np.pi, self.circuit.n_weights))
        grid = np.arange(2 ** n_qubits)[:, None] / 2 ** n_qubits
        d2 = (grid - grid.T) ** 2 * 2 ** n_qubits
        self.kernel = sum(np.exp(-d2 / (2 * h)) for h in bandwidths) / len(bandwidths)

    def distribution(self):
        return probs(self.circuit, None, None, self.weights)

    def forward(self):
        return self.distribution()

    def fit(self, samples, epochs=200, lr=0.05, verbose=False):
        target = np.bincount(np.asarray(samples, dtype=int), minlength=2 ** self.n_qubits).astype(float)
        target /= target.sum()
        self.target_ = target
        opt = Adam(self.parameters(), lr)
        self.history_ = []
        for epoch in range(epochs):
            self.zero_grad()
            diff = self.distribution() - target
            loss = (diff @ self.kernel * diff).sum()
            loss.backward()
            opt.step()
            self.history_.append(loss.item())
            if verbose and epoch % 20 == 0:
                print(f"epoch {epoch:4d}  MMD {loss.item():.6f}")
        return self

    def probabilities(self):
        with no_grad():
            return self.distribution().data[0]

    def sample(self, n, seed=None):
        p = self.probabilities()
        return np.random.default_rng(seed).choice(len(p), size=n, p=p / p.sum())


class QGAN(ParamsMixin):
    """Continuous data generator: noise z ~ U(0, 2pi)^n -> VQC -> <Z_q> -> affine map to the data scale."""

    def __init__(self, n_qubits=4, layers=3, hidden=16, epochs=200, batch_size=64, lr_g=0.02, lr_d=0.01, seed=0):
        self.n_qubits, self.layers, self.hidden = n_qubits, layers, hidden
        self.epochs, self.batch_size, self.lr_g, self.lr_d, self.seed = epochs, batch_size, lr_g, lr_d, seed

    def fit(self, data, verbose=False):
        data = np.asarray(data, dtype=float)
        data = data[:, None] if data.ndim == 1 else data
        if data.shape[1] > self.n_qubits:
            raise ValueError("need at least one qubit per data dimension")
        rng = np.random.default_rng(self.seed)
        self.lo_, self.hi_ = data.min(0), data.max(0)
        scaled = 2 * (data - self.lo_) / (self.hi_ - self.lo_ + 1e-12) - 1            # [-1, 1]
        self.dim_ = data.shape[1]
        qc = Circuit(self.n_qubits)
        angle_encoding(qc, qc.inputs(self.n_qubits), "Y")
        strongly_entangling(qc, self.layers)
        self.generator_ = QuantumLayer(qc, [Observable(f"Z{q}") for q in range(self.dim_)], None, rng=rng)
        self.discriminator_ = Sequential(Linear(self.dim_, self.hidden, rng=rng), Activation("relu"),
                                         Linear(self.hidden, 1, rng=rng), Activation("sigmoid"))
        opt_g, opt_d = Adam(self.generator_.parameters(), self.lr_g), Adam(self.discriminator_.parameters(), self.lr_d)
        self.history_ = {"d_loss": [], "g_loss": []}
        for epoch in range(self.epochs):
            real = scaled[rng.integers(len(scaled), size=self.batch_size)]
            noise = rng.uniform(0, 2 * np.pi, (self.batch_size, self.n_qubits))
            fake = self.generator_(Tensor(noise))
            self.discriminator_.zero_grad()
            d_loss = binary_cross_entropy(self.discriminator_(Tensor(real))[:, 0], np.ones(self.batch_size)) + \
                binary_cross_entropy(self.discriminator_(fake.detach())[:, 0], np.zeros(self.batch_size))
            d_loss.backward()
            opt_d.step()
            self.generator_.zero_grad()
            g_loss = binary_cross_entropy(self.discriminator_(fake)[:, 0], np.ones(self.batch_size))
            g_loss.backward()
            opt_g.step()
            self.history_["d_loss"].append(d_loss.item())
            self.history_["g_loss"].append(g_loss.item())
            if verbose and epoch % 20 == 0:
                print(f"epoch {epoch:4d}  D {d_loss.item():.4f}  G {g_loss.item():.4f}")
        return self

    def sample(self, n, seed=None):
        noise = np.random.default_rng(seed).uniform(0, 2 * np.pi, (n, self.n_qubits))
        with no_grad():
            out = self.generator_(Tensor(noise)).data
        return (out + 1) / 2 * (self.hi_ - self.lo_) + self.lo_


class QuantumAutoencoder(Module):
    """Encoder U(w) on n qubits; training pushes the ``n_qubits - n_latent`` trash qubits to |0>.

    Data are amplitude-encoded, so a 2**n-dimensional vector fits in n qubits.
    """

    def __init__(self, n_qubits=3, n_latent=2, layers=3, seed=0):
        rng = np.random.default_rng(seed)
        self.n_qubits, self.n_latent = n_qubits, n_latent
        self.trash = list(range(n_latent, n_qubits))
        self.circuit = Circuit(n_qubits)
        hardware_efficient(self.circuit, layers, ("RY", "RZ"), "full")
        self.weights = Parameter(rng.uniform(-np.pi, np.pi, self.circuit.n_weights))

    def _states(self, x):
        return amplitude_encode(x, self.n_qubits)

    def trash_fidelity(self, states):
        return probs(self.circuit, self.trash, None, self.weights, states)[:, 0]

    def fit(self, x, epochs=150, lr=0.05, batch_size=32, seed=0, verbose=False):
        states = self._states(x)
        loss = lambda model, s: 1 - model.trash_fidelity(s).mean()
        self.history_ = train(self, loss, (states,), epochs, batch_size, lr, seed=seed, verbose=verbose)
        return self

    def compression_fidelity(self, x):
        """Probability that the trash qubits end in |0...0> (1 = lossless compression)."""
        with no_grad():
            return self.trash_fidelity(self._states(x)).data

    def reconstruct(self, x):
        """Encode, reset trash qubits to |0>, decode with U^dagger. Returns (states, reconstruction fidelity)."""
        psi = self._states(x)
        enc = simulate(self.circuit, None, self.weights.data, psi)
        idx = [slice(None)] + [slice(None) if q < self.n_latent else slice(0, 1) for q in range(self.n_qubits)]
        kept = np.zeros_like(enc)
        kept[tuple(idx)] = enc[tuple(idx)]
        kept /= np.sqrt(np.sum(np.abs(kept) ** 2, axis=tuple(range(1, kept.ndim)), keepdims=True)) + 1e-12
        out = kept
        for name, wires, p in reversed(self.circuit.ops):
            inv = FIXED_GATES[name].conj().T if p is None else param_unitary(name, -self.weights.data[p.index])
            out = apply_matrix(out, inv, wires)
        out = out.reshape(len(psi), -1)
        fidelity = np.abs(np.sum(np.conj(psi) * out, axis=1)) ** 2
        return out, fidelity
