"""Reusable encodings and ansatz layers."""
from __future__ import annotations

import itertools

import numpy as np

from .circuit import Circuit

__all__ = ["angle_encoding", "amplitude_encode", "zz_feature_map", "zz_features", "hardware_efficient",
           "strongly_entangling", "real_amplitudes", "entangler", "variational_circuit"]


def angle_encoding(qc: Circuit, x_refs, rotation: str = "Y", hadamard: bool = False):
    """One input per qubit: R_rot(x_q) on qubit q (optionally after a Hadamard)."""
    for q, ref in enumerate(x_refs):
        if hadamard:
            qc.h(q)
        qc.add("R" + rotation.upper(), q, ref)
    return qc


def amplitude_encode(x: np.ndarray, n_qubits: int) -> np.ndarray:
    """Amplitude encoding: pad each row to 2**n values and L2-normalise. Use as ``initial_state``."""
    x = np.atleast_2d(np.asarray(x, dtype=float))
    dim = 2 ** n_qubits
    if x.shape[1] > dim:
        raise ValueError(f"{x.shape[1]} features do not fit in {n_qubits} qubits")
    padded = np.zeros((len(x), dim))
    padded[:, :x.shape[1]] = x
    norms = np.linalg.norm(padded, axis=1, keepdims=True)
    padded[norms[:, 0] == 0, 0] = 1.0
    return (padded / np.where(norms == 0, 1, norms)).astype(complex)


def zz_features(x: np.ndarray) -> np.ndarray:
    """Classical pre-processing for the ZZ feature map: [x_i, (pi - x_i)(pi - x_j) for i<j]."""
    x = np.asarray(x, dtype=float)
    pairs = [(np.pi - x[:, i]) * (np.pi - x[:, j]) for i, j in itertools.combinations(range(x.shape[1]), 2)]
    return np.concatenate([x, np.stack(pairs, 1)], 1) if pairs else x


def zz_feature_map(n_qubits: int, reps: int = 2) -> Circuit:
    """Havlicek et al. (2019) ZZ feature map. Feed it ``zz_features(X)`` as inputs."""
    qc = Circuit(n_qubits)
    single = qc.inputs(n_qubits)
    pairs = list(itertools.combinations(range(n_qubits), 2))
    pair_refs = qc.inputs(len(pairs))
    for _ in range(reps):
        for q in range(n_qubits):
            qc.h(q)
            qc.add("PHASE", q, single[q])           # e^{i x} phase; equivalent to RZ(2x) up to global phase
        for (i, j), ref in zip(pairs, pair_refs):
            qc.cnot(i, j).add("PHASE", j, ref).cnot(i, j)
    return qc


def entangler(qc: Circuit, kind: str = "ring", gate: str = "CNOT"):
    n = qc.n_qubits
    if n < 2:
        return qc
    if kind == "ring":
        pairs = [(q, (q + 1) % n) for q in range(n)] if n > 2 else [(0, 1)]
    elif kind == "linear":
        pairs = [(q, q + 1) for q in range(n - 1)]
    elif kind == "full":
        pairs = list(itertools.combinations(range(n), 2))
    else:
        raise ValueError(kind)
    for c, t in pairs:
        qc.add(gate, (c, t))
    return qc


def hardware_efficient(qc: Circuit, layers: int = 2, rotations=("RY", "RZ"), entangle: str = "ring"):
    """Layers of single-qubit rotations followed by an entangling pattern."""
    for _ in range(layers):
        for q in range(qc.n_qubits):
            for r in rotations:
                qc.add(r, q, qc.weights(1)[0])
        entangler(qc, entangle)
    return qc


def strongly_entangling(qc: Circuit, layers: int = 2):
    """Schuld et al. (2020): Rot = RZ RY RZ on each qubit, then CNOTs with a layer-dependent range."""
    n = qc.n_qubits
    for layer in range(layers):
        for q in range(n):
            w = qc.weights(3)
            qc.rz(q, w[0]).ry(q, w[1]).rz(q, w[2])
        if n > 1:
            r = layer % (n - 1) + 1
            for q in range(n):
                qc.cnot(q, (q + r) % n)
    return qc


def real_amplitudes(qc: Circuit, layers: int = 2, entangle: str = "linear"):
    """Qiskit-style RealAmplitudes: RY layers and CNOT entanglers, final RY layer."""
    for _ in range(layers):
        for q in range(qc.n_qubits):
            qc.ry(q, qc.weights(1)[0])
        entangler(qc, entangle)
    for q in range(qc.n_qubits):
        qc.ry(q, qc.weights(1)[0])
    return qc


def variational_circuit(n_qubits: int, layers: int = 2, ansatz: str = "strongly_entangling",
                        encoding: str = "angle", reupload: bool = False) -> Circuit:
    """Encoding + ansatz in one call. With ``reupload=True`` the encoding is repeated before every layer
    (data re-uploading, Perez-Salinas et al. 2020)."""
    qc = Circuit(n_qubits)
    x = qc.inputs(n_qubits)
    blocks = layers if reupload else 1
    per_block = 1 if reupload else layers
    for _ in range(blocks):
        if encoding == "angle":
            angle_encoding(qc, x, "Y")
        elif encoding == "angle_h":
            angle_encoding(qc, x, "Y", hadamard=True)
        else:
            raise ValueError(encoding)
        {"strongly_entangling": strongly_entangling, "hardware_efficient": hardware_efficient,
         "real_amplitudes": real_amplitudes}[ansatz](qc, per_block)
    return qc
