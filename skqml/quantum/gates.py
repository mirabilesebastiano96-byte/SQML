"""Gate library.

Fixed gates are stored as unitary matrices. Parameterised gates are stored through
their Hermitian *generator* G, with U(theta) = exp(-i * theta * G). Knowing G is all
the adjoint differentiation method needs: dU/dtheta = -i G U.
"""
from __future__ import annotations

import numpy as np

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P0 = np.array([[1, 0], [0, 0]], dtype=complex)
P1 = np.array([[0, 0], [0, 1]], dtype=complex)
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}

FIXED_GATES = {
    "I": I2,
    "X": X,
    "Y": Y,
    "Z": Z,
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "S": np.diag([1, 1j]),
    "SDG": np.diag([1, -1j]),
    "T": np.diag([1, np.exp(1j * np.pi / 4)]),
    "TDG": np.diag([1, np.exp(-1j * np.pi / 4)]),
    "SX": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]),
    "SXDG": 0.5 * np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]),
    "ISWAP": np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex),
    "CNOT": np.kron(P0, I2) + np.kron(P1, X),
    "CZ": np.kron(P0, I2) + np.kron(P1, Z),
    "CY": np.kron(P0, I2) + np.kron(P1, Y),
    "SWAP": np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex),
    "CSWAP": np.eye(8, dtype=complex)[[0, 1, 2, 3, 4, 6, 5, 7]],      # Fredkin: control, target_a, target_b
    "TOFFOLI": np.eye(8, dtype=complex)[[0, 1, 2, 3, 4, 5, 7, 6]],    # CCX: control_a, control_b, target
}

GENERATORS = {
    "RX": X / 2,
    "RY": Y / 2,
    "RZ": Z / 2,
    "PHASE": -P1,                      # diag(1, e^{i theta})
    "RXX": np.kron(X, X) / 2,
    "RYY": np.kron(Y, Y) / 2,
    "RZZ": np.kron(Z, Z) / 2,
    "RZX": np.kron(Z, X) / 2,
    "CRX": np.kron(P1, X) / 2,
    "CRY": np.kron(P1, Y) / 2,
    "CRZ": np.kron(P1, Z) / 2,
    "CPHASE": -np.kron(P1, P1),
    # Reconfigurable beam splitter: real rotation by theta in the {|01>, |10>} subspace (Kerenidis et al.)
    "RBS": np.array([[0, 0, 0, 0], [0, 0, 1j, 0], [0, -1j, 0, 0], [0, 0, 0, 0]], dtype=complex),
    # Givens-like partial swap used in hardware-efficient particle-conserving ansaetze
    "XY": (np.kron(X, X) + np.kron(Y, Y)) / 4,
}

# eigen-decomposition of each generator, so that exp(-i theta G) is exact and batched
_EIG = {name: np.linalg.eigh(g) for name, g in GENERATORS.items()}


def n_wires(name: str) -> int:
    m = FIXED_GATES.get(name, GENERATORS.get(name))
    return int(np.log2(m.shape[0]))


def param_unitary(name: str, theta) -> np.ndarray:
    """U(theta) = exp(-i theta G). ``theta`` scalar -> (d, d); array (B,) -> (B, d, d)."""
    values, vectors = _EIG[name]
    theta = np.asarray(theta, dtype=float)
    phases = np.exp(-1j * np.multiply.outer(theta, values))          # (..., d)
    return np.einsum("ij,...j,kj->...ik", vectors, phases, vectors.conj())
