"""Batched state-vector simulator, circuit builder and adjoint differentiation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gates import FIXED_GATES, GENERATORS, PAULI, n_wires, param_unitary

__all__ = ["Circuit", "InputParam", "WeightParam", "Observable", "Z", "X", "Y", "apply_matrix",
           "zero_state", "simulate", "adjoint_gradients"]


# ----------------------------------------------------------------------------- parameters
@dataclass(frozen=True)
class InputParam:
    """Angle taken from the data: ``coeff * inputs[:, index]``."""
    index: int
    coeff: float = 1.0

    def __neg__(self):
        return InputParam(self.index, -self.coeff)

    def __mul__(self, c):
        return InputParam(self.index, self.coeff * c)

    __rmul__ = __mul__


@dataclass(frozen=True)
class WeightParam:
    """Trainable angle: ``coeff * weights[index]``."""
    index: int
    coeff: float = 1.0

    def __neg__(self):
        return WeightParam(self.index, -self.coeff)

    def __mul__(self, c):
        return WeightParam(self.index, self.coeff * c)

    __rmul__ = __mul__


_DAGGER = {"S": "SDG", "SDG": "S", "T": "TDG", "TDG": "T", "SX": "SXDG", "SXDG": "SX"}


# ----------------------------------------------------------------------------- simulator
def zero_state(batch: int, n: int) -> np.ndarray:
    state = np.zeros((batch,) + (2,) * n, dtype=complex)
    state[(slice(None),) + (0,) * n] = 1.0
    return state


def apply_matrix(state: np.ndarray, matrix: np.ndarray, wires) -> np.ndarray:
    """Apply a (d,d) or batched (B,d,d) matrix to ``wires`` of a batched state (B,2,...,2)."""
    k = len(wires)
    axes = [w + 1 for w in wires]
    s = np.moveaxis(state, axes, list(range(-k, 0)))
    shape = s.shape
    s = s.reshape(shape[0], -1, 2 ** k)
    if matrix.ndim == 2:
        s = s @ matrix.T
    else:
        s = s @ np.transpose(matrix, (0, 2, 1))
    return np.moveaxis(s.reshape(shape), list(range(-k, 0)), axes)


def braket(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """<a|b> for every element of the batch."""
    return np.sum(np.conj(a) * b, axis=tuple(range(1, a.ndim)))


# ----------------------------------------------------------------------------- observables
class Observable:
    """A weighted sum of Pauli strings, e.g. ``Observable("Z0 Z1") + 0.5 * Observable("X2")``."""

    def __init__(self, spec: str | None = None, coeff: float = 1.0, terms=None):
        if terms is not None:
            self.terms = terms
        else:
            paulis = {}
            for token in spec.replace("*", " ").split():
                paulis[int(token[1:])] = token[0].upper()
            self.terms = [(coeff, paulis)]

    def __add__(self, other):
        return Observable(terms=self.terms + other.terms)

    def __mul__(self, c):
        return Observable(terms=[(c * k, p) for k, p in self.terms])

    __rmul__ = __mul__

    def apply(self, state: np.ndarray) -> np.ndarray:
        """Return O|psi>."""
        out = np.zeros_like(state)
        for coeff, paulis in self.terms:
            s = state
            for wire, p in paulis.items():
                s = apply_matrix(s, PAULI[p], [wire])
            out = out + coeff * s
        return out

    def is_single_pauli(self) -> bool:
        return len(self.terms) == 1 and self.terms[0][0] == 1.0

    def __repr__(self):
        return " + ".join(f"{c:g}*" + "".join(f"{p}{w}" for w, p in sorted(t.items())) for c, t in self.terms)


def Z(wire):
    return Observable(f"Z{wire}")


def X(wire):
    return Observable(f"X{wire}")


def Y(wire):
    return Observable(f"Y{wire}")


# ----------------------------------------------------------------------------- circuit
class Circuit:
    """A quantum circuit with data-dependent (input) and trainable (weight) angles.

    Example
    -------
    >>> qc = Circuit(2)
    >>> x, w = qc.inputs(2), qc.weights(2)
    >>> qc.ry(0, x[0]).ry(1, x[1]).cnot(0, 1).ry(0, w[0]).ry(1, w[1])
    """

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits
        self.ops: list[tuple[str, tuple, object]] = []
        self.n_inputs = 0
        self.n_weights = 0

    # ------------------------------------------------------------ parameters
    def inputs(self, n: int) -> list[InputParam]:
        refs = [InputParam(self.n_inputs + i) for i in range(n)]
        self.n_inputs += n
        return refs

    def weights(self, n: int) -> list[WeightParam]:
        refs = [WeightParam(self.n_weights + i) for i in range(n)]
        self.n_weights += n
        return refs

    # ----------------------------------------------------------------- gates
    def add(self, name: str, wires, param=None):
        name = name.upper()
        wires = tuple(np.atleast_1d(wires).tolist())
        if name not in FIXED_GATES and name not in GENERATORS:
            raise ValueError(f"unknown gate {name}")
        if len(wires) != n_wires(name):
            raise ValueError(f"{name} acts on {n_wires(name)} wire(s), got {wires}")
        if (name in GENERATORS) != (param is not None):
            raise ValueError(f"gate {name}: parameter {'missing' if param is None else 'not allowed'}")
        self.ops.append((name, wires, param))
        return self

    def __getattr__(self, attr):
        name = attr.upper()
        if name in FIXED_GATES:
            return lambda *wires: self.add(name, wires)
        if name in GENERATORS:
            return lambda *args: self.add(name, args[:-1], args[-1])
        raise AttributeError(attr)

    def compose(self, other: "Circuit", input_offset: int | None = None, weight_offset: int | None = None):
        """Append ``other``; its parameters are re-indexed after the current ones."""
        io = self.n_inputs if input_offset is None else input_offset
        wo = self.n_weights if weight_offset is None else weight_offset
        for name, wires, p in other.ops:
            if isinstance(p, InputParam):
                p = InputParam(p.index + io, p.coeff)
            elif isinstance(p, WeightParam):
                p = WeightParam(p.index + wo, p.coeff)
            self.ops.append((name, wires, p))
        self.n_inputs = max(self.n_inputs, io + other.n_inputs)
        self.n_weights = max(self.n_weights, wo + other.n_weights)
        return self

    def inverse(self) -> "Circuit":
        """U^dagger: reversed gate order, negated angles, daggered fixed gates. Same parameter indices."""
        inv = Circuit(self.n_qubits)
        inv.n_inputs, inv.n_weights = self.n_inputs, self.n_weights
        for name, wires, p in reversed(self.ops):
            if p is None:
                inv.ops.append((_DAGGER.get(name, name), wires, None))
            else:
                inv.ops.append((name, wires, -p if not isinstance(p, (int, float)) else -float(p)))
        return inv

    def depth_summary(self) -> str:
        counts = {}
        for name, _, _ in self.ops:
            counts[name] = counts.get(name, 0) + 1
        return f"Circuit({self.n_qubits} qubits, {len(self.ops)} gates, {self.n_inputs} inputs, " \
               f"{self.n_weights} weights) " + str(counts)

    __repr__ = depth_summary

    def draw(self) -> str:
        """Plain-text drawing, one line per qubit."""
        lines = [f"q{q}: " for q in range(self.n_qubits)]
        for name, wires, p in self.ops:
            label = name if p is None else f"{name}({_label(p)})"
            width = len(label) + 2
            for q in range(self.n_qubits):
                if q in wires:
                    tag = label if q == wires[0] else ("X" if name == "CNOT" else ("o" if name in ("CZ", "SWAP") else label))
                    if len(wires) > 1 and q == wires[0] and name.startswith("C"):
                        tag = "●"
                    lines[q] += "─" + tag.center(width - 2, "─") + "─"
                else:
                    lines[q] += "─" * width
        return "\n".join(lines)


def _label(p):
    sign = "" if getattr(p, "coeff", 1.0) == 1.0 else ("-" if p.coeff == -1.0 else f"{p.coeff:g}*")
    if isinstance(p, InputParam):
        return f"{sign}x{p.index}"
    if isinstance(p, WeightParam):
        return f"{sign}w{p.index}"
    return f"{p:.2f}"


def _angle(p, inputs, weights):
    if isinstance(p, InputParam):
        return p.coeff * inputs[:, p.index]
    if isinstance(p, WeightParam):
        return p.coeff * weights[p.index]
    return float(p)


def _unitary(name, p, inputs, weights):
    if p is None:
        return FIXED_GATES[name]
    return param_unitary(name, _angle(p, inputs, weights))


def simulate(circuit: Circuit, inputs=None, weights=None, initial_state=None, batch: int | None = None):
    """Return the final batched state vector, shape (B, 2, ..., 2)."""
    weights = np.zeros(0) if weights is None else np.asarray(weights, dtype=float)
    if inputs is not None:
        inputs = np.asarray(inputs, dtype=float)
        batch = inputs.shape[0]
    if initial_state is not None:
        initial_state = np.asarray(initial_state, dtype=complex)
        batch = initial_state.shape[0]
        state = initial_state.reshape((batch,) + (2,) * circuit.n_qubits)
    else:
        state = zero_state(batch or 1, circuit.n_qubits)
    for name, wires, p in circuit.ops:
        state = apply_matrix(state, _unitary(name, p, inputs, weights), wires)
    return state


def adjoint_gradients(circuit: Circuit, inputs, weights, state, lam):
    """Adjoint differentiation.

    ``lam`` is |lambda> = dL/d<psi| (e.g. sum_j g_j O_j |psi> for expectation values).
    For a gate U = exp(-i theta G):  dL/dtheta = 2 Im <lambda| G |psi>, both taken right after the gate.
    Returns (d_inputs (B, n_inputs), d_weights (n_weights,)).
    """
    weights = np.zeros(0) if weights is None else np.asarray(weights, dtype=float)
    batch = state.shape[0]
    d_in = np.zeros((batch, circuit.n_inputs))
    d_w = np.zeros(circuit.n_weights)
    psi, lam = state, lam
    for name, wires, p in reversed(circuit.ops):
        if p is not None and isinstance(p, (InputParam, WeightParam)):
            d = 2 * p.coeff * braket(lam, apply_matrix(psi, GENERATORS[name], wires)).imag
            if isinstance(p, InputParam):
                d_in[:, p.index] += d
            else:
                d_w[p.index] += d.sum()
        if p is None:
            inv = FIXED_GATES[name].conj().T
        else:
            inv = param_unitary(name, -np.asarray(_angle(p, inputs, weights)))
        psi = apply_matrix(psi, inv, wires)
        lam = apply_matrix(lam, inv, wires)
    return d_in, d_w
