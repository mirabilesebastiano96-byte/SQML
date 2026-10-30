"""Differentiable quantum operations that plug circuits into the autograd graph."""
from __future__ import annotations

import numpy as np

from ..autograd import Tensor
from .circuit import Circuit, adjoint_gradients, simulate

__all__ = ["expval", "probs", "amplitudes", "statevector", "parameter_shift"]


def _data(t):
    return None if t is None else (t.data if isinstance(t, Tensor) else np.asarray(t, dtype=float))


def _parents(*ts):
    return tuple(t for t in ts if isinstance(t, Tensor))


def expval(circuit: Circuit, observables, inputs=None, weights=None, initial_state=None,
           shots: int | None = None, rng=None) -> Tensor:
    """Expectation values <psi(x, w)| O_j |psi(x, w)>, shape (B, len(observables)).

    Gradients w.r.t. ``inputs`` and ``weights`` are exact (adjoint method).
    With ``shots`` the forward value is estimated by sampling (no gradient through the noise).
    """
    x, w = _data(inputs), _data(weights)
    state = simulate(circuit, x, w, initial_state, batch=None if x is not None else 1)
    applied = [o.apply(state) for o in observables]
    values = np.stack([np.real(np.sum(np.conj(state) * a, axis=tuple(range(1, state.ndim)))) for a in applied], 1)
    if shots is not None:
        rng = rng or np.random.default_rng()
        if not all(o.is_single_pauli() for o in observables):
            raise ValueError("shot sampling is implemented for single Pauli strings")
        p_plus = np.clip((1 + values) / 2, 0, 1)
        values = 2 * rng.binomial(shots, p_plus) / shots - 1

    def backward(g):
        lam = sum(g[:, j].reshape((-1,) + (1,) * (state.ndim - 1)) * applied[j] for j in range(len(applied)))
        d_in, d_w = adjoint_gradients(circuit, x, w, state, lam)
        if isinstance(inputs, Tensor):
            inputs._accumulate(d_in)
        if isinstance(weights, Tensor):
            weights._accumulate(d_w)

    return Tensor._make(values, _parents(inputs, weights), backward)


def probs(circuit: Circuit, wires=None, inputs=None, weights=None, initial_state=None) -> Tensor:
    """Measurement probabilities of ``wires`` (all wires by default), shape (B, 2**len(wires))."""
    x, w = _data(inputs), _data(weights)
    state = simulate(circuit, x, w, initial_state, batch=None if x is not None else 1)
    n = circuit.n_qubits
    wires = list(range(n)) if wires is None else list(wires)
    full = np.abs(state) ** 2
    others = tuple(q + 1 for q in range(n) if q not in wires)
    marg = full.sum(axis=others) if others else full
    kept = sorted(wires)
    marg = np.moveaxis(marg, [kept.index(q) + 1 for q in wires], list(range(1, len(wires) + 1)))
    out = marg.reshape(marg.shape[0], -1)

    def backward(g):
        g = g.reshape((g.shape[0],) + (2,) * len(wires))
        g = np.moveaxis(g, list(range(1, len(wires) + 1)), [kept.index(q) + 1 for q in wires])
        for ax in others:
            g = np.expand_dims(g, ax)
        lam = np.broadcast_to(g, state.shape) * state
        d_in, d_w = adjoint_gradients(circuit, x, w, state, lam)
        if isinstance(inputs, Tensor):
            inputs._accumulate(d_in)
        if isinstance(weights, Tensor):
            weights._accumulate(d_w)

    return Tensor._make(out, _parents(inputs, weights), backward)


def amplitudes(circuit: Circuit, basis_states, inputs=None, weights=None, initial_state=None) -> Tensor:
    """Real parts of selected amplitudes <k|psi>, shape (B, len(basis_states)).

    Exact for circuits with real amplitudes (e.g. RY / RBS circuits). On hardware these would need tomography.
    """
    x, w = _data(inputs), _data(weights)
    state = simulate(circuit, x, w, initial_state, batch=None if x is not None else 1)
    flat = state.reshape(state.shape[0], -1)
    idx = np.asarray(basis_states, dtype=int)
    out = flat[:, idx].real

    def backward(g):
        lam = np.zeros_like(flat)
        lam[:, idx] = g / 2                       # dRe<k|psi> = Re<k|dpsi>  ->  lambda = g/2 |k>
        d_in, d_w = adjoint_gradients(circuit, x, w, state, lam.reshape(state.shape))
        if isinstance(inputs, Tensor):
            inputs._accumulate(d_in)
        if isinstance(weights, Tensor):
            weights._accumulate(d_w)

    return Tensor._make(out, _parents(inputs, weights), backward)


def statevector(circuit: Circuit, inputs=None, weights=None, initial_state=None) -> np.ndarray:
    """Final states as a plain complex array (B, 2**n). Not differentiable."""
    x, w = _data(inputs), _data(weights)
    state = simulate(circuit, x, w, initial_state, batch=None if x is not None else 1)
    return state.reshape(state.shape[0], -1)


def parameter_shift(circuit: Circuit, observables, inputs=None, weights=None):
    """Weight gradients of sum_j <O_j> with the parameter-shift rule (Pauli-rotation gates only).

    Useful to cross-check the adjoint method, and the rule a real quantum computer would use.
    """
    w = np.asarray(_data(weights), dtype=float)
    grad = np.zeros_like(w)
    for k in range(len(w)):
        plus, minus = w.copy(), w.copy()
        plus[k] += np.pi / 2
        minus[k] -= np.pi / 2
        grad[k] = 0.5 * (expval(circuit, observables, inputs, plus).data.sum()
                         - expval(circuit, observables, inputs, minus).data.sum())
    return grad
