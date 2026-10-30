import numpy as np
import pytest

from skqml import Circuit, Observable, Tensor, concat, expval, parameter_shift, probs, stack, statevector
from skqml.quantum.gates import FIXED_GATES, GENERATORS, param_unitary

EPS = 1e-6
rng = np.random.default_rng(0)


def numeric_grad(f, x):
    g = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        old = x[i]
        x[i] = old + EPS
        a = f()
        x[i] = old - EPS
        b = f()
        x[i] = old
        g[i] = (a - b) / (2 * EPS)
    return g


def test_gates_are_unitary():
    for name, u in FIXED_GATES.items():
        assert np.allclose(u @ u.conj().T, np.eye(len(u))), name
    for name in GENERATORS:
        u = param_unitary(name, 0.731)
        assert np.allclose(u @ u.conj().T, np.eye(len(u))), name
        assert np.allclose(param_unitary(name, 0.731) @ param_unitary(name, -0.731), np.eye(len(u)))


def test_bell_state():
    qc = Circuit(2).h(0).cnot(0, 1)
    assert np.allclose(statevector(qc)[0], [1 / np.sqrt(2), 0, 0, 1 / np.sqrt(2)])


@pytest.mark.parametrize("gate", sorted(GENERATORS))
def test_adjoint_gradient_every_parameterised_gate(gate):
    qc = Circuit(3)
    x, w = qc.inputs(3), qc.weights(2)
    for q in range(3):
        qc.h(q).ry(q, x[q])
    wires = (0,) if gate in ("RX", "RY", "RZ", "PHASE") else (0, 2)
    qc.add(gate, wires, w[0]).cnot(0, 1).add(gate, wires[::-1] if len(wires) == 2 else (1,), w[1]).rx(2, x[0])
    obs = [Observable("Z0"), Observable("X1 Y2") + 0.3 * Observable("Z2")]
    xs = Tensor(rng.normal(size=(3, 3)), requires_grad=True)
    ws = Tensor(rng.normal(size=2), requires_grad=True)
    g = rng.normal(size=(3, 2))
    expval(qc, obs, xs, ws).backward(g)
    f = lambda: (expval(qc, obs, xs.data, ws.data).data * g).sum()
    assert np.allclose(ws.grad, numeric_grad(f, ws.data), atol=1e-6)
    assert np.allclose(xs.grad, numeric_grad(f, xs.data), atol=1e-6)


def test_probs_gradient_and_marginals():
    qc = Circuit(3)
    w = qc.weights(6)
    for q in range(3):
        qc.ry(q, w[q])
    qc.cnot(0, 1).cnot(1, 2)
    for q in range(3):
        qc.rx(q, w[3 + q])
    ws = Tensor(rng.normal(size=6), requires_grad=True)
    p = probs(qc, [2, 0], None, ws)
    assert np.allclose(p.data.sum(), 1)
    g = rng.normal(size=p.shape)
    p.backward(g)
    f = lambda: (probs(qc, [2, 0], None, ws.data).data * g).sum()
    assert np.allclose(ws.grad, numeric_grad(f, ws.data), atol=1e-6)


def test_parameter_shift_matches_adjoint():
    qc = Circuit(2)
    w = qc.weights(4)
    qc.ry(0, w[0]).rx(1, w[1]).cnot(0, 1).rz(0, w[2]).ry(1, w[3])
    obs = [Observable("Z0"), Observable("X1")]
    ws = Tensor(rng.normal(size=4), requires_grad=True)
    expval(qc, obs, None, ws).sum().backward()
    assert np.allclose(parameter_shift(qc, obs, None, ws.data), ws.grad)


def test_autograd_operations():
    a = Tensor(rng.normal(size=(3, 4)), requires_grad=True)
    b = Tensor(rng.normal(size=(4, 2)), requires_grad=True)

    def f():
        out = ((a @ b).tanh().softmax(-1) * concat([a[:, :2].sigmoid(), (a[:, 2:] ** 2).exp()], -1)[:, :2]).sum()
        out = out + a.logsumexp(1).mean() + stack([a.softplus().sum(), b.arctan().sum()]).sum()
        return out + (a.reshape(4, 3).T * 2).relu().sum() + (a / (a.abs() + 1).clip(0, 5)).log_softmax(0).mean()

    f().backward()
    ga, gb = a.grad.copy(), b.grad.copy()
    assert np.allclose(ga, numeric_grad(lambda: f().item(), a.data), atol=1e-5)
    assert np.allclose(gb, numeric_grad(lambda: f().item(), b.data), atol=1e-5)


def test_inverse_circuit_and_signed_parameters():
    qc = Circuit(3)
    x, w = qc.inputs(2), qc.weights(3)
    qc.h(0).ry(0, x[0]).crx(0, 1, w[0]).s(1).t(2).sx(0).rzz(1, 2, -w[1]).ry(2, x[1] * 0.5).rbs(0, 2, w[2])
    full = Circuit(3).compose(qc).compose(qc.inverse(), 0, 0)
    xs, ws = rng.normal(size=(4, 2)), rng.normal(size=3)
    assert np.allclose(np.abs(statevector(full, xs, ws)[:, 0]), 1)
    xt, wt = Tensor(xs, requires_grad=True), Tensor(ws, requires_grad=True)
    obs = [Observable("Z0 X2")]
    expval(qc, obs, xt, wt).sum().backward()
    f = lambda: expval(qc, obs, xt.data, wt.data).data.sum()
    assert np.allclose(wt.grad, numeric_grad(f, wt.data), atol=1e-6)
    assert np.allclose(xt.grad, numeric_grad(f, xt.data), atol=1e-6)


def test_amplitudes_gradient():
    from skqml import amplitudes
    qc = Circuit(3)
    w = qc.weights(3)
    qc.x(0).rbs(0, 1, w[0]).rbs(1, 2, w[1]).rbs(0, 1, w[2])
    wt = Tensor(rng.normal(size=3), requires_grad=True)
    g = rng.normal(size=(1, 3))
    amplitudes(qc, [4, 2, 1], None, wt).backward(g)
    f = lambda: (amplitudes(qc, [4, 2, 1], None, wt.data).data * g).sum()
    assert np.allclose(wt.grad, numeric_grad(f, wt.data), atol=1e-6)
