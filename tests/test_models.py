import numpy as np
import pytest

from skqml import Observable, Tensor
from skqml.models import *
from skqml.models.attention import _QSANNNet
from skqml.models.convolutional import _QCNNNet
from skqml.models.recurrent import QKANLSTMCell, QLSTMCell, QGRUCell, QRNNCell, _QMSMoENet, _SequenceNet
from skqml.nn import cross_entropy, mse_loss

rng = np.random.default_rng(1)


def check_module(net, loss_fn, n_checks=3, eps=1e-6):
    """Compare autograd gradients with finite differences on a few entries of every parameter."""
    net.zero_grad()
    loss_fn().backward()
    for p in net.parameters():
        grad = p.grad.copy()
        for idx in list(np.ndindex(p.shape))[:n_checks]:
            old = p.data[idx]
            p.data[idx] = old + eps
            a = loss_fn().item()
            p.data[idx] = old - eps
            b = loss_fn().item()
            p.data[idx] = old
            num = (a - b) / (2 * eps)
            assert abs(num - grad[idx]) <= 1e-5 * max(1.0, abs(num)), (p.shape, idx, num, grad[idx])


@pytest.mark.parametrize("cell", [QRNNCell, QLSTMCell, QGRUCell])
def test_recurrent_gradients(cell):
    x, y = rng.normal(size=(4, 3, 2)), rng.normal(size=4)
    net = _SequenceNet(cell(2, 3, n_qubits=3, layers=1, rng=rng), 3, 1, rng)
    check_module(net, lambda: mse_loss(net(Tensor(x))[:, 0], y))


def test_qkan_lstm_gradients():
    x, y = rng.normal(size=(4, 3, 1)), rng.normal(size=4)
    net = _SequenceNet(QKANLSTMCell(1, 2, reps=2, rng=rng), 2, 1, rng)
    check_module(net, lambda: mse_loss(net(Tensor(x))[:, 0], y))


@pytest.mark.parametrize("shared", [False, True])
def test_qms_moe_gradients(shared):
    u, c, y = rng.normal(size=(5, 2, 3, 2)), rng.normal(size=(5, 4)), rng.normal(size=5)
    net = _QMSMoENet(2, 2, 3, 4, 3, 2, 4, True, shared, rng)
    model = QMSMoE(shared_mean=shared, mean_l2=0.5)
    check_module(net, lambda: model._nll(net, u, c, y))


def test_qsann_and_qcnn_gradients():
    x, y = rng.normal(size=(4, 3, 2)), np.array([0, 1, 1, 0])
    net = _QSANNNet(2, 2, 3, 1, 1, rng)
    check_module(net, lambda: cross_entropy(net(Tensor(x)), y))
    x8 = rng.normal(size=(4, 4))
    qcnn = _QCNNNet(4, 2, rng)
    check_module(qcnn, lambda: cross_entropy(qcnn(Tensor(x8)), y))


def test_classifiers_learn_simple_problem():
    x = rng.normal(size=(120, 2))
    y = (x[:, 0] > 0).astype(int)
    for model in (VQCClassifier(layers=2, epochs=40, lr=0.05), HybridQNNClassifier(n_qubits=2, epochs=30),
                  QSVC(QuantumKernel("angle", reps=1))):
        assert model.fit(x[:90], y[:90]).score(x[90:], y[90:]) > 0.8, type(model).__name__


def test_vqe_finds_ground_state():
    h = Observable("Z0 Z1") + 0.5 * Observable("X0") + 0.5 * Observable("X1")
    vqe = VQE(h, 2, layers=2).fit(300)
    assert abs(vqe.energy_ - (-np.sqrt(2))) < 1e-3


def test_qaoa_finds_max_cut():
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
    q = QAOAMaxCut(edges, p=2).fit(150)
    assert q.best_cut(1)[0][1] == 4


def test_qcbm_and_autoencoder():
    samples = np.random.default_rng(0).binomial(3, 0.5, 400)
    qcbm = QCBM(2, 2).fit(samples, epochs=150)
    assert qcbm.history_[-1] < 0.1 * qcbm.history_[0]
    data = rng.normal(size=(40, 2)) @ rng.normal(size=(2, 4))
    ae = QuantumAutoencoder(2, 1, 2).fit(data, epochs=100)
    assert ae.compression_fidelity(data).mean() > 0.95


from skqml.models.structured import _ONN, _QTransformer, _TreeNet
from skqml.models.graph import _QGNN
from skqml.quantum.circuit import Observable as _Obs


def test_structured_gradients():
    x, y = rng.normal(size=(5, 3)), np.array([0, 1, 1, 0, 1])
    for net in (_ONN(3, 2, 4, rng), _TreeNet(4, 2, False, rng), _TreeNet(4, 2, True, rng)):
        xi = x if isinstance(net, _ONN) else rng.normal(size=(5, 4))
        check_module(net, lambda: cross_entropy(net(Tensor(xi)), y))
    seq = rng.normal(size=(3, 4, 2))
    qt = _QTransformer(2, 1, 3, rng)
    check_module(qt, lambda: mse_loss(qt(Tensor(seq))[:, 0], np.ones(3)))


def test_orthogonal_layer_is_orthogonal():
    layer = OrthogonalLayer(5, 5, rng)
    w = layer.matrix().data
    assert np.allclose(w @ w.T, np.eye(5), atol=1e-10)


def test_qgnn_gradients():
    a = np.array([[0, 1, 0.5], [1, 0, 1], [0.5, 1, 0]])
    graphs = [(a, np.array([0.1, -0.3, 0.7])), (a[:2, :2], np.array([1.0, 0.2]))]
    net = _QGNN(2, 2, rng)
    check_module(net, lambda: cross_entropy(net(graphs), np.array([0, 1])))


def test_trainable_kernel_and_policy_gradients():
    k = TrainableQuantumKernel(layers=1, epochs=1)
    x, y = rng.normal(size=(6, 2)), np.array([0, 1, 0, 1, 1, 0])
    k.fit(x, y)
    holder = type("H", (), {})()
    from skqml.nn import Module
    class Wrap(Module):
        def __init__(self, w):
            self.w = w
    wrap = Wrap(k.weights_)
    check_module(wrap, lambda: k._kernel_tensor(x[:3], x[3:]).sum())
    assert np.allclose(np.diag(k(x)), 1)
    pol = VQCPolicy(n_states=4, layers=2)
    s = rng.normal(size=(3, 4))
    check_module(pol, lambda: (pol(s) * np.array([[1.0, -2.0]])).sum())


def test_qbm_gradient_and_fit():
    qbm = QuantumBoltzmannMachine(3)
    q = np.random.default_rng(0).dirichlet(np.ones(8))
    theta = rng.normal(0, 0.3, len(qbm.terms))
    _, g = qbm._nll_and_grad(theta, q)
    num = np.zeros_like(theta)
    for i in range(len(theta)):
        e = np.zeros_like(theta)
        e[i] = 1e-6
        num[i] = (qbm._nll_and_grad(theta + e, q)[0] - qbm._nll_and_grad(theta - e, q)[0]) / 2e-6
    assert np.allclose(g, num, atol=1e-6)
    samples = np.random.default_rng(1).choice(8, 400, p=[.3, .05, .05, .1, .1, .05, .05, .3])
    assert QuantumBoltzmannMachine(3, epochs=150).fit(samples).kl_divergence() < 0.01


def _matrix(obs, n):
    basis = np.eye(2 ** n, dtype=complex).reshape((2 ** n,) + (2,) * n)
    return obs.apply(basis).reshape(2 ** n, -1).T


def test_vqd_finds_excited_state():
    h = _Obs("Z0 Z1") + 0.5 * _Obs("X0") + 0.5 * _Obs("X1")
    exact = np.linalg.eigvalsh(_matrix(h, 2))
    vqd = VQD(h, 2, k=2, layers=3).fit(400)
    assert np.allclose(vqd.energies_, exact[:2], atol=1e-3)


def test_qaoa_ising_matches_brute_force():
    j = np.zeros((4, 4))
    j[0, 1] = j[1, 2] = j[2, 3] = 1
    j[0, 3] = -1
    h = [0.5, 0, 0, -0.5]
    q = QAOAIsing(h, j, p=3).fit(250)
    import itertools
    best = min(q.ising_energy(np.array(z)) for z in itertools.product([-1, 1], repeat=4))
    assert q.best_solutions(1)[0][1] == best


from skqml.models.base import clone
from skqml.tuning import (Choice, GridSearchCV, HoldOut, IntUniform, KFold, LogUniform, RandomSearchCV,
                            StratifiedKFold, SuccessiveHalvingSearch, TimeSeriesSplit, cross_val_score,
                            hyperparameters)


def test_params_clone_and_nested():
    svc = QSVC(QuantumKernel("angle", reps=1), c=3.0)
    svc.set_params(kernel__reps=2, c=5.0)
    assert svc.kernel.reps == 2 and svc.c == 5.0
    copy = clone(svc)
    assert copy is not svc and copy.kernel is not svc.kernel and copy.get_params(deep=True)["kernel__reps"] == 2
    assert "layers" in hyperparameters(VQCClassifier) and hyperparameters(QMSMoE)["n_experts"] == 3
    with pytest.raises(ValueError):
        VQCClassifier().set_params(not_a_param=1)


def test_splitters():
    x = np.arange(20)
    for tr, va in TimeSeriesSplit(4, gap=2).split(x):
        assert tr.max() + 2 < va.min()
    y = np.array([0] * 10 + [1] * 10)
    for tr, va in StratifiedKFold(5).split(x, y):
        assert (y[va] == 0).sum() == (y[va] == 1).sum() == 2
    folds = list(KFold(4).split(x))
    assert sorted(np.concatenate([va for _, va in folds]).tolist()) == list(range(20))


def test_searches():
    x = rng.normal(size=(80, 2))
    y = (x[:, 0] > 0).astype(int)
    grid = GridSearchCV(VQCClassifier(epochs=5), {"layers": [1, 2], "lr": [0.01, 0.05]}, cv=3).fit(x, y)
    assert len(grid.cv_results_) == 4 and grid.best_params_["layers"] in (1, 2) and hasattr(grid, "best_estimator_")
    rand = RandomSearchCV(QSVC(), {"c": LogUniform(0.1, 10), "kernel__feature_map": Choice(["angle", "zz"]),
                                   "kernel__reps": IntUniform(1, 2)}, n_iter=4, cv=3).fit(x, y)
    assert rand.best_score_ > 0.7
    halving = SuccessiveHalvingSearch(HybridQNNClassifier(n_qubits=2, epochs=9),
                                      {"lr": LogUniform(0.005, 0.1), "layers": [1, 2]}, n_candidates=6,
                                      min_resource=3, eta=3, cv=HoldOut(0.25)).fit(x, y)
    assert [len(r) for r in halving.rounds_] == [6, 2] and halving.best_params_["epochs"] == 9
    seq = rng.normal(size=(40, 4, 1))
    scores = cross_val_score(LSTMRegressor(epochs=2), seq, seq[:, -1, 0], TimeSeriesSplit(3), "neg_mse")
    assert len(scores) == 3 and np.all(scores <= 0)
