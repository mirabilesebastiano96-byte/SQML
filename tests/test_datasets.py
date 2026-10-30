import numpy as np

from skqml.datasets import *
from skqml.models import QKMeans, QuantumKernel, adjusted_rand_score


def test_real_datasets_shapes():
    assert load_iris().X.shape == (150, 4) and len(set(load_iris().y)) == 3
    assert load_wine().X.shape == (178, 13)
    assert load_breast_cancer().X.shape == (569, 30)
    d = load_digits(3)
    assert d.X.shape[1] == 64 and set(d.y) == {0, 1, 2}
    x, y = load_iris()
    assert len(x) == len(y)


def test_generators():
    for name, d in make_clustering_suite(120).items():
        assert d.X.shape == (120, 2) and len(d.y) == 120, name
    assert pca(load_wine().X, 4).shape == (178, 4)


def test_quantum_datasets():
    ising = make_ising_phases(4, 40)
    assert ising.X.shape == (40, 3)
    ferro, para = ising.X[ising.y == 0], ising.X[ising.y == 1]
    assert ferro[:, 0].mean() > para[:, 0].mean() and para[:, 1].mean() > ferro[:, 1].mean()
    lab = QKMeans(2, QuantumKernel("angle", 1, bandwidth=0.5)).fit_predict(ising.X)
    assert adjusted_rand_score(ising.y, lab) > 0.8
    s = make_quantum_state_clusters(3, 2, 30)
    assert np.allclose(np.linalg.norm(s.X, axis=1), 1)


def test_earthquakes_snapshot():
    from skqml.datasets import load_earthquakes
    eq = load_earthquakes()
    assert eq.X.shape == (743, 5) and np.allclose(np.linalg.norm(eq.X[:, :3], axis=1), 1)
    assert eq.meta["magnitude"].max() == 7.7 and eq.target_names[-1] == "other"
