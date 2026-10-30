import numpy as np
import pytest

from skqml import Tensor
from skqml.models import *
from skqml.models.clustering import _VQCluster
from skqml.tuning import ClusterSearch

rng = np.random.default_rng(3)
X = np.concatenate([rng.normal(c, 0.3, (25, 2)) for c in [(-1.5, 0), (1.5, 0), (0, 1.8)]])
Y = np.repeat([0, 1, 2], 25)
K = QuantumKernel("angle", reps=1)


def moons(n=80):
    t = rng.uniform(0, np.pi, n)
    m = np.concatenate([np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.4 - np.sin(t)]])
    return m + rng.normal(0, 0.05, m.shape), np.repeat([0, 1], n)


def test_metrics():
    assert adjusted_rand_score([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(1.0)
    assert normalized_mutual_info_score([0, 0, 1, 1], [5, 5, 7, 7]) == pytest.approx(1.0)
    assert abs(adjusted_rand_score(rng.integers(3, size=3000), rng.integers(3, size=3000))) < 0.02
    assert silhouette_score(X, Y) > 0.6 and silhouette_score(X, rng.integers(3, size=len(Y))) < 0.1
    assert davies_bouldin_score(X, Y) < davies_bouldin_score(X, rng.integers(3, size=len(Y)))


@pytest.mark.parametrize("model", [QKMeans(3, K), QKMedoids(3, K), QAgglomerativeClustering(3, K, "average"),
                                   QAgglomerativeClustering(3, K, "ward"), SwapTestKMeans(3),
                                   SwapTestKMeans(3, shots=2048), QSpectralClustering(3, K, affinity="knn")])
def test_blob_recovery(model):
    assert adjusted_rand_score(Y, model.fit_predict(X)) > 0.95


def test_non_convex_clusters():
    m, ym = moons()
    narrow = QuantumKernel("angle", reps=1, bandwidth=0.5)
    assert adjusted_rand_score(ym, QSpectralClustering(2, narrow, affinity="knn", n_neighbors=5).fit_predict(m)) > 0.95
    assert adjusted_rand_score(ym, QAgglomerativeClustering(2, QuantumKernel("angle", 1, bandwidth=0.3),
                                                            "single").fit_predict(m)) > 0.95
    db = QDBSCAN(eps=0.1, min_samples=4, kernel=QuantumKernel("angle", reps=1, bandwidth=0.3)).fit(m)
    assert db.n_clusters_ == 2 and adjusted_rand_score(ym[db.labels_ >= 0], db.labels_[db.labels_ >= 0]) > 0.95


def test_linkage_matrix_and_predict():
    agg = QAgglomerativeClustering(3, K).fit(X)
    assert agg.linkage_matrix_.shape == (len(X) - 1, 4) and agg.linkage_matrix_[-1, 3] == len(X)
    km = QKMedoids(3, K).fit(X)
    assert adjusted_rand_score(Y, km.predict(X)) > 0.95
    assert adjusted_rand_score(Y, SwapTestKMeans(3).fit(X).predict(X)) > 0.95


def test_variational_clustering_gradients_and_fit():
    net = _VQCluster(2, 2, 2, rng)
    x = rng.normal(size=(5, 2))

    def loss():
        p = net(x, 3)
        return -(p * (p + 1e-9).log()).sum(-1).mean()

    net.zero_grad()
    loss().backward()
    g = net.weights.grad.copy()
    for i in range(4):
        old = net.weights.data[i]
        net.weights.data[i] = old + 1e-6
        a = loss().item()
        net.weights.data[i] = old - 1e-6
        b = loss().item()
        net.weights.data[i] = old
        assert abs((a - b) / 2e-6 - g[i]) < 1e-5
    vqc = VariationalQuantumClustering(3, layers=3, epochs=40).fit(X)
    assert adjusted_rand_score(Y, vqc.labels_) > 0.6 and np.allclose(vqc.predict_proba(X).sum(1), 1)


def test_qaoa_clustering_and_search():
    small = np.concatenate([rng.normal(-1.5, 0.3, (4, 2)), rng.normal(1.5, 0.3, (4, 2))])
    assert adjusted_rand_score(np.repeat([0, 1], 4), QAOAClustering(p=2).fit_predict(small)) == pytest.approx(1.0)
    search = ClusterSearch(QKMeans(kernel=QuantumKernel("angle", 1)), {"n_clusters": [2, 3, 4]}).fit(X)
    assert search.best_params_["n_clusters"] == 3
