import numpy as np

from skqml.models import QKMedoids, QSpectralClustering, QuantumKernel, adjusted_rand_score
from skqml.text import TextClustering, TfidfVectorizer, cluster_labels, label_f1, make_text_clusters, tokenize
from skqml.timeseries import (DTWKernel, EuclideanKernel, QuantumSeriesKernel, dtw, dtw_matrix,
                              make_time_series_clusters)


def _naive_dtw(a, b, w=None):
    n, m = len(a), len(b)
    w = max(w if w is not None else max(n, m), abs(n - m))
    d = np.full((n + 1, m + 1), np.inf)
    d[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(max(1, i - w), min(m, i + w) + 1):
            d[i, j] = (a[i - 1] - b[j - 1]) ** 2 + min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1])
    return np.sqrt(d[n, m])


def test_dtw_matches_naive_and_is_shift_tolerant():
    rng = np.random.default_rng(0)
    for w in (None, 2, 5):
        a, b = rng.normal(size=15), rng.normal(size=12)
        assert np.isclose(dtw(a, b, w), _naive_dtw(a, b, w))
    x = rng.normal(size=(6, 10))
    m = dtw_matrix(x)
    assert np.allclose(m, m.T) and np.allclose(np.diag(m), 0) and np.isclose(m[1, 4], dtw(x[1], x[4]))
    t = np.linspace(0, 1, 60)
    s1, s2 = np.sin(2 * np.pi * 3 * t), np.sin(2 * np.pi * 3 * (t + 0.05))
    assert dtw(s1, s2) < np.linalg.norm(s1 - s2)


def test_time_series_clustering():
    d = make_time_series_clusters(20, 60)
    eu = adjusted_rand_score(d.y, QSpectralClustering(5, EuclideanKernel(), affinity="knn").fit_predict(d.X))
    dt = adjusted_rand_score(d.y, QSpectralClustering(5, DTWKernel(), affinity="knn").fit_predict(d.X))
    qs = adjusted_rand_score(d.y, QKMedoids(5, QuantumSeriesKernel(4, 0.5)).fit_predict(d.X))
    assert dt > eu + 0.2 and qs > eu + 0.2


def test_text_tools():
    assert tokenize("The Quantum kernels") == ["quantum", "kernels"]
    assert tokenize("文本聚类") == ["文本", "本聚", "聚类"]
    assert label_f1("quantum kernel methods", "quantum kernel") == 0.8
    d = make_text_clusters(30)
    tc = TextClustering(QSpectralClustering(5, QuantumKernel("angle", 1), affinity="knn"), 6, max_df=0.1).fit(d.X)
    assert adjusted_rand_score(d.y, tc.labels_) > 0.9
    labels = tc.labels()
    assert len(labels) == 5 and all(1 <= len(l.split()) <= 3 for l in labels.values())
    x = TfidfVectorizer().fit_transform(d.X)
    assert np.allclose(np.linalg.norm(x, axis=1), 1)
