"""Quantum clustering.

Kernel-based (any quantum kernel; the Hilbert-space distance is d(x,y)^2 = k(x,x) + k(y,y) - 2 k(x,y)):
* QKMeans (kernel k-means), QKMedoids, QSpectralClustering (Ng, Jordan & Weiss 2002 on a quantum affinity),
  QAgglomerativeClustering (single / complete / average / ward), QDBSCAN (Ester et al. 1996),
  QMeanShift (kernel mean shift in the projected feature space).
Circuit-based:
* SwapTestKMeans (Lloyd, Mohseni & Rebentrost 2013; Khan et al. 2019): distances from swap-test circuits,
  optionally estimated with a finite number of shots.
* VariationalQuantumClustering: a circuit maps x to cluster probabilities p(c|x) (Born rule) trained to maximise
  the mutual information I(X; C) = H(mean_x p) - mean_x H(p) (information-maximising clustering).
* QAOAClustering (Otterbach et al. 2017): 2-way clustering as weighted Max-Cut solved with QAOA.
Metrics: silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score.
"""
from __future__ import annotations

import itertools

import numpy as np

from ..autograd import Tensor, no_grad
from ..nn import Adam, Module, Parameter
from ..quantum.circuit import Circuit, simulate
from ..quantum.functions import probs
from ..quantum.templates import amplitude_encode, angle_encoding, strongly_entangling
from .base import ParamsMixin, Scaler
from .kernels import ProjectedQuantumKernel, QKMeans, QuantumKernel

__all__ = ["QKMeans", "QKMedoids", "QSpectralClustering", "QAgglomerativeClustering", "QDBSCAN", "QMeanShift",
           "SwapTestKMeans", "VariationalQuantumClustering", "QAOAClustering", "kernel_distances",
           "silhouette_score", "davies_bouldin_score", "adjusted_rand_score", "normalized_mutual_info_score"]


def kernel_distances(k: np.ndarray, diag_a=None, diag_b=None) -> np.ndarray:
    """Hilbert-space distances from a kernel matrix."""
    da = np.diag(k) if diag_a is None else diag_a
    db = np.diag(k) if diag_b is None else diag_b
    return np.sqrt(np.maximum(da[:, None] + db[None, :] - 2 * k, 0))


class _ClusterBase(ParamsMixin):
    def fit_predict(self, x):
        return self.fit(x).labels_

    def _scale(self, x, fit=False):
        x = np.asarray(x, dtype=float)
        if getattr(getattr(self, "kernel", None), "handles_scaling", False):
            return x                      # sequence kernels normalise each series themselves
        if fit:
            self.scaler_ = Scaler().fit(x)
        return self.scaler_.transform(x)

    def _kernel_fit(self, x):
        self.x_ = self._scale(x, fit=True)
        self.k_ = self.kernel(self.x_)
        self.d_ = kernel_distances(self.k_)
        return self.d_


# ============================================================================== kernel methods
class QKMedoids(_ClusterBase):
    """k-medoids (PAM-style alternation): cluster centres are actual data points, robust to outliers."""

    def __init__(self, n_clusters=2, kernel=None, max_iter=100, n_init=5, seed=0):
        self.n_clusters, self.kernel, self.max_iter, self.n_init, self.seed = n_clusters, kernel or QuantumKernel(), \
            max_iter, n_init, seed

    def fit(self, x):
        d = self._kernel_fit(x)
        rng = np.random.default_rng(self.seed)
        best = (np.inf, None, None)
        for _ in range(self.n_init):
            medoids = rng.choice(len(d), self.n_clusters, replace=False)
            for _ in range(self.max_iter):
                labels = np.argmin(d[:, medoids], 1)
                new = np.array([np.where(labels == c)[0][np.argmin(d[np.ix_(labels == c, labels == c)].sum(1))]
                                if np.any(labels == c) else medoids[c] for c in range(self.n_clusters)])
                if np.array_equal(np.sort(new), np.sort(medoids)):
                    break
                medoids = new
            cost = d[np.arange(len(d)), medoids[labels]].sum()
            if cost < best[0]:
                best = (cost, labels, medoids)
        self.inertia_, self.labels_, self.medoid_indices_ = best
        return self

    def predict(self, x):
        k = self.kernel(self._scale(x), self.x_[self.medoid_indices_])
        return np.argmin(kernel_distances(k, np.ones(len(k)), np.ones(k.shape[1])), 1)


class QSpectralClustering(_ClusterBase):
    """Spectral clustering. ``affinity="kernel"`` uses the quantum kernel matrix directly; ``affinity="knn"``
    connects every point to its ``n_neighbors`` nearest points in Hilbert-space distance (better for curved,
    non-convex clusters)."""

    def __init__(self, n_clusters=2, kernel=None, affinity="kernel", n_neighbors=10, n_init=10, seed=0):
        self.n_clusters, self.kernel, self.n_init, self.seed = n_clusters, kernel or QuantumKernel(), n_init, seed
        self.affinity, self.n_neighbors = affinity, n_neighbors

    def fit(self, x):
        self.x_ = self._scale(x, fit=True)
        k = self.kernel(self.x_)
        if self.affinity == "knn":
            d = kernel_distances(k)
            near = np.argsort(d, 1)[:, 1:self.n_neighbors + 1]
            a = np.zeros_like(d)
            a[np.repeat(np.arange(len(d)), self.n_neighbors), near.ravel()] = 1
            a = np.maximum(a, a.T)
        else:
            a = k.copy()
        np.fill_diagonal(a, 0)
        deg = a.sum(1) + 1e-12
        lap = a / np.sqrt(np.outer(deg, deg))                          # D^-1/2 A D^-1/2
        vals, vecs = np.linalg.eigh(lap)
        emb = vecs[:, -self.n_clusters:]
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        self.embedding_, self.eigenvalues_ = emb, vals[-self.n_clusters:]
        self.labels_ = _kmeans(emb, self.n_clusters, self.n_init, self.seed)[0]
        return self


def _kmeans(x, k, n_init=10, seed=0, max_iter=200):
    rng = np.random.default_rng(seed)
    best = (None, None, np.inf)
    for _ in range(n_init):
        centres = x[[rng.integers(len(x))]]
        for _ in range(1, k):                                          # k-means++ seeding
            d2 = ((x[:, None] - centres[None]) ** 2).sum(-1).min(1)
            centres = np.vstack([centres, x[rng.choice(len(x), p=d2 / d2.sum())]])
        for _ in range(max_iter):
            labels = ((x[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
            new = np.array([x[labels == c].mean(0) if np.any(labels == c) else centres[c] for c in range(k)])
            if np.allclose(new, centres):
                break
            centres = new
        inertia = ((x - centres[labels]) ** 2).sum()
        if inertia < best[2]:
            best = (labels, centres, inertia)
    return best


class QAgglomerativeClustering(_ClusterBase):
    """Bottom-up hierarchical clustering on Hilbert-space distances.

    Pure-Python merging costs O(n^3): comfortable up to a few hundred points.
    ``linkage``: "single", "complete", "average" or "ward". ``linkage_matrix_`` has the SciPy layout
    [idx_a, idx_b, distance, size], so it can be drawn with ``scipy.cluster.hierarchy.dendrogram``.
    """

    def __init__(self, n_clusters=2, kernel=None, linkage="average"):
        self.n_clusters, self.kernel, self.linkage = n_clusters, kernel or QuantumKernel(), linkage

    def fit(self, x):
        d = self._kernel_fit(x)
        n = len(d)
        clusters = {i: [i] for i in range(n)}
        dist = {(i, j): (d[i, j] ** 2 / 2 if self.linkage == "ward" else d[i, j]) for i in range(n) for j in range(i + 1, n)}
        next_id, merges = n, []
        labels_at = None
        while len(clusters) > 1:
            (a, b), dab = min(dist.items(), key=lambda kv: kv[1])
            merged = clusters.pop(a) + clusters.pop(b)
            merges.append([a, b, float(np.sqrt(2 * dab) if self.linkage == "ward" else dab), len(merged)])
            for c in list(clusters):
                dist[(min(c, next_id), max(c, next_id))] = self._link(clusters[c], merged, d, dist.get(
                    (min(a, c), max(a, c))), dist.get((min(b, c), max(b, c))), len(clusters[c]), a, b)
            dist = {k: v for k, v in dist.items() if a not in k and b not in k}
            clusters[next_id] = merged
            next_id += 1
            if len(clusters) == self.n_clusters:
                labels_at = np.empty(n, dtype=int)
                for lab, members in enumerate(clusters.values()):
                    labels_at[members] = lab
        self.linkage_matrix_ = np.array(merges)
        self.labels_ = labels_at if labels_at is not None else np.zeros(n, dtype=int)
        return self

    def _link(self, c, merged, d, dac, dbc, nc, a, b):
        block = d[np.ix_(c, merged)]
        if self.linkage == "single":
            return block.min()
        if self.linkage == "complete":
            return block.max()
        if self.linkage == "average":
            return block.mean()
        # ward: increase in within-cluster sum of squares (Lance-Williams on squared distances)
        cm, cc = np.array(merged), np.array(c)
        both = np.concatenate([cm, cc])
        ss = lambda idx: (d[np.ix_(idx, idx)] ** 2).sum() / (2 * len(idx))
        return ss(both) - ss(cm) - ss(cc)


class QDBSCAN(_ClusterBase):
    """Density-based clustering: points with >= ``min_samples`` neighbours within ``eps`` (Hilbert-space distance)
    are core points; label -1 marks noise. Fidelity-kernel distances lie in [0, sqrt(2)]."""

    def __init__(self, eps=0.5, min_samples=5, kernel=None):
        self.eps, self.min_samples, self.kernel = eps, min_samples, kernel or QuantumKernel()

    def fit(self, x):
        d = self._kernel_fit(x)
        n = len(d)
        neigh = [np.where(d[i] <= self.eps)[0] for i in range(n)]
        core = np.array([len(nb) >= self.min_samples for nb in neigh])
        labels = np.full(n, -1)
        cluster = 0
        for i in range(n):
            if labels[i] != -1 or not core[i]:
                continue
            labels[i], stack = cluster, [i]
            while stack:
                j = stack.pop()
                if not core[j]:
                    continue
                for nb in neigh[j]:
                    if labels[nb] == -1:
                        labels[nb] = cluster
                        stack.append(nb)
            cluster += 1
        self.labels_, self.core_sample_indices_ = labels, np.where(core)[0]
        self.n_clusters_ = cluster
        return self


class QMeanShift(_ClusterBase):
    """Mean shift in the feature space of a projected quantum kernel (Bloch-vector features), Gaussian bandwidth."""

    def __init__(self, bandwidth=0.8, kernel=None, max_iter=100, merge_tol=None):
        self.bandwidth, self.kernel, self.max_iter, self.merge_tol = bandwidth, kernel or ProjectedQuantumKernel(), \
            max_iter, merge_tol

    def fit(self, x):
        self.x_ = self._scale(x, fit=True)
        f = self.kernel.features(self.x_)
        modes = f.copy()
        for _ in range(self.max_iter):
            w = np.exp(-((modes[:, None] - f[None]) ** 2).sum(-1) / (2 * self.bandwidth ** 2))
            new = w @ f / w.sum(1, keepdims=True)
            if np.allclose(new, modes, atol=1e-6):
                break
            modes = new
        tol = self.merge_tol or self.bandwidth / 2
        centres, labels = [], np.empty(len(f), dtype=int)
        for i, m in enumerate(modes):
            for c, centre in enumerate(centres):
                if np.linalg.norm(m - centre) < tol:
                    labels[i] = c
                    break
            else:
                centres.append(m)
                labels[i] = len(centres) - 1
        self.cluster_centers_, self.labels_, self.features_ = np.array(centres), labels, f
        return self

    def predict(self, x):
        f = self.kernel.features(self._scale(x))
        return ((f[:, None] - self.cluster_centers_[None]) ** 2).sum(-1).argmin(1)


# ============================================================================== circuit methods
def _swap_test_circuit(n):
    """ancilla 0, register A = 1..n, register B = n+1..2n.  P(ancilla = 0) = (1 + |<a|b>|^2) / 2."""
    qc = Circuit(1 + 2 * n)
    qc.h(0)
    for q in range(n):
        qc.cswap(0, 1 + q, 1 + n + q)
    qc.h(0)
    return qc


class SwapTestKMeans(_ClusterBase):
    """k-means whose distances are measured with swap-test circuits on amplitude-encoded vectors.

    Features are min-max scaled to [0, 1] and a constant is appended, so all inner products are positive and the
    swap test (which returns |<x|c>|^2) determines the cosine similarity. With ``shots`` the overlap is sampled.
    """

    def __init__(self, n_clusters=2, shots=None, max_iter=50, n_init=3, seed=0):
        self.n_clusters, self.shots, self.max_iter, self.n_init, self.seed = n_clusters, shots, max_iter, n_init, seed

    def _prepare(self, x, fit=False):
        x = np.asarray(x, dtype=float)
        if fit:
            self.lo_, self.hi_ = x.min(0), x.max(0)
        z = (x - self.lo_) / (self.hi_ - self.lo_ + 1e-12)
        return np.concatenate([z, np.full((len(z), 1), 0.5)], 1)

    def _overlaps(self, a, b, rng):
        n = int(np.ceil(np.log2(a.shape[1])))
        sa, sb = amplitude_encode(a, n), amplitude_encode(b, n)
        qc = _swap_test_circuit(n)
        pairs = np.einsum("ia,jb->ijab", sa, sb).reshape(len(a) * len(b), -1)
        ancilla0 = np.zeros((len(pairs), 2 * 2 ** (2 * n)), dtype=complex)
        ancilla0[:, :2 ** (2 * n)] = pairs                          # |0>_anc |a>|b>
        out = simulate(qc, None, None, ancilla0).reshape(len(pairs), 2, -1)
        p0 = (np.abs(out[:, 0]) ** 2).sum(1)
        if self.shots:
            p0 = rng.binomial(self.shots, np.clip(p0, 0, 1)) / self.shots
        return np.clip(2 * p0 - 1, 0, 1).reshape(len(a), len(b))      # |<a|b>|^2

    def _distances(self, a, b, rng):
        cos = np.sqrt(self._overlaps(a, b, rng))
        na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        return np.maximum(na[:, None] ** 2 + nb[None] ** 2 - 2 * np.outer(na, nb) * cos, 0)

    def fit(self, x):
        z = self._prepare(x, fit=True)
        rng = np.random.default_rng(self.seed)
        best = (np.inf, None, None)
        for _ in range(self.n_init):
            centres = z[rng.choice(len(z), self.n_clusters, replace=False)]
            for _ in range(self.max_iter):
                labels = self._distances(z, centres, rng).argmin(1)
                new = np.array([z[labels == c].mean(0) if np.any(labels == c) else centres[c]
                                for c in range(self.n_clusters)])
                if np.allclose(new, centres):
                    break
                centres = new
            inertia = ((z - centres[labels]) ** 2).sum()
            if inertia < best[0]:
                best = (inertia, labels, centres)
        self.inertia_, self.labels_, self.cluster_centers_ = best
        return self

    def predict(self, x):
        return self._distances(self._prepare(x), self.cluster_centers_, np.random.default_rng(self.seed)).argmin(1)


class _VQCluster(Module):
    def __init__(self, n_features, n_readout, layers, rng):
        self.circuit = Circuit(n_features)
        x = self.circuit.inputs(n_features)
        for _ in range(layers):                                        # data re-uploading
            angle_encoding(self.circuit, x, "Y")
            strongly_entangling(self.circuit, 1)
        self.readout = list(range(n_readout))
        self.weights = Parameter(rng.uniform(-np.pi, np.pi, self.circuit.n_weights))

    def forward(self, x, k):
        p = probs(self.circuit, self.readout, Tensor(np.arctan(x) * 2), self.weights)[:, :k]
        return p / p.sum(-1, keepdims=True)


class VariationalQuantumClustering(_ClusterBase):
    """p(c|x) = Born-rule probabilities of the readout qubits of a re-uploading circuit, trained to maximise
    I(X;C) = H(E_x[p(c|x)]) - E_x[H(p(c|x))]: confident assignments (low conditional entropy) and balanced
    clusters (high marginal entropy). ``balance`` weighs the marginal-entropy term."""

    def __init__(self, n_clusters=2, layers=3, epochs=60, lr=0.05, batch_size=64, balance=1.0, seed=0):
        self.n_clusters, self.layers, self.epochs, self.lr = n_clusters, layers, epochs, lr
        self.batch_size, self.balance, self.seed = batch_size, balance, seed

    def fit(self, x):
        xs = self._scale(x, fit=True)
        n_readout = max(1, int(np.ceil(np.log2(self.n_clusters))))
        if n_readout > xs.shape[1]:
            raise ValueError("need at least log2(n_clusters) features (one qubit per feature)")
        rng = np.random.default_rng(self.seed)
        self.net_ = _VQCluster(xs.shape[1], n_readout, self.layers, rng)
        opt = Adam(self.net_.parameters(), self.lr)
        self.history_ = []
        for _ in range(self.epochs):
            order = rng.permutation(len(xs))
            for i in range(0, len(xs), self.batch_size):
                idx = order[i:i + self.batch_size]
                self.net_.zero_grad()
                p = self.net_(xs[idx], self.n_clusters)
                cond = -(p * (p + 1e-9).log()).sum(-1).mean()
                marg = p.mean(0)
                ent = -(marg * (marg + 1e-9).log()).sum()
                loss = cond - self.balance * ent
                loss.backward()
                opt.step()
            self.history_.append(-loss.item())
        self.labels_ = self.predict(x)
        return self

    def predict_proba(self, x):
        with no_grad():
            return self.net_(self._scale(x), self.n_clusters).data

    def predict(self, x):
        return self.predict_proba(x).argmax(1)


class QAOAClustering(_ClusterBase):
    """Two clusters as weighted Max-Cut: points are qubits, edge weights are distances, and QAOA looks for the
    cut that separates distant points. One qubit per point, so it suits small data sets (<= ~14 points)."""

    def __init__(self, p=2, steps=200, lr=0.05, kernel=None, seed=0):
        self.p, self.steps, self.lr, self.kernel, self.seed = p, steps, lr, kernel, seed

    def fit(self, x):
        from .optimization import QAOAIsing
        xs = self._scale(x, fit=True)
        if self.kernel is None:
            d = np.sqrt(((xs[:, None] - xs[None]) ** 2).sum(-1))
        else:
            d = kernel_distances(self.kernel(xs))
        w = d / d.max()
        j = np.triu(w, 1)                     # minimise sum w_ij z_i z_j  <=>  cut the heaviest edges
        self.qaoa_ = QAOAIsing(np.zeros(len(xs)), j, p=self.p, seed=self.seed).fit(self.steps, self.lr)
        z, energy, prob = self.qaoa_.best_solutions(1)[0]
        self.labels_ = (np.array(z) < 0).astype(int)
        self.cut_energy_, self.solution_probability_ = energy, prob
        return self


QKMeans.fit_predict = _ClusterBase.fit_predict


# ============================================================================== metrics
def silhouette_score(x=None, labels=None, distances=None):
    """Mean silhouette in [-1, 1] (higher is better). Pass ``distances`` to use Hilbert-space distances."""
    d = distances if distances is not None else np.sqrt(((x[:, None] - x[None]) ** 2).sum(-1))
    labels = np.asarray(labels)
    valid = labels >= 0
    d, labels = d[np.ix_(valid, valid)], labels[valid]
    clusters = np.unique(labels)
    if len(clusters) < 2:
        return 0.0
    s = np.zeros(len(labels))
    for i in range(len(labels)):
        own = labels == labels[i]
        if own.sum() <= 1:
            continue
        a = d[i, own].sum() / (own.sum() - 1)
        b = min(d[i, labels == c].mean() for c in clusters if c != labels[i])
        s[i] = (b - a) / max(a, b)
    return float(s.mean())


def davies_bouldin_score(x, labels):
    """Lower is better."""
    x, labels = np.asarray(x, dtype=float), np.asarray(labels)
    clusters = [c for c in np.unique(labels) if c >= 0]
    cent = np.array([x[labels == c].mean(0) for c in clusters])
    scat = np.array([np.linalg.norm(x[labels == c] - cent[i], axis=1).mean() for i, c in enumerate(clusters)])
    worst = [max((scat[i] + scat[j]) / np.linalg.norm(cent[i] - cent[j]) for j in range(len(clusters)) if j != i)
             for i in range(len(clusters))]
    return float(np.mean(worst))


def _contingency(a, b):
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    table = np.zeros((len(ua), len(ub)))
    np.add.at(table, (ia, ib), 1)
    return table


def adjusted_rand_score(labels_true, labels_pred):
    """Agreement between two partitions, corrected for chance: 1 = identical, ~0 = random."""
    t = _contingency(labels_true, labels_pred)
    comb = lambda v: v * (v - 1) / 2
    index, a, b = comb(t).sum(), comb(t.sum(1)).sum(), comb(t.sum(0)).sum()
    expected = a * b / comb(t.sum())
    maximum = (a + b) / 2
    return float((index - expected) / (maximum - expected)) if maximum != expected else 1.0


def normalized_mutual_info_score(labels_true, labels_pred):
    t = _contingency(labels_true, labels_pred) / len(labels_true)
    pa, pb = t.sum(1), t.sum(0)
    nz = t > 0
    mi = (t[nz] * np.log(t[nz] / np.outer(pa, pb)[nz])).sum()
    ha, hb = -(pa * np.log(pa)).sum(), -(pb * np.log(pb)).sum()
    return float(mi / np.sqrt(ha * hb)) if ha > 0 and hb > 0 else 1.0
