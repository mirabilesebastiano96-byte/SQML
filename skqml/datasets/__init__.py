"""Datasets for clustering, classification and quantum experiments.

Real data (bundled, no download):  load_iris, load_wine, load_breast_cancer, load_digits
Recent text:                       load_arxiv_recent (500 arXiv abstracts, Oct 2026), fetch_arxiv (live)
Recent real-world data:            load_earthquakes (USGS feed: snapshot of the last month, or live with refresh=True)
Synthetic clustering benchmarks:   make_blobs, make_moons, make_circles, make_anisotropic, make_varied_density,
                                   make_spirals, make_clustering_suite
Quantum data:                      make_ising_phases (ground states of the transverse-field Ising chain),
                                   make_quantum_state_clusters (noisy copies of random reference states)
Helpers:                           pca (reduce features to the number of qubits), minmax

Every loader returns a :class:`Dataset` with ``X``, ``y``, ``feature_names``, ``target_names`` and ``description``.
Sources and licences of the real data are listed in ``DATASETS.md``.
"""
from __future__ import annotations

import csv
import gzip
import os
from dataclasses import dataclass, field

import numpy as np

__all__ = ["Dataset", "load_iris", "load_wine", "load_breast_cancer", "load_digits", "make_blobs", "make_moons",
           "make_circles", "make_anisotropic", "make_varied_density", "make_spirals", "make_clustering_suite",
           "make_ising_phases", "make_quantum_state_clusters", "pca", "minmax", "list_datasets", "load_earthquakes", "load_arxiv_recent", "fetch_arxiv"]

_DATA = os.path.join(os.path.dirname(__file__), "data")


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: list = field(default_factory=list)
    target_names: list = field(default_factory=list)
    description: str = ""

    def __iter__(self):                       # allows  X, y = load_iris()
        return iter((self.X, self.y))

    def __repr__(self):
        kind = f"{self.X.shape[1]} features" if np.ndim(self.X) > 1 else "texts"
        return f"Dataset({len(self.X)} samples, {kind}, {len(self.target_names)} classes)"

    def to_frame(self):
        import pandas as pd
        df = pd.DataFrame(self.X, columns=self.feature_names)
        df["target"] = self.y
        return df


def _read(name):
    path = os.path.join(_DATA, name)
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "rt") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    x = np.array([[float(v) for v in r[:-1]] for r in body])
    labels = [r[-1] for r in body]
    names = sorted(set(labels), key=lambda s: (len(s), s)) if all(l.isdigit() for l in labels) else \
        list(dict.fromkeys(labels))
    y = np.array([names.index(l) for l in labels])
    return x, y, header[:-1], names


def load_iris():
    """150 iris flowers, 4 measurements (cm), 3 species. Fisher (1936). Classic for 3-cluster problems."""
    x, y, f, t = _read("iris.csv")
    return Dataset(x, y, f, t, "Iris (Fisher 1936): sepal/petal length and width of 3 species, 50 each.")


def load_wine():
    """178 wines from 3 cultivars, 13 chemical measurements. Aeberhard & Forina (UCI)."""
    x, y, f, t = _read("wine.csv")
    return Dataset(x, y, f, t, "Wine (UCI): chemical analysis of wines from three cultivars in Italy.")


def load_breast_cancer():
    """569 tumours, 30 cell-nucleus features, malignant / benign. Wolberg et al. (UCI, Wisconsin Diagnostic)."""
    x, y, f, t = _read("breast_cancer.csv.gz")
    return Dataset(x, y, f, t, "Breast Cancer Wisconsin (Diagnostic), UCI.")


def load_digits(n_classes: int = 10):
    """1797 handwritten digits as 8x8 images (pixel values 0-16). Alpaydin & Kaynak (UCI)."""
    x, y, f, t = _read("digits.csv.gz")
    keep = y < n_classes
    return Dataset(x[keep], y[keep], f, t[:n_classes], "Optical recognition of handwritten digits (UCI), 8x8.")


# ============================================================================ synthetic
def make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=0.5, spread=4.0, seed=0):
    """Isotropic Gaussian clusters. ``centers`` = number of clusters or an array of centres."""
    rng = np.random.default_rng(seed)
    centers = np.asarray(centers, dtype=float) if np.ndim(centers) else rng.uniform(-spread, spread, (centers, n_features))
    k = len(centers)
    sizes = np.full(k, n_samples // k)
    sizes[: n_samples % k] += 1
    std = np.broadcast_to(np.asarray(cluster_std, dtype=float), (k,))
    x = np.concatenate([rng.normal(c, s, (m, centers.shape[1])) for c, s, m in zip(centers, std, sizes)])
    y = np.repeat(np.arange(k), sizes)
    return Dataset(x, y, [f"x{i}" for i in range(x.shape[1])], list(range(k)), "Gaussian blobs")


def make_moons(n_samples=300, noise=0.06, seed=0):
    """Two interleaved half circles: non-convex, defeats centroid-based clustering."""
    rng = np.random.default_rng(seed)
    n1 = n_samples // 2
    t1, t2 = rng.uniform(0, np.pi, n1), rng.uniform(0, np.pi, n_samples - n1)
    x = np.concatenate([np.c_[np.cos(t1), np.sin(t1)], np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]])
    x += rng.normal(0, noise, x.shape)
    return Dataset(x, np.repeat([0, 1], [n1, n_samples - n1]), ["x0", "x1"], [0, 1], "Two moons")


def make_circles(n_samples=300, factor=0.45, noise=0.05, seed=0):
    """Two concentric rings (inner radius = ``factor``)."""
    rng = np.random.default_rng(seed)
    n1 = n_samples // 2
    t = rng.uniform(0, 2 * np.pi, n_samples)
    r = np.where(np.arange(n_samples) < n1, 1.0, factor)
    x = np.c_[r * np.cos(t), r * np.sin(t)] + rng.normal(0, noise, (n_samples, 2))
    return Dataset(x, (np.arange(n_samples) >= n1).astype(int), ["x0", "x1"], [0, 1], "Concentric circles")


def make_anisotropic(n_samples=300, seed=0):
    """Three elongated (sheared) Gaussian clusters."""
    d = make_blobs(n_samples, centers=[[-2, 0], [2, 0.5], [0, 3]], cluster_std=0.6, seed=seed)
    d.X = d.X @ np.array([[0.6, -0.6], [-0.4, 0.8]])
    d.description = "Anisotropic blobs"
    return d


def make_varied_density(n_samples=300, seed=0):
    """Three clusters with very different spreads (0.2, 0.8, 0.4)."""
    d = make_blobs(n_samples, centers=[[-3, 0], [1, 0], [5, 2]], cluster_std=[0.2, 0.8, 0.4], seed=seed)
    d.description = "Blobs with varied density"
    return d


def make_spirals(n_samples=300, turns=1.5, noise=0.08, seed=0):
    """Two intertwined spirals: the hardest of the 2-D benchmarks."""
    rng = np.random.default_rng(seed)
    n1 = n_samples // 2
    t = np.sqrt(rng.uniform(0, 1, n_samples)) * turns * 2 * np.pi
    sign = np.where(np.arange(n_samples) < n1, 1.0, -1.0)
    x = np.c_[sign * t * np.cos(t), sign * t * np.sin(t)] / (turns * 2 * np.pi)
    x += rng.normal(0, noise, x.shape) * 0.5
    return Dataset(x, (sign < 0).astype(int), ["x0", "x1"], [0, 1], "Two spirals")


def make_clustering_suite(n_samples=200, seed=0) -> dict:
    """The six 2-D benchmarks in one dict, handy for a comparison grid."""
    return {"blobs": make_blobs(n_samples, seed=seed), "moons": make_moons(n_samples, seed=seed),
            "circles": make_circles(n_samples, seed=seed), "anisotropic": make_anisotropic(n_samples, seed=seed),
            "varied density": make_varied_density(n_samples, seed=seed), "spirals": make_spirals(n_samples, seed=seed)}


# ============================================================================ quantum data
def make_ising_phases(n_qubits=6, n_samples=120, g_range=(0.2, 2.0), seed=0):
    """Ground states of the transverse-field Ising chain  H = -sum Z_i Z_{i+1} - g sum X_i  (periodic).

    For g < 1 the chain is ferromagnetic (ordered), for g > 1 paramagnetic: a quantum phase transition at g = 1.
    Features are measurable quantities of each ground state: <Z_i Z_{i+1}>, <X_i> and the connected
    correlation <Z_0 Z_{n/2}>. ``y`` is the phase (0 = ferromagnetic, 1 = paramagnetic), ``g`` is in ``description``.
    A natural test for clustering *quantum* data.
    """
    from ..quantum.circuit import Observable
    rng = np.random.default_rng(seed)
    g = np.sort(rng.uniform(*g_range, n_samples))
    n = n_qubits
    dim = 2 ** n
    basis = np.eye(dim, dtype=complex).reshape((dim,) + (2,) * n)

    def matrix(obs):
        return obs.apply(basis).reshape(dim, -1).T

    zz = sum((matrix(Observable(f"Z{i} Z{(i + 1) % n}")) for i in range(n)), np.zeros((dim, dim)))
    xx = sum((matrix(Observable(f"X{i}")) for i in range(n)), np.zeros((dim, dim)))
    corr = matrix(Observable(f"Z0 Z{n // 2}"))
    feats = []
    for gi in g:
        _, vecs = np.linalg.eigh(-zz - gi * xx)
        psi = vecs[:, 0]
        feats.append([np.real(psi.conj() @ zz @ psi) / n, np.real(psi.conj() @ xx @ psi) / n,
                      np.real(psi.conj() @ corr @ psi)])
    x = np.array(feats)
    return Dataset(x, (g > 1).astype(int), ["<ZZ> per bond", "<X> per site", "<Z0 Z_n/2>"],
                   ["ferromagnetic", "paramagnetic"], f"TFIM ground states, g = {np.round(g, 3).tolist()}")


def make_quantum_state_clusters(n_clusters=3, n_qubits=3, n_samples=150, noise=0.15, seed=0):
    """Noisy copies of ``n_clusters`` random reference states. ``X`` holds the state vectors (complex);
    use them as ``initial_state`` or compare them with fidelities."""
    rng = np.random.default_rng(seed)
    dim = 2 ** n_qubits
    refs = rng.normal(size=(n_clusters, dim)) + 1j * rng.normal(size=(n_clusters, dim))
    refs /= np.linalg.norm(refs, axis=1, keepdims=True)
    y = rng.integers(n_clusters, size=n_samples)
    states = refs[y] + noise * (rng.normal(size=(n_samples, dim)) + 1j * rng.normal(size=(n_samples, dim)))
    states /= np.linalg.norm(states, axis=1, keepdims=True)
    return Dataset(states, y, [f"amp{i}" for i in range(dim)], list(range(n_clusters)), "Noisy quantum states")


# ============================================================================ helpers
def pca(x, n_components):
    """Principal components (to fit many features into few qubits). Returns the projected data."""
    x = np.asarray(x, dtype=float)
    xc = x - x.mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:n_components].T


def minmax(x, low=0.0, high=1.0):
    x = np.asarray(x, dtype=float)
    return low + (x - x.min(0)) / (x.max(0) - x.min(0) + 1e-12) * (high - low)


def list_datasets() -> dict:
    return {"real": ["load_iris (150x4, 3 classes)", "load_wine (178x13, 3)", "load_breast_cancer (569x30, 2)",
                     "load_digits (1797x64, 10)"],
            "synthetic": ["make_blobs", "make_moons", "make_circles", "make_anisotropic", "make_varied_density",
                          "make_spirals", "make_clustering_suite"],
            "text": ["load_arxiv_recent (500 arXiv abstracts, 5 fields, 11-29 Oct 2026)", "fetch_arxiv (live)"],
            "recent": ["load_earthquakes (USGS, 743 events M>=4.5, 30 Sep - 30 Oct 2026; refresh=True for live data)"],
            "quantum": ["make_ising_phases", "make_quantum_state_clusters"]}


# ============================================================================ live / recent data
_EQ_SNAPSHOT = "2026-10-30T13:33:42Z"


def load_earthquakes(refresh: bool = False, min_magnitude: str = "4.5", period: str = "month", top_regions: int = 8):
    """Recent earthquakes worldwide from the USGS real-time feed (public domain).

    The bundled snapshot covers 30 Sep - 30 Oct 2026 (743 events with magnitude >= 4.5, generated
    2026-10-30 13:33 UTC). With ``refresh=True`` the latest data are downloaded from USGS:
    ``min_magnitude`` in {"significant", "4.5", "2.5", "1.0", "all"}, ``period`` in {"hour", "day", "week", "month"}.

    Features (good for clustering seismic zones): the epicentre on the unit sphere (x, y, z, which avoids the
    +-180 degrees longitude jump), log depth and magnitude. ``y`` is the region (the ``top_regions`` most frequent
    countries / areas, the rest is "other"): a weak label to compare clusters with. Extra columns are in
    ``Dataset.meta`` (time, latitude, longitude, depth, magnitude, place).
    """
    if refresh:
        import json
        import urllib.request
        url = f"https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/{min_magnitude}_{period}.geojson"
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.load(resp)
        rows = [(f["properties"]["time"], f["geometry"]["coordinates"][1], f["geometry"]["coordinates"][0],
                 f["geometry"]["coordinates"][2], f["properties"]["mag"], f["properties"].get("place") or "")
                for f in data["features"] if f["properties"]["mag"] is not None]
        time = np.array([np.datetime64(int(r[0]), "ms") for r in rows])
        lat, lon, depth, mag = (np.array([r[k] for r in rows], dtype=float) for k in (1, 2, 3, 4))
        place = [r[5].replace(",", ";") for r in rows]
        stamp = str(np.datetime64(int(data["metadata"]["generated"]), "ms"))
    else:
        with open(os.path.join(_DATA, "earthquakes_recent.csv")) as f:
            rows = list(csv.reader(f))[1:]
        time = np.array([np.datetime64(r[0]) for r in rows])
        lat, lon, depth, mag = (np.array([float(r[k]) for r in rows]) for k in (1, 2, 3, 4))
        place = [r[6] for r in rows]
        stamp = _EQ_SNAPSHOT
    region = [p.split("; ")[-1].strip() if "; " in p else p.strip() for p in place]
    names, counts = np.unique(region, return_counts=True)
    top = [str(n) for n in names[np.argsort(counts)[::-1][:top_regions]]]
    target_names = top + ["other"]
    y = np.array([top.index(r) if r in top else len(top) for r in region])
    la, lo = np.radians(lat), np.radians(lon)
    x = np.c_[np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la), np.log1p(np.maximum(depth, 0)), mag]
    ds = Dataset(x, y, ["x_sphere", "y_sphere", "z_sphere", "log_depth_km", "magnitude"], target_names,
                 f"USGS earthquakes M>={min_magnitude if refresh else '4.5'}, {len(x)} events, data generated {stamp}")
    ds.meta = {"time": time, "latitude": lat, "longitude": lon, "depth_km": depth, "magnitude": mag, "place": place}
    return ds


_ARXIV_FIELDS = ["quant-ph", "q-fin", "cs.CL", "astro-ph", "q-bio"]


def load_arxiv_recent(field_names=None):
    """500 arXiv preprints submitted 11-29 October 2026: 100 each from quantum physics (quant-ph), quantitative
    finance (q-fin), computation and language (cs.CL), astrophysics (astro-ph) and quantitative biology (q-bio),
    selected by primary category. ``X`` holds "title. abstract" strings, ``y`` the field; ``meta`` has ids, dates,
    sub-categories, titles and abstracts. arXiv metadata are released under CC0 1.0.
    Use ``fetch_arxiv`` for the latest papers."""
    with gzip.open(os.path.join(_DATA, "arxiv_recent.csv.gz"), "rt", newline="") as f:
        rows = list(csv.reader(f))[1:]
    fields = field_names or _ARXIV_FIELDS
    rows = [r for r in rows if r[2] in fields]
    x = np.array([f"{r[4]}. {r[5]}" for r in rows], dtype=object)
    y = np.array([fields.index(r[2]) for r in rows])
    ds = Dataset(x, y, ["text"], list(fields), "arXiv preprints, 11-29 Oct 2026 (CC0 metadata)")
    ds.meta = {"id": [r[0] for r in rows], "date": [r[1] for r in rows], "category": [r[3] for r in rows],
               "title": [r[4] for r in rows], "abstract": [r[5] for r in rows]}
    return ds


def fetch_arxiv(queries=None, max_per_query=100):
    """Latest arXiv papers via the public API (needs internet). ``queries`` maps a field name to an arXiv query,
    e.g. {"quant-ph": "cat:quant-ph", "q-fin": "cat:q-fin.RM+OR+cat:q-fin.ST"}. Papers are kept only if their
    primary category starts with the field name. Respect the API etiquette: one request every 3 seconds."""
    import time
    import urllib.request
    import xml.etree.ElementTree as ET
    queries = queries or {f: f"cat:{f}*" if "." not in f else f"cat:{f}" for f in _ARXIV_FIELDS}
    ns = {"a": "http://www.w3.org/2005/Atom", "x": "http://arxiv.org/schemas/atom"}
    texts, labels, meta = [], [], {"id": [], "date": [], "category": [], "title": [], "abstract": []}
    for k, (field_name, q) in enumerate(queries.items()):
        if k:
            time.sleep(3)
        url = (f"https://export.arxiv.org/api/query?search_query={q}&sortBy=submittedDate&sortOrder=descending"
               f"&max_results={3 * max_per_query}")
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        kept = 0
        for e in root.findall("a:entry", ns):
            prim = e.find("x:primary_category", ns).get("term")
            if not prim.startswith(field_name) or kept >= max_per_query:
                continue
            title = " ".join(e.find("a:title", ns).text.split())
            abstract = " ".join(e.find("a:summary", ns).text.split())
            texts.append(f"{title}. {abstract}")
            labels.append(k)
            for key, val in (("id", e.find("a:id", ns).text.rsplit("/", 1)[-1]), ("date", e.find("a:published", ns).text[:10]),
                             ("category", prim), ("title", title), ("abstract", abstract)):
                meta[key].append(val)
            kept += 1
    ds = Dataset(np.array(texts, dtype=object), np.array(labels), ["text"], list(queries), "arXiv API, latest papers")
    ds.meta = meta
    return ds
