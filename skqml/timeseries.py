"""Time-series clustering tools: Dynamic Time Warping and a quantum sequence kernel.

* ``dtw`` / ``dtw_matrix``: DTW distance (Sakoe & Chiba 1978) with an optional warping window, vectorised over all
  pairs of series.
* ``DTWKernel``: k(a, b) = exp(-DTW(a, b)^2 / (2 sigma^2)), usable as ``kernel=`` in every clustering model
  (QKMedoids, QSpectralClustering, QAgglomerativeClustering, QDBSCAN, QKMeans).
* ``QuantumSeriesKernel``: shift-invariant shape descriptors (Fourier magnitudes, autocorrelations, moments)
  projected onto the qubits and compared with a quantum fidelity kernel.
* ``make_time_series_clusters``: synthetic benchmark (sine, square, sawtooth, trend, burst) with random phase
  shifts and time warping, where DTW clearly beats Euclidean distance.
* ``fetch_stock_returns``: daily returns via yfinance (optional dependency), for clustering stocks by co-movement.
"""
from __future__ import annotations

import numpy as np

from .datasets import Dataset

__all__ = ["znorm", "dtw", "dtw_matrix", "DTWKernel", "EuclideanKernel", "QuantumSeriesKernel", "series_features",
           "make_time_series_clusters", "fetch_stock_returns"]


def znorm(x):
    """Normalise every series (row) to zero mean and unit variance."""
    x = np.asarray(x, dtype=float)
    return (x - x.mean(-1, keepdims=True)) / (x.std(-1, keepdims=True) + 1e-9)


def _dtw_pairs(a, b, window):
    """a, b: (P, T1), (P, T2) -> DTW distances (P,). Dynamic programming over two rolling rows,
    vectorised over the P pairs."""
    p, t1 = a.shape
    t2 = b.shape[1]
    w = max(window if window is not None else max(t1, t2), abs(t1 - t2))
    prev = np.full((p, t2 + 1), np.inf)
    prev[:, 0] = 0.0
    for i in range(1, t1 + 1):
        cur = np.full((p, t2 + 1), np.inf)
        lo, hi = max(1, i - w), min(t2, i + w)
        cost = (a[:, i - 1, None] - b[:, lo - 1:hi]) ** 2
        best_diag_up = np.minimum(prev[:, lo - 1:hi], prev[:, lo:hi + 1])       # (i-1, j-1) and (i-1, j)
        for j in range(lo, hi + 1):
            cur[:, j] = cost[:, j - lo] + np.minimum(best_diag_up[:, j - lo], cur[:, j - 1])
        prev = cur
    return np.sqrt(prev[:, t2])


def dtw(a, b, window=None):
    """DTW distance between two 1-D series. ``window`` = Sakoe-Chiba band half-width in time steps."""
    return float(_dtw_pairs(np.atleast_2d(a), np.atleast_2d(b), window)[0])


def dtw_matrix(x1, x2=None, window=None, batch=4096):
    """All pairwise DTW distances between the rows of x1 and x2 (x2 = x1 if omitted)."""
    x1 = np.asarray(x1, dtype=float)
    same = x2 is None
    x2 = x1 if same else np.asarray(x2, dtype=float)
    if same:
        ii, jj = np.triu_indices(len(x1), 1)
    else:
        ii, jj = (g.ravel() for g in np.meshgrid(np.arange(len(x1)), np.arange(len(x2)), indexing="ij"))
    out = np.zeros(len(ii))
    for s in range(0, len(ii), batch):
        out[s:s + batch] = _dtw_pairs(x1[ii[s:s + batch]], x2[jj[s:s + batch]], window)
    d = np.zeros((len(x1), len(x2)))
    d[ii, jj] = out
    if same:
        d += d.T
    return d


class DTWKernel:
    """Gaussian kernel on DTW distances. ``sigma=None`` uses the median pairwise distance of the first call."""

    handles_scaling = True

    def __init__(self, window=None, sigma=None, normalize=True):
        self.window, self.sigma, self.normalize = window, sigma, normalize

    def get_params(self):
        return {"window": self.window, "sigma": self.sigma, "normalize": self.normalize}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self

    def __call__(self, x1, x2=None):
        a = znorm(x1) if self.normalize else np.asarray(x1, dtype=float)
        b = None if x2 is None else (znorm(x2) if self.normalize else np.asarray(x2, dtype=float))
        d = dtw_matrix(a, b, self.window)
        if self.sigma is None:
            self.sigma_ = getattr(self, "sigma_", None) or float(np.median(d[d > 0]))
        sigma = self.sigma or self.sigma_
        return np.exp(-d ** 2 / (2 * sigma ** 2))


class EuclideanKernel(DTWKernel):
    """Gaussian kernel on plain Euclidean distance between (z-normalised) series: the baseline DTW improves on."""

    def __call__(self, x1, x2=None):
        a = znorm(x1) if self.normalize else np.asarray(x1, dtype=float)
        b = a if x2 is None else (znorm(x2) if self.normalize else np.asarray(x2, dtype=float))
        d = np.sqrt(np.maximum((a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None] - 2 * a @ b.T, 0))
        if self.sigma is None:
            self.sigma_ = getattr(self, "sigma_", None) or float(np.median(d[d > 0]))
        return np.exp(-d ** 2 / (2 * (self.sigma or self.sigma_) ** 2))


def series_features(x, n_freq=12):
    """Shift-invariant shape descriptors of each series: normalised Fourier magnitudes (first ``n_freq`` harmonics),
    autocorrelations at lags 1/3/6/10, skewness, kurtosis and mean absolute increment."""
    z = znorm(x)
    spec = np.abs(np.fft.rfft(z, axis=1))[:, 1:n_freq + 1]
    spec = spec / (np.linalg.norm(spec, axis=1, keepdims=True) + 1e-9)
    lags = [l for l in (1, 3, 6, 10) if l < z.shape[1]]
    acf = np.stack([(z[:, l:] * z[:, :-l]).mean(1) for l in lags], 1)
    moments = np.stack([(z ** 3).mean(1), (z ** 4).mean(1), np.abs(np.diff(z, axis=1)).mean(1)], 1)
    return np.concatenate([spec, acf, moments], 1)


class QuantumSeriesKernel:
    """Quantum fidelity kernel on time series.

    Each series is summarised by shift-invariant shape descriptors (``series_features``), which are standardised
    and projected by PCA onto ``n_qubits`` components (basis learned at the first call, then reused), and
    angle-encoded into a quantum feature map: k(a, b) = |<phi(a)|phi(b)>|^2.
    Invariance to phase shifts makes it a fast alternative to DTW.
    """

    handles_scaling = True

    def __init__(self, n_qubits=4, bandwidth=0.5, n_freq=12, feature_map="angle", reps=1):
        self.n_qubits, self.bandwidth, self.n_freq, self.feature_map, self.reps = n_qubits, bandwidth, n_freq, \
            feature_map, reps

    def get_params(self):
        return {k: getattr(self, k) for k in ("n_qubits", "bandwidth", "n_freq", "feature_map", "reps")}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        self.__dict__.pop("basis_", None)
        return self

    def embed(self, x):
        f = series_features(x, self.n_freq)
        if not hasattr(self, "basis_"):
            self.mean_, self.std_ = f.mean(0), f.std(0) + 1e-9
            fs = (f - self.mean_) / self.std_
            _, _, vt = np.linalg.svd(fs - fs.mean(0), full_matrices=False)
            self.basis_ = vt[:self.n_qubits].T
            proj = fs @ self.basis_
            self.proj_scale_ = proj.std(0) + 1e-9
        return ((f - self.mean_) / self.std_) @ self.basis_ / self.proj_scale_

    def __call__(self, x1, x2=None):
        from .models.kernels import QuantumKernel
        qk = QuantumKernel(self.feature_map, self.reps, self.bandwidth)
        e1 = self.embed(x1)
        return qk(e1) if x2 is None else qk(e1, self.embed(x2))


def make_time_series_clusters(n_per_class=40, length=80, noise=0.2, warp=0.3, seed=0):
    """Five shapes (sine, square, sawtooth, trend + oscillation, single burst) with random phase, frequency jitter
    and smooth time warping. Labels = shape."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, length)
    shapes = ["sine", "square", "sawtooth", "trend", "burst"]
    xs, ys = [], []
    for c, name in enumerate(shapes):
        for _ in range(n_per_class):
            tw = t + warp * 0.1 * np.sin(2 * np.pi * (t + rng.uniform())) * rng.uniform(0.5, 1.5)   # time warping
            ph, f = rng.uniform(0, 1), rng.uniform(2.5, 3.5)
            if name == "sine":
                s = np.sin(2 * np.pi * (f * tw + ph))
            elif name == "square":
                s = np.sign(np.sin(2 * np.pi * (f * tw + ph)))
            elif name == "sawtooth":
                s = 2 * ((f * tw + ph) % 1) - 1
            elif name == "trend":
                s = 3 * (tw - 0.5) + 0.4 * np.sin(2 * np.pi * (2 * f * tw + ph))
            else:
                centre = rng.uniform(0.2, 0.8)
                s = 3 * np.exp(-((tw - centre) / 0.05) ** 2)
            xs.append(s + noise * rng.normal(size=length))
            ys.append(c)
    order = rng.permutation(len(ys))
    return Dataset(np.array(xs)[order], np.array(ys)[order], [f"t{i}" for i in range(length)], shapes,
                   "Synthetic time-series shapes with phase shifts and time warping")


def fetch_stock_returns(tickers, start="2023-01-01", end=None, window=None):
    """Daily log-returns of ``tickers`` (yfinance). Returns a Dataset whose rows are the stocks (series = returns),
    ready for DTW / quantum-kernel clustering of co-moving assets. ``y`` is the ticker index."""
    import yfinance as yf
    data = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    data = data.dropna(axis=1, how="all").dropna()
    rets = np.log(data).diff().dropna().values.T
    if window:
        rets = rets[:, -window:]
    return Dataset(rets, np.arange(len(rets)), [str(d.date()) for d in data.index[1:]], list(data.columns),
                   f"Daily log-returns {start} - {end or 'today'}")
