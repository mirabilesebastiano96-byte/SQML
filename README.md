# scikit-qml

**Quantum machine learning, written from scratch.** Import name: `skqml`.

A pure-NumPy library that implements the main quantum machine
learning models, together with everything they need underneath: a state-vector simulator, exact circuit
gradients and a small automatic-differentiation engine.

The only runtime dependency is NumPy. There is no Qiskit, PennyLane, PyTorch or TensorFlow.

![tests](https://github.com/mirabilesebastiano96-byte/scikit-qml/actions/workflows/tests.yml/badge.svg)

```
pip install git+https://github.com/mirabilesebastiano96-byte/scikit-qml.git
```

or, for development:

```
git clone https://github.com/mirabilesebastiano96-byte/scikit-qml.git
cd scikit-qml
pip install -e ".[dev]"
python -m pytest tests      # 65 tests, including gradient checks against finite differences
```

Examples (`examples/`): `quickstart.ipynb` (tour of all models), `text_and_qnlp.ipynb` (embeddings, QNLP),
`arxiv_text_clustering.ipynb` (quantum clustering of 500 arXiv papers from October 2026).

---

## Why

Most QML code hides the interesting parts inside large frameworks. In scikit-qml every piece is a few hundred
readable lines that you can open, step through and modify:

| Layer | File | What it does |
|---|---|---|
| Autograd | `skqml/autograd.py` | reverse-mode differentiation on NumPy arrays (`Tensor`, incl. `einsum`) |
| Gates | `skqml/quantum/gates.py` | 16 fixed gates, 14 parameterised gates (incl. RBS, XY) defined by their generator `G`, with `U(θ) = exp(-iθG)` |
| Simulator | `skqml/quantum/circuit.py` | batched state vectors, circuit builder, **adjoint differentiation** |
| Differentiable ops | `skqml/quantum/functions.py` | `expval`, `probs`, `amplitudes`, `statevector`, `parameter_shift`; `Circuit.inverse()` |
| Templates | `skqml/quantum/templates.py` | angle, amplitude and ZZ encodings; hardware-efficient, strongly-entangling and RealAmplitudes ansätze; data re-uploading |
| Neural nets | `skqml/nn.py` | `Linear`, `QuantumLayer`, `Sequential`, losses, `Adam`/`SGD`, `train()` |
| Text / time series | `skqml/text.py`, `skqml/embeddings.py`, `skqml/qnlp.py`, `skqml/timeseries.py` | TF-IDF, LSA, Word2Vec, GloVe, QNLP (DisCoCat, spiders, word sequence); DTW, quantum series kernel |
| Tuning | `skqml/tuning.py` | grid / random / successive-halving search, cross-validation splitters, scorers |
| Models | `skqml/models/` | the estimators listed below, with a scikit-learn-style API |

### How gradients flow

```
 data ──► Linear ──► arctan ──► [ quantum circuit ] ──► <Z_q> ──► Linear ──► loss
          ▲ autograd ▲          ▲ adjoint method  ▲              ▲ autograd ▲
```

For a gate `U = exp(-iθG)`, with |λ⟩ = ∂L/∂⟨ψ| back-propagated through the circuit:

$$\frac{\partial L}{\partial\theta} = 2\,\mathrm{Im}\,\langle\lambda|\,G\,|\psi\rangle$$

This needs about three circuit simulations per gradient, independent of the number of parameters.
`parameter_shift()` gives the hardware-compatible ±π/2 rule as a cross-check.

---

## Models

60+ estimators and components across 12 families. Every trainable model has a gradient test against finite
differences in `tests/`.

| Family | Estimator | Reference |
|---|---|---|
| **Classification / regression** | `VQCClassifier`, `VQCRegressor` | Havlíček et al., *Nature* 2019; Schuld et al., *PRA* 2020 |
| | `DataReuploadingClassifier`, `DataReuploadingRegressor` | Pérez-Salinas et al., *Quantum* 2020 |
| | `HybridQNNClassifier`, `HybridQNNRegressor` | Mari et al., *Quantum* 2020 |
| | `QKANClassifier`, `QKANRegressor` | Jiang et al., quantum KAN with DARUAN activations, 2025 |
| | `QuantumExtremeLearningMachine` | Mujal et al., *Adv. Quantum Technol.* 2021 |
| **Kernels** | `QuantumKernel` (fidelity: ZZ / angle / IQP) | Havlíček et al. 2019; Schuld & Killoran, *PRL* 2019 |
| | `ProjectedQuantumKernel` | Huang et al., *Nat. Commun.* 2021 |
| | `TrainableQuantumKernel` (kernel-target alignment) | Hubregtsen et al., *PRA* 2022; Glick et al., *Nat. Phys.* 2024 |
| | `QSVC` (SMO, one-vs-rest), `QKernelRidge`, `QGaussianProcessRegressor` | Rebentrost et al., *PRL* 2014 (QSVM) |
| **Clustering** | `QKMeans`, `QKMedoids` | kernel k-means / k-medoids in the quantum feature space |
| | `QSpectralClustering` (kernel or k-NN affinity) | Ng, Jordan & Weiss, *NeurIPS* 2002, on a quantum affinity |
| | `QAgglomerativeClustering` (single / complete / average / ward, dendrogram) | hierarchical clustering on Hilbert-space distances |
| | `QDBSCAN` | Ester et al., *KDD* 1996, density-based with noise detection |
| | `QMeanShift` | mean shift on projected-kernel (Bloch-vector) features |
| | `SwapTestKMeans` (exact or with shots) | Lloyd, Mohseni & Rebentrost 2013; Khan et al. 2019 (swap-test distances) |
| | `VariationalQuantumClustering` | Born-rule cluster probabilities trained to maximise I(X;C) |
| | `QAOAClustering` | Otterbach et al., arXiv:1712.05771 (clustering as weighted Max-Cut) |
| | metrics: `silhouette_score`, `davies_bouldin_score`, `adjusted_rand_score`, `normalized_mutual_info_score` | — |
| **Dimensionality reduction** | `QKernelPCA` | kernel PCA in the quantum feature space |
| **Tensor networks** | `TTNClassifier`, `MERAClassifier` | Grant et al., *npj Quantum Inf.* 2018 |
| **Images** | `QCNNClassifier` | Cong, Choi & Lukin, *Nat. Phys.* 2019 |
| | `QuanvolutionalClassifier`, `QuanvolutionFilter` | Henderson et al., *Quantum Mach. Intell.* 2020 |
| **Orthogonal / transformers** | `OrthogonalLayer`, `OrthogonalQNNClassifier` (RBS pyramids) | Landman et al., *Quantum* 2022 |
| | `QuantumTransformerClassifier`, `QuantumTransformerRegressor` | Cherrat et al., *Quantum* 2024 |
| | `QSANNClassifier`, `QSANNRegressor` | Li et al., *Sci. China Inf. Sci.* 2023 |
| **Sequences / time series** | `QRNNRegressor`, `QGRURegressor` | hybrid quantum recurrent cells |
| | `QLSTMRegressor` | Chen, Yoo & Fang, *ICASSP* 2022 |
| | `QKANLSTMRegressor`, `QKANLayer` | Hsu et al., arXiv:2512.05049 |
| | `QuantumReservoirRegressor` | Fujii & Nakajima, *PR Applied* 2017 |
| | `QMSMoE` (probabilistic forecaster) | this library: quantum multi-scale memory + mixture of experts |
| | `LSTMRegressor` | classical benchmark |
| **Graphs** | `QGNNClassifier`, `QGNNRegressor` | Verdon et al. 2019; Skolik et al., *npj Quantum Inf.* 2023 |
| **Generative** | `QCBM` | Liu & Wang, *PRA* 2018; Benedetti et al., *npj QI* 2019 |
| | `QGAN` | Lloyd & Weedbrook, *PRL* 2018; Dallaire-Demers & Killoran, *PRA* 2018 |
| | `QuantumBoltzmannMachine` | Amin et al., *PRX* 2018 |
| | `QuantumAutoencoder` | Romero, Olson & Aspuru-Guzik, *QST* 2017 |
| **Reinforcement learning** | `VQCPolicy` + `QuantumREINFORCE` | Jerbi et al., *NeurIPS* 2021 |
| | `QuantumDQN` | Skolik, Jerbi & Dunjko, *Quantum* 2022 |
| | `CartPole` | built-in environment (Barto et al. 1983) |
| **Variational algorithms** | `VQE` (Adam / QNG / SPSA) | Peruzzo et al., *Nat. Commun.* 2014 |
| | `VQD` (excited states) | Higgott, Wang & Brierley, *Quantum* 2019 |
| | `QAOAMaxCut`, `QAOAIsing` (QUBO / Ising) | Farhi, Goldstone & Gutmann, arXiv:1411.4028 |
| **Optimisers** | `SPSA`, `QuantumNaturalGradient`, `Adam`, `SGD` | Spall 1992; Stokes et al., *Quantum* 2020 |

## Quick start

### Build and differentiate a circuit

```python
import numpy as np
from skqml import Circuit, Observable, Tensor, expval

qc = Circuit(2)
x, w = qc.inputs(2), qc.weights(2)
qc.ry(0, x[0]).ry(1, x[1]).cnot(0, 1).ry(0, w[0]).rz(1, w[1])
print(qc.draw())

inputs  = Tensor(np.random.randn(5, 2), requires_grad=True)
weights = Tensor(np.random.randn(2), requires_grad=True)
out = expval(qc, [Observable("Z0"), Observable("Z0 Z1")], inputs, weights)   # (5, 2)
out.sum().backward()
print(weights.grad, inputs.grad.shape)
```

### Classify

```python
from skqml.models import VQCClassifier, QSVC, QuantumKernel

clf = VQCClassifier(layers=2, epochs=40, lr=0.05).fit(X_train, y_train)   # one qubit per feature
print(clf.score(X_test, y_test), clf.predict_proba(X_test[:3]))

svc = QSVC(QuantumKernel("zz", reps=2)).fit(X_train, y_train)
```

### Forecast a time series

```python
from skqml.models import QLSTMRegressor, QMSMoE

# windows: X (N, T, D), target y (N,)
qlstm = QLSTMRegressor(hidden=4, n_qubits=4, epochs=30).fit(X_train, y_train)

qms = QMSMoE(n_qubits=4, n_experts=3, epochs=60).fit(X_train, y_train)
mean = qms.predict(X_test)
low, high = qms.predict_interval(X_test, level=0.9)       # 90% prediction interval
```

### Cluster

```python
from skqml.models import QSpectralClustering, QDBSCAN, SwapTestKMeans, QuantumKernel, adjusted_rand_score

labels = QSpectralClustering(2, QuantumKernel("angle", bandwidth=0.5), affinity="knn").fit_predict(X)
noise_aware = QDBSCAN(eps=0.1, min_samples=4).fit(X).labels_          # -1 = noise
swap = SwapTestKMeans(3, shots=1024).fit(X)                             # distances from swap-test circuits
```

### Generate, compress, optimise

```python
from skqml.models import QCBM, QGAN, QuantumAutoencoder, VQE, QAOAMaxCut

QCBM(n_qubits=4).fit(integer_samples).sample(100)
QGAN(n_qubits=4).fit(continuous_data).sample(100)
QuantumAutoencoder(n_qubits=3, n_latent=2).fit(vectors).reconstruct(vectors)
VQE(Observable("Z0 Z1") + 0.5 * Observable("X0"), n_qubits=2).fit().energy_
QAOAMaxCut([(0, 1), (1, 2), (2, 0)]).fit().best_cut()
```

### Write your own model

Any `Module` that combines `QuantumLayer`s and classical layers can be trained with `skqml.nn.train`:

```python
from skqml import Tensor
from skqml.nn import Module, Linear, QuantumLayer, train, mse_loss
from skqml.quantum.templates import variational_circuit

class MyNet(Module):
    def __init__(self):
        self.q = QuantumLayer(variational_circuit(4, layers=3, reupload=True))
        self.out = Linear(4, 1)
    def forward(self, x):
        return self.out(self.q(x))

net = MyNet()
train(net, lambda m, x, y: mse_loss(m(Tensor(x))[:, 0], y), (X, y), epochs=50, lr=0.02)
```

---

## Datasets (`skqml.datasets`)

| Kind | Loader | Size | Good for |
|---|---|---|---|
| Real | `load_iris()` | 150 × 4, 3 classes | clustering with 4 qubits, one per feature |
| Real | `load_wine()` | 178 × 13, 3 classes | clustering after `pca(X, 4)` |
| Real | `load_breast_cancer()` | 569 × 30, 2 classes | binary tasks after PCA |
| Real | `load_digits(n_classes)` | 1797 × 64 (8×8 images) | quanvolution, QCNN, clustering after PCA |
| **Recent** | `load_earthquakes()` | 743 earthquakes M ≥ 4.5 worldwide, 30 Sep – 30 Oct 2026 (USGS); `refresh=True` downloads the latest data | clustering seismic zones on the sphere |
| Synthetic | `make_blobs`, `make_moons`, `make_circles`, `make_anisotropic`, `make_varied_density`, `make_spirals` | any size | stress tests: convex, non-convex, uneven density |
| Synthetic | `make_clustering_suite()` | 6 data sets | a comparison grid in one call |
| Quantum | `make_ising_phases(n_qubits)` | measurable quantities of transverse-field Ising ground states | clustering quantum phases (ferro / para, transition at g = 1) |
| Quantum | `make_quantum_state_clusters()` | noisy copies of random states | clustering state vectors by fidelity |

```python
from skqml.datasets import load_wine, pca
X, y = load_wine()
X4 = pca((X - X.mean(0)) / X.std(0), 4)          # 13 features -> 4 qubits
```

Real data come from the UCI Machine Learning Repository (CC BY 4.0) and the USGS Earthquake Hazards Program (public domain); see `DATASETS.md`.

## Text and time-series clustering

### Time series (`skqml.timeseries`)

| Tool | What it does |
|---|---|
| `dtw(a, b, window)`, `dtw_matrix(X)` | Dynamic Time Warping (Sakoe & Chiba 1978), with an optional Sakoe-Chiba band, vectorised over all pairs |
| `DTWKernel()` | Gaussian kernel on DTW distances; pass it as `kernel=` to any clustering model |
| `QuantumSeriesKernel(n_qubits)` | shift-invariant shape descriptors (Fourier magnitudes, autocorrelations, moments) -> PCA onto the qubits -> quantum fidelity kernel |
| `EuclideanKernel()` | the point-by-point baseline that DTW improves on |
| `make_time_series_clusters()` | 5 shapes with random phase and time warping |
| `fetch_stock_returns(tickers)` | daily log-returns via yfinance, to cluster stocks that move together |

```python
from skqml.timeseries import DTWKernel, QuantumSeriesKernel, make_time_series_clusters
X, y = make_time_series_clusters()
QKMedoids(5, DTWKernel()).fit_predict(X)                       # k-medoids + DTW, the classic choice
QSpectralClustering(5, QuantumSeriesKernel(4), affinity="knn").fit_predict(X)
```

On the synthetic benchmark (150 series), ARI is about 0.25 with Euclidean distance, 0.71-0.76 with DTW and
about 0.68 with the quantum series kernel.

### Text (`skqml.text`)

| Tool | What it does |
|---|---|
| `TfidfVectorizer` | word n-grams (English / Italian stop words), character bigrams for Chinese, Japanese and Korean |
| `lsa(X, k)` | latent semantic analysis: documents -> k dimensions = k qubits |
| `TextClustering(clusterer, n_components, max_df)` | texts -> TF-IDF -> LSA -> any quantum clustering model |
| `cluster_keywords`, `cluster_labels` | names each cluster with its most distinctive words / 2-3 word phrase (c-TF-IDF) |
| `label_f1` | word-overlap F1 between an extracted and a reference label |
| `load_text_clusters(path)` | CSV / TSV / JSONL (`text`, `cluster`, `label`) or a folder of `.txt` files, one per cluster |
| `make_text_clusters()` | synthetic research-question sentences on 5 topics |

`load_text_clusters` reads, for example, the *Cluster label dataset* of the Science Data Bank (Wang & Zhang 2024,
doi:10.57760/sciencedb.j00133.00433): 275 clusters of 15-25 research-question sentences, each with a manual
2-3 word label. Its licence is CC BY-NC-SA 4.0, so it is not bundled: download it from the Science Data Bank.

### Word and document embeddings (`skqml.embeddings`)

| Tool | What it does |
|---|---|
| `Word2Vec` | skip-gram with negative sampling, sub-sampling, linear learning-rate decay (Mikolov et al. 2013) |
| `GloVe` | co-occurrence factorisation with AdaGrad (Pennington et al. 2014) |
| `SIFEmbedder` | smooth-inverse-frequency sentence vectors (Arora et al. 2017) |
| `embed_documents(texts, method)` | `"tfidf_lsa"`, `"word2vec"` (TF-IDF weighted), `"word2vec_sif"`, `"glove"`, `"hybrid"` (LSA + Word2Vec) |
| `most_similar`, `analogy` | nearest words and vector analogies |

### Quantum NLP (`skqml.qnlp`)

| Model | Idea | Reference |
|---|---|---|
| `DisCoCatClassifier` | pregroup types (n, adj, iv, tv); word states on qubits; grammar reductions contract the wires; the sentence wire is measured | Coecke, Sadrzadeh & Clark 2010; Lorenz et al., *JAIR* 2023 |
| `SpiderQNLPClassifier` | bag of words: word states merged by the Frobenius spider (elementwise product, i.e. a post-selected circuit) | lambeq "spiders" reader (Kartsaklis et al. 2021) |
| `WordSequenceQNLPClassifier` | each word is a parameterised block applied in reading order to one register | lambeq "stairs" / word-sequence readers |
| `QuantumWordEmbeddings` | words as quantum states, skip-gram with negative sampling on fidelities | quantum word embeddings |
| `make_mc_dataset` | synthetic typed sentences, food vs IT (meaning-classification task) | in the spirit of Lorenz et al. |

### Recent text data

`load_arxiv_recent()` bundles 500 arXiv preprints (titles + abstracts) submitted 11-29 October 2026: 100 each from
quant-ph, q-fin, cs.CL, astro-ph and q-bio (arXiv metadata, CC0). `fetch_arxiv()` downloads the latest ones.

| Representation (8 dimensions) | QKMeans ARI | Quantum spectral ARI |
|---|---|---|
| TF-IDF + LSA | 0.67 | 0.82 |
| Hybrid (LSA + Word2Vec) | 0.51 | **0.87** |
| Word2Vec (TF-IDF weighted) | 0.40 | 0.48 |

Word vectors trained on 500 abstracts are noisy: on small corpora topical (LSA) features do most of the work.

## Choosing hyper-parameters

Every model exposes its hyper-parameters as constructor arguments, with scikit-learn-style
`get_params()` / `set_params()` / `clone()`. Nested objects use `__`, as in `kernel__reps`.

```python
from skqml.tuning import hyperparameters
hyperparameters(VQCClassifier)
# {'layers': 2, 'ansatz': 'strongly_entangling', 'encoding': 'angle', 'input_transform': 'arctan',
#  'reupload': False, 'epochs': 50, 'lr': 0.02, 'batch_size': 32, 'optimizer': 'adam', 'weight_decay': 0.0, ...}
```

| Tool | What it does |
|---|---|
| `GridSearchCV(est, {"layers": [1, 2, 3], "lr": [0.01, 0.05]})` | tries every combination |
| `RandomSearchCV(est, {"lr": LogUniform(1e-3, 1e-1), "ansatz": Choice([...])}, n_iter=20)` | random combinations from lists or distributions (`Uniform`, `LogUniform`, `IntUniform`, `Choice`) |
| `SuccessiveHalvingSearch(est, space, n_candidates=27, resource="epochs")` | many candidates with few epochs, keeps the best third, triples the epochs; cheap for slow quantum models |
| `KFold`, `StratifiedKFold`, `TimeSeriesSplit(gap=...)`, `HoldOut` | validation schemes (`TimeSeriesSplit` never trains on the future) |
| `cross_val_score(est, X, y, cv, scoring)` | scores of one configuration |
| `ClusterSearch(est, grid, scoring="silhouette")` | unsupervised search for clustering models |
| scoring | `"accuracy"`, `"f1_macro"`, `"r2"`, `"neg_mse"`, `"neg_rmse"`, `"neg_mae"`, `"log_likelihood"` (for `QMSMoE`), or any `f(estimator, X, y)` |

For clustering there are no labels, so `ClusterSearch` ranks configurations by silhouette, Davies-Bouldin,
or, when labels are available for evaluation, ARI / NMI:

```python
from skqml.tuning import ClusterSearch
search = ClusterSearch(QSpectralClustering(affinity="knn"),
                       {"n_clusters": [2, 3, 4, 5], "n_neighbors": [5, 10, 15], "kernel__bandwidth": [0.3, 0.5, 1.0]},
                       scoring="silhouette").fit(X)
search.best_params_, search.labels_
```

```python
from skqml.tuning import GridSearchCV, RandomSearchCV, SuccessiveHalvingSearch, TimeSeriesSplit, LogUniform, Choice

search = RandomSearchCV(
    QSVC(),
    {"c": LogUniform(0.1, 100), "kernel__feature_map": Choice(["zz", "angle", "iqp_angle"]), "kernel__reps": [1, 2, 3]},
    n_iter=20, cv=5, scoring="accuracy", verbose=True,
).fit(X, y)
print(search.best_params_, search.best_score_)
search.results()            # pandas DataFrame of every trial, if pandas is installed
model = search.best_estimator_

halving = SuccessiveHalvingSearch(
    QLSTMRegressor(epochs=30),
    {"hidden": [2, 4, 8], "n_qubits": [3, 4], "layers": [1, 2], "lr": LogUniform(1e-3, 5e-2)},
    n_candidates=18, min_resource=3, cv=TimeSeriesSplit(3), scoring="neg_rmse",
).fit(windows, targets)
```

## Conventions and limits

* Wire 0 is the most significant bit: basis state `|q0 q1 … q_{n-1}⟩` has index `Σ q_k 2^(n-1-k)`.
* Simulation is exact: the state vector costs `2^n` complex numbers per sample, so up to about 20 qubits on a laptop.
* `QuantumLayer(shots=1024)` adds measurement noise at evaluation time; training uses exact expectations.
* All circuits are simulated, so run times say nothing about real quantum hardware.

## Roadmap

Noise models and density matrices · Quixer (LCU/QSVT transformer) · QRWKV · quantum diffusion models ·
trainable quanvolution filters · continuous-variable models · export to OpenQASM.

## License

MIT
