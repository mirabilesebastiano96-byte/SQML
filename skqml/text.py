"""Text clustering: TF-IDF, LSA, quantum clustering pipeline and automatic cluster labels.

* ``TfidfVectorizer``: word n-grams for alphabetic languages, character bigrams for Chinese / Japanese / Korean,
  English and Italian stop words, sublinear tf, L2 normalisation.
* ``lsa``: latent semantic analysis (truncated SVD) to compress documents to a few dimensions = qubits.
* ``TextClustering``: texts -> TF-IDF -> LSA -> any scikit-qml clustering model (quantum kernel inside).
* ``cluster_keywords`` / ``cluster_labels``: class-based TF-IDF (c-TF-IDF, Grootendorst 2022) to name each cluster
  with its most distinctive words or 2-3 word phrases, as in cluster-label extraction benchmarks.
* ``label_f1``: token-overlap F1 between extracted and reference labels.
* ``load_text_clusters``: loads a labelled text-clustering corpus from CSV / TSV / JSONL or a folder of .txt files,
  e.g. the Science Data Bank "Cluster label dataset" (Wang & Zhang 2024, doi:10.57760/sciencedb.j00133.00433,
  CC BY-NC-SA 4.0), which must be downloaded by the user.
* ``make_text_clusters``: a small synthetic English corpus with known topics.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter

import numpy as np

from .datasets import Dataset

__all__ = ["tokenize", "TfidfVectorizer", "lsa", "TextClustering", "cluster_keywords", "cluster_labels",
           "label_f1", "load_text_clusters", "make_text_clusters", "STOP_WORDS"]

STOP_WORDS = set("""a an the and or but if of to in on at by for with from as is are was were be been being this that
these those it its into than then there their they we our you your he she his her them which who whom what when where
why how all any both each few more most other some such no nor not only own same so too very can will just do does did
has have had having i me my also about above after again against below between during through under until up down out
over off further once here while based using use used via may might must shall should would could new two one study
paper propose proposed approach method methods results show
il lo la i gli le un uno una di da in con su per tra fra e o ma se che chi cui non come anche più del dello della dei
degli delle al allo alla ai agli alle dal dallo dalla dai dagli dalle nel nello nella nei negli nelle sul sullo sulla
sui sugli sulle è sono era erano essere stato questo questa questi queste quello quella si ci ne""".split())

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]+")
_WORD = re.compile(r"[a-zà-öø-ÿ0-9]+(?:[-'][a-zà-öø-ÿ0-9]+)*")


def tokenize(text: str, stop_words=STOP_WORDS):
    """Lower-cased words (stop words removed) plus character bigrams for CJK runs."""
    text = text.lower()
    tokens = [w for w in _WORD.findall(text) if w not in stop_words and len(w) > 1]
    for run in _CJK.findall(text):
        tokens += [run[i:i + 2] for i in range(len(run) - 1)] if len(run) > 1 else [run]
    return tokens


def _ngrams(tokens, n_min, n_max):
    out = []
    for n in range(n_min, n_max + 1):
        out += [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return out


class TfidfVectorizer:
    def __init__(self, ngram_range=(1, 2), min_df=1, max_df=1.0, max_features=None, sublinear_tf=True,
                 stop_words=STOP_WORDS):
        self.ngram_range, self.min_df, self.max_df = ngram_range, min_df, max_df
        self.max_features, self.sublinear_tf, self.stop_words = max_features, sublinear_tf, stop_words

    def _docs(self, texts):
        return [_ngrams(tokenize(t, self.stop_words), *self.ngram_range) for t in texts]

    def fit(self, texts):
        docs = self._docs(texts)
        df = Counter(term for d in docs for term in set(d))
        n = len(docs)
        max_df = self.max_df if isinstance(self.max_df, int) else self.max_df * n
        terms = [t for t, c in df.items() if self.min_df <= c <= max_df]
        terms.sort(key=lambda t: (-df[t], t))
        if self.max_features:
            terms = terms[:self.max_features]
        self.vocabulary_ = {t: i for i, t in enumerate(sorted(terms))}
        self.idf_ = np.array([np.log((1 + n) / (1 + df[t])) + 1 for t in sorted(terms)])
        return self

    def transform(self, texts):
        x = np.zeros((len(texts), len(self.vocabulary_)))
        for row, doc in enumerate(self._docs(texts)):
            for term, c in Counter(doc).items():
                j = self.vocabulary_.get(term)
                if j is not None:
                    x[row, j] = 1 + np.log(c) if self.sublinear_tf else c
        x *= self.idf_
        return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)

    def fit_transform(self, texts):
        return self.fit(texts).transform(texts)

    def get_feature_names(self):
        return sorted(self.vocabulary_, key=self.vocabulary_.get)


def lsa(x, n_components=4):
    """Truncated SVD of the document-term matrix; rows re-normalised to unit length."""
    u, s, vt = np.linalg.svd(np.asarray(x, dtype=float), full_matrices=False)
    z = u[:, :n_components] * s[:n_components]
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12), vt[:n_components]


class TextClustering:
    """texts -> TF-IDF -> LSA (``n_components`` = qubits) -> ``clusterer`` (any scikit-qml clustering model)."""

    def __init__(self, clusterer, n_components=4, max_df=0.5, vectorizer=None):
        """``max_df``: drop words present in more than this fraction of documents. Lower it (e.g. 0.1) when many
        documents share generic phrasing ("we investigate the relation between...")."""
        self.clusterer, self.n_components, self.max_df = clusterer, n_components, max_df
        self.vectorizer = vectorizer or TfidfVectorizer(ngram_range=(1, 1), min_df=2, max_df=max_df)

    def fit(self, texts):
        self.texts_ = list(texts)
        self.tfidf_ = self.vectorizer.fit_transform(self.texts_)
        self.embedding_, self.components_ = lsa(self.tfidf_, self.n_components)
        self.labels_ = self.clusterer.fit_predict(self.embedding_)
        return self

    def fit_predict(self, texts):
        return self.fit(texts).labels_

    def keywords(self, top_n=8):
        return cluster_keywords(self.texts_, self.labels_, top_n=top_n)

    def labels(self, max_words=3):
        return cluster_labels(self.texts_, self.labels_, max_words=max_words)


def _ctfidf(texts, labels, ngram_range):
    labels = np.asarray(labels)
    clusters = [c for c in np.unique(labels) if c >= 0]
    counts = {c: Counter() for c in clusters}
    for t, c in zip(texts, labels):
        if c >= 0:
            counts[c].update(_ngrams(tokenize(t), *ngram_range))
    total = Counter()
    for c in clusters:
        total.update(counts[c])
    avg_words = np.mean([sum(counts[c].values()) for c in clusters])
    scores = {}
    for c in clusters:
        size = sum(counts[c].values()) + 1e-12
        scores[c] = {t: (k / size) * np.log(1 + avg_words / total[t]) for t, k in counts[c].items()}
    return scores


def cluster_keywords(texts, labels, top_n=8):
    """Most distinctive single words of every cluster (c-TF-IDF)."""
    scores = _ctfidf(texts, labels, (1, 1))
    return {int(c): [t for t, _ in sorted(s.items(), key=lambda kv: -kv[1])[:top_n]] for c, s in scores.items()}


def cluster_labels(texts, labels, max_words=3, min_count=2):
    """A short phrase (2..max_words words) naming each cluster: the n-gram with the highest c-TF-IDF that occurs at
    least ``min_count`` times in the cluster; falls back to the top two keywords."""
    labels = np.asarray(labels)
    scores = _ctfidf(texts, labels, (2, max_words))
    words = cluster_keywords(texts, labels, 2)
    out = {}
    for c, s in scores.items():
        members = [t for t, l in zip(texts, labels) if l == c]
        cnt = Counter(g for t in members for g in set(_ngrams(tokenize(t), 2, max_words)))
        good = [(t, v) for t, v in s.items() if cnt[t] >= min_count]
        out[int(c)] = max(good, key=lambda kv: kv[1])[0] if good else " ".join(words[int(c)])
    return out


def label_f1(predicted: str, reference: str):
    """Token-overlap F1 between two labels (1 = same words)."""
    p, r = tokenize(predicted, set()), tokenize(reference, set())
    common = sum((Counter(p) & Counter(r)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(r)
    return 2 * precision * recall / (precision + recall)


def load_text_clusters(path, text_col="text", cluster_col="cluster", label_col="label", encoding="utf-8"):
    """Load a labelled text-clustering corpus.

    Accepted layouts: a CSV / TSV / JSONL file with columns ``text``, ``cluster`` and optionally ``label``; or a
    folder with one .txt file per cluster (one sentence per line; the file name, or a first line starting with
    ``label:``, gives the cluster label). Returns a Dataset whose ``X`` is the list of texts.
    """
    texts, clusters, names = [], [], {}
    if os.path.isdir(path):
        for i, fn in enumerate(sorted(f for f in os.listdir(path) if f.endswith(".txt"))):
            with open(os.path.join(path, fn), encoding=encoding) as f:
                lines = [l.strip() for l in f if l.strip()]
            label = os.path.splitext(fn)[0]
            if lines and lines[0].lower().startswith("label:"):
                label, lines = lines[0].split(":", 1)[1].strip(), lines[1:]
            names[i] = label
            texts += lines
            clusters += [i] * len(lines)
    else:
        rows = []
        with open(path, encoding=encoding) as f:
            if path.endswith(".jsonl"):
                rows = [json.loads(l) for l in f if l.strip()]
            else:
                import csv
                rows = list(csv.DictReader(f, delimiter="\t" if path.endswith(".tsv") else ","))
        keys = {}
        for r in rows:
            cid = keys.setdefault(r[cluster_col], len(keys))
            texts.append(r[text_col])
            clusters.append(cid)
            if label_col in r and r[label_col]:
                names[cid] = r[label_col]
        for k, cid in keys.items():
            names.setdefault(cid, str(k))
    return Dataset(np.array(texts, dtype=object), np.array(clusters), ["text"],
                   [names[i] for i in range(len(names))], f"text clusters from {os.path.basename(path)}")


def make_text_clusters(n_per_topic=40, seed=0):
    """Synthetic sentences on five research topics (quantum computing, finance, medicine, climate, linguistics),
    built from templates and topic vocabularies. Labels are the topics. Synthetic: for demos and tests only."""
    rng = np.random.default_rng(seed)
    topics = {
        "quantum computing": ["qubit", "entanglement", "quantum circuit", "superposition", "variational ansatz",
                              "quantum kernel", "decoherence", "quantum gate", "error correction", "quantum hardware"],
        "financial risk": ["volatility", "value at risk", "portfolio", "stock returns", "credit risk", "market crash",
                           "hedging", "interest rates", "asset pricing", "liquidity"],
        "medical imaging": ["tumour", "radiology", "mri scan", "segmentation", "lesion", "diagnosis", "patients",
                            "ct images", "clinical trial", "pathology"],
        "climate science": ["rainfall", "temperature anomaly", "carbon emissions", "sea level", "drought",
                            "climate model", "heatwave", "precipitation", "global warming", "glacier"],
        "linguistics": ["syntax", "word embeddings", "morphology", "corpus", "translation", "language model",
                        "semantics", "dialect", "phonology", "parsing"],
    }
    templates = ["How does {a} affect {b}?", "We investigate the relation between {a} and {b}.",
                 "Can {a} improve the prediction of {b}?", "What is the role of {a} in {b} under uncertainty?",
                 "This work measures {a} together with {b} on real data.", "Why do {a} and {b} change over time?"]
    texts, labels = [], []
    names = list(topics)
    for c, name in enumerate(names):
        vocab = topics[name]
        for _ in range(n_per_topic):
            a, b = rng.choice(vocab, 2, replace=False)
            texts.append(templates[rng.integers(len(templates))].format(a=a, b=b))
            labels.append(c)
    order = rng.permutation(len(texts))
    return Dataset(np.array(texts, dtype=object)[order], np.array(labels)[order], ["text"], names,
                   "Synthetic research-question sentences on five topics")
