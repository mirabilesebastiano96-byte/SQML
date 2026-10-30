"""Word and document embeddings written from scratch in NumPy.

* ``Word2Vec``: skip-gram with negative sampling (Mikolov et al. 2013), frequent-word subsampling,
  mini-batch SGD with a linearly decaying learning rate.
* ``GloVe``: global co-occurrence factorisation (Pennington, Socher & Manning 2014) trained with AdaGrad.
* ``SIFEmbedder``: smooth-inverse-frequency sentence vectors (Arora, Liang & Ma 2017): weighted average of word
  vectors, then removal of the first principal component.
* ``embed_documents``: one call from texts to document vectors ("tfidf_lsa", "word2vec", "word2vec_sif", "glove",
  "hybrid").
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .text import STOP_WORDS, TfidfVectorizer, lsa, tokenize

__all__ = ["Vocabulary", "Word2Vec", "GloVe", "SIFEmbedder", "embed_documents"]


def _corpus(texts, keep_stop_words=False):
    stops = set() if keep_stop_words else STOP_WORDS
    return [tokenize(t, stops) for t in texts]


class Vocabulary:
    def __init__(self, docs, min_count=2):
        counts = Counter(w for d in docs for w in d)
        self.words = sorted((w for w, c in counts.items() if c >= min_count), key=lambda w: (-counts[w], w))
        self.index = {w: i for i, w in enumerate(self.words)}
        self.counts = np.array([counts[w] for w in self.words], dtype=float)

    def __len__(self):
        return len(self.words)

    def encode(self, doc):
        return np.array([self.index[w] for w in doc if w in self.index], dtype=int)


class _WordVectors:
    def __getitem__(self, word):
        return self.vectors_[self.vocab_.index[word]]

    def __contains__(self, word):
        return word in self.vocab_.index

    def most_similar(self, word, top_n=8):
        v = self.vectors_ / (np.linalg.norm(self.vectors_, axis=1, keepdims=True) + 1e-12)
        sims = v @ v[self.vocab_.index[word]]
        order = [i for i in np.argsort(-sims) if self.vocab_.words[i] != word][:top_n]
        return [(self.vocab_.words[i], float(sims[i])) for i in order]

    def analogy(self, a, b, c, top_n=5):
        """a is to b as c is to ? (vector b - a + c)."""
        v = self.vectors_ / (np.linalg.norm(self.vectors_, axis=1, keepdims=True) + 1e-12)
        q = v[self.vocab_.index[b]] - v[self.vocab_.index[a]] + v[self.vocab_.index[c]]
        sims = v @ q
        bad = {a, b, c}
        return [(self.vocab_.words[i], float(sims[i])) for i in np.argsort(-sims) if self.vocab_.words[i] not in bad][:top_n]


class Word2Vec(_WordVectors):
    def __init__(self, dim=100, window=5, negative=5, min_count=2, epochs=15, lr=0.025, batch_size=512,
                 subsample=1e-3, seed=0):
        self.dim, self.window, self.negative, self.min_count = dim, window, negative, min_count
        self.epochs, self.lr, self.batch_size, self.subsample, self.seed = epochs, lr, batch_size, subsample, seed

    def _pairs(self, docs, rng):
        total = self.vocab_.counts.sum()
        freq = self.vocab_.counts / total
        keep_p = np.minimum(1.0, (np.sqrt(freq / self.subsample) + 1) * self.subsample / freq)
        centers, contexts = [], []
        for d in docs:
            d = d[rng.random(len(d)) < keep_p[d]]
            for i in range(len(d)):
                w = rng.integers(1, self.window + 1)             # dynamic window, as in the original code
                ctx = np.concatenate([d[max(0, i - w):i], d[i + 1:i + 1 + w]])
                centers.extend([d[i]] * len(ctx))
                contexts.extend(ctx.tolist())
        return np.array(centers, dtype=int), np.array(contexts, dtype=int)

    def fit(self, texts):
        rng = np.random.default_rng(self.seed)
        docs = _corpus(texts)
        self.vocab_ = Vocabulary(docs, self.min_count)
        docs = [self.vocab_.encode(d) for d in docs]
        v, d = len(self.vocab_), self.dim
        self.w_in = (rng.random((v, d)) - 0.5) / d
        self.w_out = np.zeros((v, d))
        noise = self.vocab_.counts ** 0.75
        noise /= noise.sum()
        sig = lambda z: 0.5 * (1 + np.tanh(0.5 * z))
        self.history_ = []
        total_steps, step = None, 0
        for epoch in range(self.epochs):
            c, o = self._pairs(docs, rng)
            order = rng.permutation(len(c))
            total_steps = total_steps or self.epochs * (len(c) // self.batch_size + 1)
            losses = []
            for s in range(0, len(c), self.batch_size):
                idx = order[s:s + self.batch_size]
                lr = self.lr * max(1e-4, 1 - step / total_steps)
                step += 1
                ci, oi = c[idx], o[idx]
                neg = rng.choice(v, (len(idx), self.negative), p=noise)
                vc, uo, un = self.w_in[ci], self.w_out[oi], self.w_out[neg]          # (B,d) (B,d) (B,k,d)
                sp = sig((vc * uo).sum(1))
                sn = sig(-np.einsum("bkd,bd->bk", un, vc))
                losses.append(-(np.log(sp + 1e-9).mean() + np.log(sn + 1e-9).sum(1).mean()))
                g_pos = (sp - 1)[:, None]                       # dL/d(vc.uo)
                g_neg = (1 - sn)[:, :, None]                    # dL/d(vc.un)
                grad_vc = g_pos * uo + (g_neg * un).sum(1)
                np.add.at(self.w_out, oi, -lr * g_pos * vc)
                np.add.at(self.w_out, neg, -lr * g_neg * vc[:, None, :])
                np.add.at(self.w_in, ci, -lr * grad_vc)
            self.history_.append(float(np.mean(losses)))
        self.vectors_ = self.w_in
        return self


class GloVe(_WordVectors):
    def __init__(self, dim=50, window=6, min_count=2, epochs=40, lr=0.05, x_max=20.0, alpha=0.75, seed=0):
        self.dim, self.window, self.min_count, self.epochs = dim, window, min_count, epochs
        self.lr, self.x_max, self.alpha, self.seed = lr, x_max, alpha, seed

    def fit(self, texts):
        rng = np.random.default_rng(self.seed)
        docs = _corpus(texts)
        self.vocab_ = Vocabulary(docs, self.min_count)
        co = Counter()
        for doc in docs:
            ids = self.vocab_.encode(doc)
            for i, w in enumerate(ids):
                for k in range(1, self.window + 1):
                    if i + k < len(ids):
                        co[(w, ids[i + k])] += 1.0 / k
                        co[(ids[i + k], w)] += 1.0 / k
        pairs = np.array(list(co.keys()), dtype=int)
        x = np.array(list(co.values()))
        wi, wj = pairs[:, 0], pairs[:, 1]
        v, d = len(self.vocab_), self.dim
        w, wt = (rng.random((v, d)) - 0.5) / d, (rng.random((v, d)) - 0.5) / d
        b, bt = np.zeros(v), np.zeros(v)
        gw, gwt, gb, gbt = np.ones((v, d)), np.ones((v, d)), np.ones(v), np.ones(v)      # AdaGrad accumulators
        fx = np.minimum(1.0, (x / self.x_max) ** self.alpha)
        logx = np.log(x)
        self.history_ = []
        for _ in range(self.epochs):
            for batch in np.array_split(rng.permutation(len(x)), max(1, len(x) // 4096)):
                i, j = wi[batch], wj[batch]
                diff = (w[i] * wt[j]).sum(1) + b[i] + bt[j] - logx[batch]
                g = fx[batch] * diff
                dw, dwt = g[:, None] * wt[j], g[:, None] * w[i]
                for arr, acc, idx, grad in ((w, gw, i, dw), (wt, gwt, j, dwt)):
                    np.add.at(acc, idx, grad ** 2)
                    np.add.at(arr, idx, -self.lr * grad / np.sqrt(acc[idx]))
                for arr, acc, idx in ((b, gb, i), (bt, gbt, j)):
                    np.add.at(acc, idx, g ** 2)
                    np.add.at(arr, idx, -self.lr * g / np.sqrt(acc[idx]))
            self.history_.append(float(0.5 * (fx * ((w[wi] * wt[wj]).sum(1) + b[wi] + bt[wj] - logx) ** 2).mean()))
        self.vectors_ = w + wt
        return self


class SIFEmbedder:
    """Document vector = sum_w a / (a + p(w)) * v_w / |doc|, then the projection on the first principal component
    (common discourse direction) is removed."""

    def __init__(self, word_vectors, a=1e-3, remove_pc=1):
        self.wv, self.a, self.remove_pc = word_vectors, a, remove_pc

    def fit_transform(self, texts):
        vocab = self.wv.vocab_
        p = vocab.counts / vocab.counts.sum()
        docs = [vocab.encode(d) for d in _corpus(texts)]
        x = np.zeros((len(docs), self.wv.vectors_.shape[1]))
        for k, ids in enumerate(docs):
            if len(ids):
                weights = self.a / (self.a + p[ids])
                x[k] = (weights[:, None] * self.wv.vectors_[ids]).sum(0) / len(ids)
        if self.remove_pc:
            _, _, vt = np.linalg.svd(x - x.mean(0), full_matrices=False)
            u = vt[:self.remove_pc]
            x = x - x @ u.T @ u
        self.components_ = x
        return x


def _tfidf_weighted(texts, wv, max_df=1.0):
    """Document vector = TF-IDF weighted average of word vectors."""
    tf = TfidfVectorizer((1, 1), min_df=1, max_df=max_df)
    t = tf.fit_transform(texts)
    names = tf.get_feature_names()
    idx = np.array([wv.vocab_.index.get(n, -1) for n in names])
    table = np.zeros((len(names), wv.vectors_.shape[1]))
    table[idx >= 0] = wv.vectors_[idx[idx >= 0]]
    return t @ table, t


def _pca(x, k):
    xc = x - x.mean(0)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    return xc @ vt[:k].T


def _unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def embed_documents(texts, method="hybrid", dim=100, n_components=None, seed=0, lsa_dim=20, **kwargs):
    """Texts -> document vectors, optionally reduced with PCA to ``n_components`` (e.g. the number of qubits).

    ``method``:
      "tfidf_lsa"      TF-IDF + latent semantic analysis
      "word2vec"       TF-IDF weighted average of Word2Vec vectors
      "word2vec_sif"   SIF average of Word2Vec vectors (Arora et al. 2017)
      "glove"          TF-IDF weighted average of GloVe vectors
      "hybrid"         [LSA (``lsa_dim``) | TF-IDF weighted Word2Vec]: topical + distributional information
    On small corpora (hundreds of documents) the word vectors are noisy, so LSA or the hybrid work best.
    """
    texts = list(texts)
    max_df = kwargs.pop("max_df", 0.5)
    if method == "tfidf_lsa":
        x = lsa(TfidfVectorizer((1, 1), min_df=2, max_df=max_df).fit_transform(texts), n_components or dim)[0]
        return _unit(x)
    if method in ("word2vec", "word2vec_sif", "hybrid"):
        wv = Word2Vec(dim=dim, seed=seed, **kwargs).fit(texts)
    else:
        wv = GloVe(dim=dim, seed=seed, **kwargs).fit(texts)
    if method == "word2vec_sif":
        x = SIFEmbedder(wv).fit_transform(texts)
    else:
        x, t = _tfidf_weighted(texts, wv)
        if method == "hybrid":
            x = np.c_[lsa(TfidfVectorizer((1, 1), min_df=2, max_df=max_df).fit_transform(texts), lsa_dim)[0], _unit(x)]
    if n_components:
        x = _pca(x, n_components)
    return _unit(x)
