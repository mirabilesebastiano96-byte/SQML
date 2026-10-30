"""Quantum natural language processing (QNLP), written from scratch.

Sentences become quantum circuits or tensor networks of word states; word meanings are trainable.

* ``WordSequenceQNLPClassifier`` (word-sequence / "stairs" reader, as in lambeq): every word is a parameterised
  block (RX, RZ on every qubit + CRZ ring) applied in reading order to one sentence register. Padding words have all
  angles at zero, which makes their block the identity, so sentences of different length run in one batch.
* ``SpiderQNLPClassifier`` (bag-of-words DisCoCirc / "spiders" reader): each word is a state on ``n_qubits``;
  the Frobenius spider merges them by elementwise multiplication of amplitudes (exact simulation of the
  post-selected circuit); the sentence state is measured.
* ``DisCoCatClassifier`` (Coecke, Sadrzadeh & Clark 2010; Lorenz et al. 2023): words carry pregroup types
  n (noun), adj = n.n^l, iv = n^r.s, tv = n^r.s.n^l. Word states live on one qubit per wire; grammar reductions
  (cups) contract the wires, leaving the sentence wire s, which is measured. Handles noun phrases with adjectives
  and intransitive / transitive sentences.
* ``QuantumWordEmbeddings``: every word is a quantum state; skip-gram with negative sampling on the fidelity
  |<w|c>|^2. ``similarity`` and ``most_similar`` use fidelities.
* ``make_mc_dataset``: synthetic meaning-classification sentences (food vs IT) with pregroup types, in the spirit
  of the MC task of Lorenz et al.

Word circuits use RY + CNOT (real amplitudes), so all contractions stay real and differentiable.
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from .autograd import Tensor, concat, einsum, no_grad, stack
from .nn import Adam, Module, Parameter, cross_entropy, train
from .quantum.circuit import Circuit, Observable
from .quantum.functions import amplitudes, probs
from .text import tokenize

__all__ = ["WordSequenceQNLPClassifier", "SpiderQNLPClassifier", "DisCoCatClassifier", "QuantumWordEmbeddings",
           "make_mc_dataset", "word_state_circuit"]


class _TextClassifierBase:
    def _vocab(self, texts):
        counts = Counter(w for t in texts for w in self._tok(t))
        self.vocab_ = {w: i for i, w in enumerate(sorted(w for w, c in counts.items() if c >= self.min_count))}

    def _tok(self, t):
        return tokenize(t) if isinstance(t, str) else [w for w, _ in t]

    def predict_proba(self, texts):
        with no_grad():
            return np.exp(self.net_(self._encode(texts)).data)

    def predict(self, texts):
        return self.classes_[np.argmax(self.predict_proba(texts), 1)]

    def score(self, texts, y):
        return float(np.mean(self.predict(texts) == np.asarray(y)))

    def fit(self, texts, y):
        self.classes_, yi = np.unique(y, return_inverse=True)
        self._vocab(texts)
        self.net_ = self._build(len(self.classes_))
        enc = self._encode(texts)
        self.history_ = train(self.net_, lambda m, e, t: cross_entropy(m(e), t), (enc, yi), self.epochs,
                              self.batch_size, self.lr, seed=self.seed)
        return self


# =============================================================================== word sequence (stairs)
class _SeqNet(Module):
    def __init__(self, vocab_size, n_qubits, max_words, n_classes, rng):
        self.n, self.max_words, self.k = n_qubits, max_words, n_classes
        per_word = 3 * n_qubits
        self.table = Parameter(rng.normal(0, 0.3, (vocab_size + 1, per_word)))      # last row = padding
        self.mask = np.ones((vocab_size + 1, 1))
        self.mask[-1] = 0.0
        qc = Circuit(n_qubits)
        for q in range(n_qubits):
            qc.h(q)
        for _ in range(max_words):
            a = qc.inputs(per_word)
            for q in range(n_qubits):
                qc.rx(q, a[3 * q]).rz(q, a[3 * q + 1])
            for q in range(n_qubits if n_qubits > 2 else n_qubits - 1):
                qc.crz(q, (q + 1) % n_qubits, a[3 * q + 2])
        self.circuit = qc
        self.readout = list(range(max(1, int(np.ceil(np.log2(n_classes))))))

    def forward(self, tokens):
        angles = (self.table * self.mask)[tokens]                                    # (B, L, P)
        p = probs(self.circuit, self.readout, angles.reshape(len(tokens), -1))[:, :self.k]
        return (p / p.sum(-1, keepdims=True) + 1e-9).log()


class WordSequenceQNLPClassifier(_TextClassifierBase):
    """Sentence = one register of ``n_qubits``; words act in reading order (first ``max_words`` known words)."""

    def __init__(self, n_qubits=4, max_words=10, min_count=2, epochs=30, lr=0.05, batch_size=32, seed=0):
        self.n_qubits, self.max_words, self.min_count = n_qubits, max_words, min_count
        self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed

    def _build(self, k):
        return _SeqNet(len(self.vocab_), self.n_qubits, self.max_words, k, np.random.default_rng(self.seed))

    def _encode(self, texts):
        pad = len(self.vocab_)
        out = np.full((len(texts), self.max_words), pad)
        for i, t in enumerate(texts):
            ids = [self.vocab_[w] for w in self._tok(t) if w in self.vocab_][:self.max_words]
            out[i, :len(ids)] = ids
        return out


# =============================================================================== spiders (bag of words)
def word_state_circuit(n_qubits, layers=2):
    """Real-amplitude word ansatz: layers of RY on every qubit + CNOT chain, final RY layer. All angles are inputs."""
    qc = Circuit(n_qubits)
    for _ in range(layers):
        for q in range(n_qubits):
            qc.ry(q, qc.inputs(1)[0])
        for q in range(n_qubits - 1):
            qc.cnot(q, q + 1)
    for q in range(n_qubits):
        qc.ry(q, qc.inputs(1)[0])
    return qc


class _WordStates(Module):
    """Trainable table of word states: amplitudes of ``word_state_circuit`` for each word's angles."""

    def __init__(self, vocab_size, n_qubits, layers, rng):
        self.circuit = word_state_circuit(n_qubits, layers)
        self.table = Parameter(rng.uniform(-np.pi, np.pi, (vocab_size, self.circuit.n_inputs)))
        self.basis = list(range(2 ** n_qubits))

    def states(self, ids=None):
        angles = self.table if ids is None else self.table[ids]
        return amplitudes(self.circuit, self.basis, angles)                          # (W, 2^n), unit norm


class _SpiderNet(Module):
    def __init__(self, vocab_size, n_qubits, layers, n_classes, rng):
        self.words = _WordStates(vocab_size, n_qubits, layers, rng)
        self.k = n_classes
        self.r = max(1, int(np.ceil(np.log2(n_classes))))
        self.n = n_qubits

    def forward(self, tokens):
        uniq, inv = np.unique(tokens[tokens >= 0], return_inverse=True)
        st = self.words.states(uniq)                                                  # (U, 2^n)
        ones = Tensor(np.ones((1, 2 ** self.n)))
        table = concat([st, ones], 0)                                                 # padding = spider identity
        lookup = np.full(tokens.shape, len(uniq))
        lookup[tokens >= 0] = inv
        merged = table[lookup[:, 0]]
        for j in range(1, tokens.shape[1]):
            merged = merged * table[lookup[:, j]]                                     # Frobenius multiplication
        p = (merged ** 2).reshape(len(tokens), 2 ** self.r, -1).sum(-1)[:, :self.k]    # measure readout qubits
        return (p / p.sum(-1, keepdims=True) + 1e-12).log()


class SpiderQNLPClassifier(_TextClassifierBase):
    def __init__(self, n_qubits=3, layers=2, max_words=30, min_count=2, epochs=40, lr=0.05, batch_size=32, seed=0):
        self.n_qubits, self.layers, self.max_words, self.min_count = n_qubits, layers, max_words, min_count
        self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed

    def _build(self, k):
        return _SpiderNet(len(self.vocab_), self.n_qubits, self.layers, k, np.random.default_rng(self.seed))

    def _encode(self, texts):
        out = np.full((len(texts), self.max_words), -1)
        for i, t in enumerate(texts):
            ids = [self.vocab_[w] for w in self._tok(t) if w in self.vocab_][:self.max_words]
            out[i, :len(ids)] = ids
        return out


# =============================================================================== DisCoCat (pregroup)
_ARITY = {"n": 1, "adj": 2, "iv": 2, "tv": 3}


def _parse(sentence):
    """Typed sentence -> (subject NP, verb, object NP or None). NP = list of adjectives + noun."""
    words = list(sentence)
    vi = next(i for i, (_, t) in enumerate(words) if t in ("iv", "tv"))
    subj, verb, obj = words[:vi], words[vi], words[vi + 1:]
    if verb[1] == "tv" and not obj:
        raise ValueError(f"transitive verb without object: {sentence}")
    return subj, verb, (obj if verb[1] == "tv" else None)


class _DisCoCatNet(Module):
    def __init__(self, vocab, qubits_per_wire, layers, n_classes, rng):
        self.vocab, self.q, self.k = vocab, qubits_per_wire, n_classes
        self.dim = 2 ** qubits_per_wire
        self.ansatz = {t: _WordStates(sum(1 for (_, tt) in vocab if tt == t) or 1, a * qubits_per_wire, layers, rng)
                       for t, a in _ARITY.items()}
        self.index = {}
        for t in _ARITY:
            for i, w in enumerate(sorted(w for (w, tt) in vocab if tt == t)):
                self.index[(w, t)] = i

    def _states(self, t, words):
        ids = np.array([self.index[(w, t)] for w in words])
        st = self.ansatz[t].states(ids)
        return st.reshape((len(words),) + (self.dim,) * _ARITY[t])

    def _noun_phrase(self, np_words):
        noun = self._states("n", [np_words[-1][0]])[0]
        for adj, _ in reversed(np_words[:-1]):
            noun = einsum("ij,j->i", self._states("adj", [adj])[0], noun)            # cup between adj n^l and n
        return noun

    def forward(self, sentences):
        outs = []
        for s in sentences:
            subj, verb, obj = s
            subj_v = self._noun_phrase(subj)
            if verb[1] == "iv":
                sent = einsum("i,is->s", subj_v, self._states("iv", [verb[0]])[0])
            else:
                sent = einsum("i,isj,j->s", subj_v, self._states("tv", [verb[0]])[0], self._noun_phrase(obj))
            p = sent ** 2
            outs.append(p)
        p = stack(outs, 0)[:, :self.k]
        return (p / p.sum(-1, keepdims=True) + 1e-12).log()


class DisCoCatClassifier(_TextClassifierBase):
    """Input: typed sentences, e.g. [("skilful", "adj"), ("chef", "n"), ("prepares", "tv"), ("sauce", "n")].
    Types: n, adj (n.n^l), iv (n^r.s), tv (n^r.s.n^l). One sentence wire of ``qubits_per_wire`` qubits is
    measured; up to 2**qubits_per_wire classes."""

    def __init__(self, qubits_per_wire=1, layers=2, epochs=60, lr=0.05, batch_size=16, seed=0):
        self.qubits_per_wire, self.layers = qubits_per_wire, layers
        self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed
        self.min_count = 1

    def _vocab(self, sentences):
        self.vocab_ = sorted({(w, t) for s in sentences for (w, t) in s})

    def _build(self, k):
        return _DisCoCatNet(self.vocab_, self.qubits_per_wire, self.layers, k, np.random.default_rng(self.seed))

    def _encode(self, sentences):
        known = set(self.vocab_)
        for s in sentences:
            for wt in s:
                if wt not in known:
                    raise ValueError(f"unknown word/type {wt}: DisCoCat needs every word seen in training")
        out = np.empty(len(sentences), dtype=object)
        out[:] = [_parse(s) for s in sentences]
        return out


# =============================================================================== quantum word embeddings
class QuantumWordEmbeddings:
    """Words as ``n_qubits``-qubit states; skip-gram with negative sampling on the fidelity F = |<w|c>|^2:
    loss = -log sigma(a F(w, c) + b) - sum_neg log sigma(-(a F(w, n) + b))."""

    def __init__(self, n_qubits=4, layers=2, window=3, negative=3, min_count=3, epochs=5, lr=0.05,
                 batch_size=256, seed=0):
        self.n_qubits, self.layers, self.window, self.negative = n_qubits, layers, window, negative
        self.min_count, self.epochs, self.lr, self.batch_size, self.seed = min_count, epochs, lr, batch_size, seed

    def fit(self, texts):
        rng = np.random.default_rng(self.seed)
        docs = [tokenize(t) for t in texts]
        counts = Counter(w for d in docs for w in d)
        self.words_ = sorted((w for w, c in counts.items() if c >= self.min_count), key=lambda w: (-counts[w], w))
        self.index_ = {w: i for i, w in enumerate(self.words_)}
        v = len(self.words_)
        centre, ctx = [], []
        for d in docs:
            ids = [self.index_[w] for w in d if w in self.index_]
            for i, w in enumerate(ids):
                for j in ids[max(0, i - self.window):i] + ids[i + 1:i + 1 + self.window]:
                    centre.append(w)
                    ctx.append(j)
        centre, ctx = np.array(centre), np.array(ctx)
        noise = np.array([counts[w] for w in self.words_], dtype=float) ** 0.75
        noise /= noise.sum()
        self.states_net_ = _WordStates(v, self.n_qubits, self.layers, rng)
        self.scale = Parameter(np.array([4.0, -2.0]))
        opt = Adam(self.states_net_.parameters() + [self.scale], self.lr)
        self.history_ = []
        for _ in range(self.epochs):
            order = rng.permutation(len(centre))
            losses = []
            for s in range(0, len(order), self.batch_size):
                idx = order[s:s + self.batch_size]
                neg = rng.choice(v, (len(idx), self.negative), p=noise)
                ids = np.concatenate([centre[idx], ctx[idx], neg.ravel()])
                uniq, inv = np.unique(ids, return_inverse=True)
                st = self.states_net_.states(uniq)
                b = len(idx)
                wc, wo, wn = st[inv[:b]], st[inv[b:2 * b]], st[inv[2 * b:]].reshape(b, self.negative, -1)
                f_pos = (wc * wo).sum(-1) ** 2
                f_neg = (wn * wc.reshape(b, 1, -1)).sum(-1) ** 2
                a, c = self.scale[0], self.scale[1]
                loss = -((f_pos * a + c).sigmoid() + 1e-9).log().mean() - \
                    ((-(f_neg * a + c)).sigmoid() + 1e-9).log().sum(-1).mean()
                self.states_net_.zero_grad()
                self.scale.grad = None
                loss.backward()
                opt.step()
                losses.append(loss.item())
            self.history_.append(float(np.mean(losses)))
        with no_grad():
            self.vectors_ = self.states_net_.states().data
        return self

    def similarity(self, a, b):
        return float((self.vectors_[self.index_[a]] @ self.vectors_[self.index_[b]]) ** 2)

    def most_similar(self, word, top_n=8):
        f = (self.vectors_ @ self.vectors_[self.index_[word]]) ** 2
        return [(self.words_[i], float(f[i])) for i in np.argsort(-f) if self.words_[i] != word][:top_n]

    def document_vectors(self, texts):
        """Mean of the word states' measurement probabilities (a point in the probability simplex)."""
        out = np.zeros((len(texts), self.vectors_.shape[1]))
        for k, t in enumerate(texts):
            ids = [self.index_[w] for w in tokenize(t) if w in self.index_]
            if ids:
                out[k] = (self.vectors_[ids] ** 2).mean(0)
        return out


# =============================================================================== MC-style dataset
def make_mc_dataset(n_sentences=130, seed=0):
    """Synthetic meaning-classification sentences with pregroup types: food (label 0) vs IT (label 1).
    Patterns: [adj] noun verb [adj] noun, and noun iv. Returns (sentences, labels, plain strings)."""
    rng = np.random.default_rng(seed)
    people = ["chef", "cook", "programmer", "engineer", "student", "woman", "man"]
    adjs = ["skilful", "tired", "clever", "young"]
    topics = {0: {"tv": ["cooks", "prepares", "bakes", "tastes"], "n": ["meal", "sauce", "dinner", "bread", "soup"],
                  "adj": ["tasty", "fresh", "warm"], "iv": ["eats", "dines"]},
              1: {"tv": ["runs", "debugs", "writes", "compiles"], "n": ["program", "application", "code", "software", "script"],
                  "adj": ["useful", "buggy", "fast"], "iv": ["codes", "deploys"]}}
    sents, labels = [], []
    while len(sents) < n_sentences:
        lab = int(rng.integers(2))
        t = topics[lab]
        subj = ([(rng.choice(adjs), "adj")] if rng.random() < 0.3 else []) + [(rng.choice(people), "n")]
        if rng.random() < 0.2:
            s = subj + [(rng.choice(t["iv"]), "iv")]
        else:
            obj = ([(rng.choice(t["adj"]), "adj")] if rng.random() < 0.5 else []) + [(rng.choice(t["n"]), "n")]
            s = subj + [(rng.choice(t["tv"]), "tv")] + obj
        s = [(str(w), ty) for w, ty in s]
        if s not in sents:
            sents.append(s)
            labels.append(lab)
    return sents, np.array(labels), [" ".join(w for w, _ in s) for s in sents]
