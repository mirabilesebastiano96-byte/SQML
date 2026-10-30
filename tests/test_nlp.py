import numpy as np

from skqml import Tensor, einsum
from skqml.datasets import load_arxiv_recent
from skqml.embeddings import Word2Vec, embed_documents
from skqml.nn import cross_entropy
from skqml.qnlp import (DisCoCatClassifier, SpiderQNLPClassifier, WordSequenceQNLPClassifier, _DisCoCatNet,
                        _parse, _SeqNet, _SpiderNet, make_mc_dataset)

rng = np.random.default_rng(5)


def _check(net, loss_fn, n=3, eps=1e-6):
    net.zero_grad()
    loss_fn().backward()
    for p in net.parameters():
        g = p.grad.copy()
        for idx in list(np.ndindex(p.shape))[:n]:
            if g[idx] == 0:
                continue
            old = p.data[idx]
            p.data[idx] = old + eps
            a = loss_fn().item()
            p.data[idx] = old - eps
            b = loss_fn().item()
            p.data[idx] = old
            assert abs((a - b) / (2 * eps) - g[idx]) < 1e-5 * max(1, abs(g[idx]))


def test_einsum_gradient():
    a, b = Tensor(rng.normal(size=(3, 4)), requires_grad=True), Tensor(rng.normal(size=(4, 2)), requires_grad=True)
    (einsum("ij,jk->ik", a, b) ** 2).sum().backward()
    assert np.allclose(a.grad, 2 * (a.data @ b.data) @ b.data.T)


def test_qnlp_gradients():
    y = np.array([0, 1, 1])
    seq = _SeqNet(5, 2, 3, 2, rng)
    toks = np.array([[0, 1, 5], [2, 3, 4], [4, 5, 5]])
    _check(seq, lambda: cross_entropy(seq(toks), y))
    sp = _SpiderNet(5, 2, 2, 2, rng)
    toks = np.array([[0, 1, -1], [2, 3, 4], [4, -1, -1]])
    _check(sp.words, lambda: cross_entropy(sp(toks), y))
    sents, labels, _ = make_mc_dataset(10)
    vocab = sorted({wt for s in sents for wt in s})
    dc = _DisCoCatNet(vocab, 1, 2, 2, rng)
    enc = np.empty(4, dtype=object)
    enc[:] = [_parse(s) for s in sents[:4]]
    _check(dc, lambda: cross_entropy(dc(enc), labels[:4]))


def test_qnlp_classifiers_learn_mc():
    sents, y, plain = make_mc_dataset(110)
    assert DisCoCatClassifier(epochs=40).fit(sents[:80], y[:80]).score(sents[80:], y[80:]) > 0.85
    assert SpiderQNLPClassifier(n_qubits=2, min_count=1, epochs=30).fit(plain[:80], y[:80]).score(plain[80:], y[80:]) > 0.85
    assert WordSequenceQNLPClassifier(3, 6, min_count=1, epochs=30).fit(plain[:80], y[:80]).score(plain[80:], y[80:]) > 0.85


def test_arxiv_and_embeddings():
    d = load_arxiv_recent()
    assert len(d.X) == 500 and np.bincount(d.y).tolist() == [100] * 5
    w2v = Word2Vec(dim=30, epochs=3).fit(d.X[:200])
    assert w2v.vectors_.shape[1] == 30 and w2v.history_[-1] < w2v.history_[0]
    x = embed_documents(d.X[:100], "tfidf_lsa", n_components=4)
    assert x.shape == (100, 4) and np.allclose(np.linalg.norm(x, axis=1), 1)
