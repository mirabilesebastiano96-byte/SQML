"""A tiny reverse-mode automatic differentiation engine on NumPy arrays.

Every model in skqml is built from :class:`Tensor` operations, so gradients flow
through classical layers and quantum circuits alike. The design follows the
classic "micrograd" idea, extended to n-dimensional arrays with broadcasting.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Tensor", "tensor", "concat", "stack", "einsum", "no_grad", "is_grad_enabled"]

_GRAD_ENABLED = True


class no_grad:
    """Context manager that disables graph construction (faster inference)."""

    def __enter__(self):
        global _GRAD_ENABLED
        self._prev, _GRAD_ENABLED = _GRAD_ENABLED, False

    def __exit__(self, *exc):
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._prev


def is_grad_enabled() -> bool:
    return _GRAD_ENABLED


def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum ``grad`` over the axes that were broadcast to produce it."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


def _as_tensor(x) -> "Tensor":
    return x if isinstance(x, Tensor) else Tensor(x)


class Tensor:
    """An array that remembers how it was computed, so it can be differentiated."""

    __array_priority__ = 1000  # make NumPy defer to Tensor operators

    def __init__(self, data, requires_grad: bool = False, _parents=(), _backward=None):
        self.data = np.asarray(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = None
        track = _GRAD_ENABLED and any(p.requires_grad for p in _parents)
        self._parents = tuple(_parents) if track else ()
        self._backward = _backward if track else None
        if track:
            self.requires_grad = True

    # ------------------------------------------------------------------ basics
    shape = property(lambda self: self.data.shape)
    ndim = property(lambda self: self.data.ndim)
    T = property(lambda self: self.transpose())

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"

    def numpy(self) -> np.ndarray:
        return self.data

    def item(self) -> float:
        return float(self.data)

    def detach(self) -> "Tensor":
        return Tensor(self.data.copy())

    def zero_grad(self):
        self.grad = None

    def _accumulate(self, g):
        if self.requires_grad:
            self.grad = g if self.grad is None else self.grad + g

    def backward(self, grad=None):
        """Back-propagate from this tensor through the whole graph."""
        order, seen = [], set()

        def visit(t):
            if id(t) not in seen:
                seen.add(id(t))
                for p in t._parents:
                    visit(p)
                order.append(t)

        visit(self)
        self.grad = np.ones_like(self.data) if grad is None else np.asarray(grad, dtype=float)
        for t in reversed(order):
            if t._backward is not None and t.grad is not None:
                t._backward(t.grad)

    @staticmethod
    def _make(data, parents, backward):
        return Tensor(data, _parents=parents, _backward=backward)

    # -------------------------------------------------------------- arithmetic
    def __add__(self, other):
        other = _as_tensor(other)

        def bw(g):
            self._accumulate(_unbroadcast(g, self.shape))
            other._accumulate(_unbroadcast(g, other.shape))

        return self._make(self.data + other.data, (self, other), bw)

    __radd__ = __add__

    def __neg__(self):
        return self._make(-self.data, (self,), lambda g: self._accumulate(-g))

    def __sub__(self, other):
        return self + (-_as_tensor(other))

    def __rsub__(self, other):
        return _as_tensor(other) + (-self)

    def __mul__(self, other):
        other = _as_tensor(other)

        def bw(g):
            self._accumulate(_unbroadcast(g * other.data, self.shape))
            other._accumulate(_unbroadcast(g * self.data, other.shape))

        return self._make(self.data * other.data, (self, other), bw)

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = _as_tensor(other)

        def bw(g):
            self._accumulate(_unbroadcast(g / other.data, self.shape))
            other._accumulate(_unbroadcast(-g * self.data / other.data ** 2, other.shape))

        return self._make(self.data / other.data, (self, other), bw)

    def __rtruediv__(self, other):
        return _as_tensor(other) / self

    def __pow__(self, k: float):
        def bw(g):
            self._accumulate(g * k * self.data ** (k - 1))

        return self._make(self.data ** k, (self,), bw)

    def __matmul__(self, other):
        other = _as_tensor(other)

        def bw(g):
            a, b = self.data, other.data
            if b.ndim == 1:
                self._accumulate(_unbroadcast(np.multiply.outer(g, b), self.shape))
                other._accumulate((a.reshape(-1, a.shape[-1]).T @ g.reshape(-1)))
                return
            ga = g @ np.swapaxes(b, -1, -2)
            gb = np.swapaxes(a, -1, -2) @ g if a.ndim > 1 else np.multiply.outer(a, g)
            self._accumulate(_unbroadcast(ga, self.shape))
            other._accumulate(_unbroadcast(gb, other.shape))

        return self._make(self.data @ other.data, (self, other), bw)

    def __rmatmul__(self, other):
        return _as_tensor(other) @ self

    # ------------------------------------------------------------ elementwise
    def _unary(self, value, derivative):
        return self._make(value, (self,), lambda g: self._accumulate(g * derivative))

    def exp(self):
        e = np.exp(self.data)
        return self._unary(e, e)

    def log(self):
        return self._unary(np.log(self.data), 1.0 / self.data)

    def sqrt(self):
        s = np.sqrt(self.data)
        return self._unary(s, 0.5 / s)

    def tanh(self):
        t = np.tanh(self.data)
        return self._unary(t, 1 - t ** 2)

    def sigmoid(self):
        s = 0.5 * (1 + np.tanh(0.5 * self.data))
        return self._unary(s, s * (1 - s))

    def relu(self):
        return self._unary(np.maximum(self.data, 0), (self.data > 0).astype(float))

    def softplus(self):
        return self._unary(np.logaddexp(0, self.data), 0.5 * (1 + np.tanh(0.5 * self.data)))

    def arctan(self):
        return self._unary(np.arctan(self.data), 1 / (1 + self.data ** 2))

    def sin(self):
        return self._unary(np.sin(self.data), np.cos(self.data))

    def cos(self):
        return self._unary(np.cos(self.data), -np.sin(self.data))

    def abs(self):
        return self._unary(np.abs(self.data), np.sign(self.data))

    def clip(self, lo, hi):
        inside = ((self.data >= lo) & (self.data <= hi)).astype(float)
        return self._unary(np.clip(self.data, lo, hi), inside)

    # -------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims=False):
        def bw(g):
            if axis is not None and not keepdims:
                g = np.expand_dims(g, axis)
            self._accumulate(np.broadcast_to(g, self.shape).copy())

        return self._make(self.data.sum(axis=axis, keepdims=keepdims), (self,), bw)

    def mean(self, axis=None, keepdims=False):
        n = self.data.size if axis is None else np.prod([self.shape[a] for a in np.atleast_1d(axis)])
        return self.sum(axis, keepdims) / n

    def logsumexp(self, axis=-1):
        m = self.data.max(axis=axis, keepdims=True)
        e = np.exp(self.data - m)
        s = e.sum(axis=axis, keepdims=True)
        out = (m + np.log(s)).squeeze(axis)

        def bw(g):
            self._accumulate(np.expand_dims(g, axis) * e / s)

        return self._make(out, (self,), bw)

    def softmax(self, axis=-1):
        return (self - self.logsumexp(axis).unsqueeze(axis)).exp()

    def log_softmax(self, axis=-1):
        return self - self.logsumexp(axis).unsqueeze(axis)

    # ----------------------------------------------------------------- shaping
    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
        return self._make(self.data.reshape(shape), (self,), lambda g: self._accumulate(g.reshape(self.shape)))

    def unsqueeze(self, axis):
        return self._make(np.expand_dims(self.data, axis), (self,),
                          lambda g: self._accumulate(g.reshape(self.shape)))

    def transpose(self, *axes):
        axes = axes or tuple(reversed(range(self.ndim)))
        inverse = np.argsort(axes)
        return self._make(self.data.transpose(axes), (self,), lambda g: self._accumulate(g.transpose(inverse)))

    def __getitem__(self, index):
        def bw(g):
            full = np.zeros_like(self.data)
            np.add.at(full, index, g)
            self._accumulate(full)

        return self._make(self.data[index], (self,), bw)


def tensor(data, requires_grad: bool = False) -> Tensor:
    return Tensor(data, requires_grad=requires_grad)


def concat(tensors, axis=-1) -> Tensor:
    tensors = [_as_tensor(t) for t in tensors]
    sizes = np.cumsum([t.shape[axis] for t in tensors])[:-1]

    def bw(g):
        for t, part in zip(tensors, np.split(g, sizes, axis=axis)):
            t._accumulate(part)

    return Tensor._make(np.concatenate([t.data for t in tensors], axis=axis), tuple(tensors), bw)


def stack(tensors, axis=0) -> Tensor:
    tensors = [_as_tensor(t) for t in tensors]

    def bw(g):
        for i, t in enumerate(tensors):
            t._accumulate(np.take(g, i, axis=axis))

    return Tensor._make(np.stack([t.data for t in tensors], axis=axis), tuple(tensors), bw)


def einsum(subscripts: str, *operands) -> Tensor:
    """Differentiable ``np.einsum`` (explicit output, e.g. "ij,jk->ik"; no repeated index inside one operand)."""
    operands = [_as_tensor(t) for t in operands]
    inputs, output = subscripts.replace(" ", "").split("->")
    in_subs = inputs.split(",")
    out = np.einsum(subscripts, *[t.data for t in operands])
    sizes = {}
    for sub, t in zip(in_subs, operands):
        sizes.update(dict(zip(sub, t.shape)))

    def bw(g):
        for i, t in enumerate(operands):
            if not t.requires_grad:
                continue
            others = [(sub, o.data) for k, (sub, o) in enumerate(zip(in_subs, operands)) if k != i]
            have = set(output) | set("".join(sub for sub, _ in others))
            missing = "".join(c for c in in_subs[i] if c not in have)
            extra = [(missing, np.ones([sizes[c] for c in missing]))] if missing else []
            expr = ",".join([output] + [sub for sub, _ in others + extra]) + "->" + in_subs[i]
            t._accumulate(np.einsum(expr, g, *[d for _, d in others + extra]))

    return Tensor._make(out, tuple(operands), bw)
