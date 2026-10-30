"""Layers, losses, optimisers and a generic training loop."""
from __future__ import annotations

import time

import numpy as np

from .autograd import Tensor, concat, no_grad
from .quantum.circuit import Circuit, Observable
from .quantum.functions import expval

__all__ = ["Module", "Parameter", "Linear", "Sequential", "Activation", "QuantumLayer", "Adam", "SGD",
           "mse_loss", "cross_entropy", "binary_cross_entropy", "gaussian_nll", "train"]


def Parameter(data) -> Tensor:
    return Tensor(np.array(data, dtype=float), requires_grad=True)


class Module:
    """Base class: collects every trainable Tensor stored as an attribute (also inside lists/dicts)."""

    training = True

    def parameters(self) -> list[Tensor]:
        found, seen = [], set()

        def collect(obj):
            if isinstance(obj, Tensor):
                if obj.requires_grad and id(obj) not in seen:
                    seen.add(id(obj))
                    found.append(obj)
            elif isinstance(obj, Module):
                for v in vars(obj).values():
                    collect(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    collect(v)
            elif isinstance(obj, dict):
                for v in obj.values():
                    collect(v)

        for v in vars(self).values():
            collect(v)
        return found

    def n_parameters(self) -> int:
        return int(sum(p.data.size for p in self.parameters()))

    def set_mode(self, training: bool):
        """Switch this module and all sub-modules between training and evaluation mode."""
        def walk(obj):
            if isinstance(obj, Module):
                obj.training = training
                for v in vars(obj).values():
                    walk(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj:
                    walk(v)
            elif isinstance(obj, dict):
                for v in obj.values():
                    walk(v)
        walk(self)
        return self

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def state_dict(self) -> list[np.ndarray]:
        return [p.data.copy() for p in self.parameters()]

    def load_state_dict(self, state):
        for p, s in zip(self.parameters(), state):
            p.data = s.copy()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Linear(Module):
    def __init__(self, n_in: int, n_out: int, bias: bool = True, rng=None):
        rng = rng or np.random.default_rng()
        self.w = Parameter(rng.normal(0, np.sqrt(1.0 / n_in), (n_in, n_out)))
        self.b = Parameter(np.zeros(n_out)) if bias else None

    def forward(self, x):
        out = x @ self.w
        return out + self.b if self.b is not None else out


class Activation(Module):
    def __init__(self, name: str):
        self.name = name

    def forward(self, x):
        return getattr(x, self.name)()


class Sequential(Module):
    def __init__(self, *layers):
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class QuantumLayer(Module):
    """Wraps a :class:`Circuit` as a layer: inputs (B, n_inputs) -> expectation values (B, n_obs).

    ``input_transform`` maps raw features to angles inside the graph: ``"pi"`` (x * pi),
    ``"arctan"`` (arctan x), ``"tanh_pi"`` (pi * tanh x) or ``None``.
    """

    def __init__(self, circuit: Circuit, observables=None, input_transform: str | None = "arctan",
                 init_scale: float = np.pi, rng=None, shots: int | None = None):
        rng = rng or np.random.default_rng()
        self.circuit = circuit
        self.observables = observables or [Observable(f"Z{q}") for q in range(circuit.n_qubits)]
        self.input_transform = input_transform
        self.weights = Parameter(rng.uniform(-init_scale, init_scale, circuit.n_weights))
        self.shots = shots

    def angles(self, x):
        if self.input_transform is None:
            return x
        if self.input_transform == "pi":
            return x * np.pi
        if self.input_transform == "arctan":
            return x.arctan()
        if self.input_transform == "tanh_pi":
            return x.tanh() * np.pi
        raise ValueError(self.input_transform)

    def forward(self, x=None, initial_state=None):
        angles = None if x is None else self.angles(x)
        return expval(self.circuit, self.observables, angles, self.weights, initial_state,
                      shots=None if self.training else self.shots)


# ----------------------------------------------------------------------------- losses
def mse_loss(pred: Tensor, target) -> Tensor:
    return ((pred - target) ** 2).mean()


def cross_entropy(logits: Tensor, labels) -> Tensor:
    labels = np.asarray(labels, dtype=int)
    logp = logits.log_softmax(-1)
    onehot = np.eye(logits.shape[-1])[labels]
    return -(logp * onehot).sum(-1).mean()


def binary_cross_entropy(prob: Tensor, target, eps: float = 1e-9) -> Tensor:
    target = np.asarray(target, dtype=float)
    return -(target * (prob + eps).log() + (1 - target) * (1 - prob + eps).log()).mean()


def gaussian_nll(mu: Tensor, sigma: Tensor, target) -> Tensor:
    return (sigma.log() + 0.5 * ((target - mu) / sigma) ** 2).mean() + 0.5 * np.log(2 * np.pi)


# ------------------------------------------------------------------------- optimisers
class SGD:
    def __init__(self, params, lr: float = 0.01, momentum: float = 0.0):
        self.params, self.lr, self.momentum = list(params), lr, momentum
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.v):
            if p.grad is None:
                continue
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v


class Adam:
    def __init__(self, params, lr: float = 0.01, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        self.params, self.lr, self.b1, self.b2, self.eps, self.wd = list(params), lr, *betas, eps, weight_decay
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]
        self.t = 0

    def step(self):
        self.t += 1
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad + self.wd * p.data
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g ** 2
            p.data -= self.lr * (m / (1 - self.b1 ** self.t)) / (np.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)


# ------------------------------------------------------------------------ training loop
def train(model: Module, loss_fn, data, epochs: int = 50, batch_size: int = 32, lr: float = 0.01,
          val_data=None, patience: int | None = None, optimizer: str = "adam", seed: int = 0,
          verbose: bool = False, weight_decay: float = 0.0):
    """Mini-batch training. ``data`` is a tuple of arrays sharing the first axis;
    ``loss_fn(model, *batch) -> Tensor``. Returns the history dict."""
    rng = np.random.default_rng(seed)
    params = model.parameters()
    opt = Adam(params, lr, weight_decay=weight_decay) if optimizer == "adam" else SGD(params, lr)
    n = len(data[0])
    history = {"loss": [], "val_loss": []}
    best, best_state, wait, t0 = np.inf, None, 0, time.time()
    for epoch in range(epochs):
        model.set_mode(True)
        order = rng.permutation(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            model.zero_grad()
            loss = loss_fn(model, *(d[idx] for d in data))
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["loss"].append(total / n)
        if val_data is not None:
            model.set_mode(False)
            with no_grad():
                val = loss_fn(model, *val_data).item()
            history["val_loss"].append(val)
            if val < best - 1e-6:
                best, best_state, wait = val, model.state_dict(), 0
            else:
                wait += 1
                if patience is not None and wait >= patience:
                    break
        if verbose:
            msg = f"epoch {epoch + 1:3d}  loss {history['loss'][-1]:.4f}"
            if val_data is not None:
                msg += f"  val {history['val_loss'][-1]:.4f}"
            print(msg)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.set_mode(False)
    history["time"] = time.time() - t0
    return history


def _concat(xs, axis=-1):
    return concat(xs, axis)
