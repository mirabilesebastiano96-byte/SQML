"""Quantum reinforcement learning.

* VQCPolicy + QuantumREINFORCE (Jerbi et al., NeurIPS 2021): data re-uploading policy with trainable input
  scaling, Born-rule / softmax action probabilities, trained with the REINFORCE policy gradient.
* QuantumDQN (Skolik, Jerbi & Dunjko, Quantum 2022): Q-values read out as scaled expectation values,
  trained with experience replay and a target network.
* CartPole: a minimal, dependency-free implementation of the classic control task.
"""
from __future__ import annotations

import numpy as np

from ..autograd import Tensor, no_grad
from ..nn import Adam, Module, Parameter
from ..quantum.circuit import Circuit, Observable
from ..quantum.functions import expval

__all__ = ["CartPole", "VQCPolicy", "QuantumREINFORCE", "QuantumDQN"]


class CartPole:
    """Classic cart-pole balancing (Barto, Sutton & Anderson 1983). reset() -> state; step(a) -> state, r, done."""

    def __init__(self, max_steps=200, seed=None):
        self.max_steps, self.rng = max_steps, np.random.default_rng(seed)
        self.n_states, self.n_actions = 4, 2

    def reset(self):
        self.state, self.t = self.rng.uniform(-0.05, 0.05, 4), 0
        return self.state.copy()

    def step(self, action):
        x, v, th, w = self.state
        force = 10.0 if action == 1 else -10.0
        g, mc, mp, length, dt = 9.8, 1.0, 0.1, 0.5, 0.02
        tmp = (force + mp * length * w ** 2 * np.sin(th)) / (mc + mp)
        acc_th = (g * np.sin(th) - np.cos(th) * tmp) / (length * (4 / 3 - mp * np.cos(th) ** 2 / (mc + mp)))
        acc_x = tmp - mp * length * acc_th * np.cos(th) / (mc + mp)
        self.state = np.array([x + dt * v, v + dt * acc_x, th + dt * w, w + dt * acc_th])
        self.t += 1
        done = abs(self.state[0]) > 2.4 or abs(self.state[2]) > 0.2095 or self.t >= self.max_steps
        return self.state.copy(), 1.0, done


class VQCPolicy(Module):
    """Re-uploading circuit: per layer RY(lambda * s_q) RZ(lambda' * s_q), trainable RY/RZ, CZ ring.
    Action scores are beta * <Z...Z> parities, turned into probabilities with a softmax."""

    def __init__(self, n_states=4, n_actions=2, layers=4, beta=1.0, seed=0):
        rng = np.random.default_rng(seed)
        qc = Circuit(n_states)
        for _ in range(layers):
            x = qc.inputs(2 * n_states)
            for q in range(n_states):
                qc.ry(q, x[q]).rz(q, x[n_states + q])
            for q in range(n_states):
                w = qc.weights(2)
                qc.ry(q, w[0]).rz(q, w[1])
            for q in range(n_states):
                qc.cz(q, (q + 1) % n_states) if n_states > 1 else None
        self.circuit, self.layers, self.n_states = qc, layers, n_states
        self.weights = Parameter(rng.uniform(0, np.pi, qc.n_weights))
        self.input_scale = Parameter(np.ones(2 * n_states * layers))
        half = n_states // 2
        self.obs = [Observable(" ".join(f"Z{q}" for q in range(half))),
                    Observable(" ".join(f"Z{q}" for q in range(half, n_states)))][:n_actions]
        self.beta = Parameter(np.full(len(self.obs), beta))

    def scores(self, states):
        s = Tensor(np.tile(np.arctan(np.asarray(states, dtype=float)), (1, 2 * self.layers)))
        return expval(self.circuit, self.obs, s * self.input_scale, self.weights) * self.beta

    def forward(self, states):
        return self.scores(states).softmax(-1)


class QuantumREINFORCE:
    def __init__(self, policy: VQCPolicy | None = None, gamma=0.99, lr=0.05, seed=0):
        self.policy, self.gamma, self.lr, self.rng = policy or VQCPolicy(seed=seed), gamma, lr, np.random.default_rng(seed)

    def act(self, state, greedy=False):
        with no_grad():
            p = self.policy(state[None])[0].data
        return int(np.argmax(p)) if greedy else int(self.rng.choice(len(p), p=p / p.sum()))

    def fit(self, env, episodes=100, batch_episodes=5, verbose=False):
        opt = Adam(self.policy.parameters(), self.lr)
        self.history_ = []
        for start in range(0, episodes, batch_episodes):
            states, actions, returns = [], [], []
            for _ in range(batch_episodes):
                s, done, rewards = env.reset(), False, []
                while not done:
                    a = self.act(s)
                    states.append(s)
                    actions.append(a)
                    s, r, done = env.step(a)
                    rewards.append(r)
                g, ret = 0.0, []
                for r in reversed(rewards):
                    g = r + self.gamma * g
                    ret.append(g)
                ret = np.array(ret[::-1])
                returns.append((ret - ret.mean()) / (ret.std() + 1e-8))
                self.history_.append(len(rewards))
            self.policy.zero_grad()
            probs = self.policy(np.array(states))
            onehot = np.eye(probs.shape[1])[actions]
            logp = ((probs * onehot).sum(-1) + 1e-9).log()
            loss = -(logp * np.concatenate(returns)).sum() / batch_episodes
            loss.backward()
            opt.step()
            if verbose:
                print(f"episodes {start + batch_episodes:4d}  mean length {np.mean(self.history_[-batch_episodes:]):.1f}")
        return self


class QuantumDQN:
    """Q(s, a) = beta_a * <O_a> from a VQCPolicy-style circuit; epsilon-greedy, replay buffer, target network."""

    def __init__(self, n_states=4, n_actions=2, layers=4, gamma=0.99, lr=0.01, batch=16, buffer=5000,
                 target_every=10, seed=0):
        self.q = VQCPolicy(n_states, n_actions, layers, beta=50.0, seed=seed)
        self.target = VQCPolicy(n_states, n_actions, layers, beta=50.0, seed=seed)
        self._sync()
        self.gamma, self.lr, self.batch, self.buffer_size, self.target_every = gamma, lr, batch, buffer, target_every
        self.rng = np.random.default_rng(seed)
        self.buffer = []

    def _sync(self):
        self.target.load_state_dict(self.q.state_dict())

    def act(self, state, epsilon=0.0):
        if self.rng.random() < epsilon:
            return int(self.rng.integers(len(self.q.obs)))
        with no_grad():
            return int(np.argmax(self.q.scores(state[None]).data[0]))

    def fit(self, env, episodes=100, eps_start=1.0, eps_end=0.05, eps_decay=0.97, verbose=False):
        opt = Adam(self.q.parameters(), self.lr)
        eps, self.history_ = eps_start, []
        for ep in range(episodes):
            s, done, steps = env.reset(), False, 0
            while not done:
                a = self.act(s, eps)
                s2, r, done = env.step(a)
                self.buffer = (self.buffer + [(s, a, r, s2, float(done))])[-self.buffer_size:]
                s, steps = s2, steps + 1
                if len(self.buffer) >= self.batch:
                    idx = self.rng.choice(len(self.buffer), self.batch, replace=False)
                    bs, ba, br, bs2, bd = (np.array(v) for v in zip(*[self.buffer[i] for i in idx]))
                    with no_grad():
                        target = br + self.gamma * (1 - bd) * self.target.scores(bs2).data.max(1)
                    self.q.zero_grad()
                    q = (self.q.scores(bs) * np.eye(len(self.q.obs))[ba]).sum(-1)
                    loss = ((q - target) ** 2).mean()
                    loss.backward()
                    opt.step()
            eps = max(eps_end, eps * eps_decay)
            if ep % self.target_every == 0:
                self._sync()
            self.history_.append(steps)
            if verbose:
                print(f"episode {ep:4d}  length {steps}")
        return self
