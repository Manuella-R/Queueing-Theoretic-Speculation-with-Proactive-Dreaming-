import numpy as np

class EpsilonGreedyBandit:
    def __init__(self, n_actions: int, eps: float=0.1, seed: int=0):
        self.n = n_actions
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(self.n, dtype=int)
        self.values = np.zeros(self.n, dtype=float)

    def select(self):
        if self.rng.random() < self.eps:
            return int(self.rng.integers(self.n))
        return int(np.argmax(self.values))

    def update(self, a: int, reward: float):
        self.counts[a] += 1
        n = self.counts[a]
        q = self.values[a]
        self.values[a] = q + (reward - q)/n
