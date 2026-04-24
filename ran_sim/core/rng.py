import random as pyrandom
import numpy as np


class RNGManager:

    def __init__(self, seed: int):
        self.seed = seed

        self._py_rng = pyrandom.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    # =========================
    # Basic random
    # =========================
    def random(self) -> float:
        return self._py_rng.random()

    def uniform(self, a: float, b: float) -> float:
        return self._py_rng.uniform(a, b)

    def randint(self, a: int, b: int) -> int:
        return self._py_rng.randint(a, b)

    def choice(self, seq):
        return self._py_rng.choice(seq)

    def normal(self, mean: float = 0.0, std: float = 1.0) -> float:
        return self._py_rng.gauss(mean, std)

    # =========================
    # NumPy RNG
    # =========================
    def np(self):
        return self._np_rng

    # =========================
    # Derived RNG
    # =========================
    def derive_seed(self, *args: int) -> int:
        h = self.seed
        for v in args:
            h = (h * 31 + int(v)) % (2**32 - 1)
        return h

    def get_rng_for(self, *args: int) -> pyrandom.Random:
        seed = self.derive_seed(*args)
        return pyrandom.Random(seed)

    def get_np_rng_for(self, *args: int):
        seed = self.derive_seed(*args)
        return np.random.default_rng(seed)