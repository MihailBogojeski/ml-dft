import numpy as np


class RBFKernel():
    def __init__(self, gamma=0):
        self.gamma = gamma

    def apply(self, x1, x2, gamma=None):
        if gamma is None:
            gamma = self.gamma
        return np.exp(-np.linalg.norm(x1 - x2)**2 / (2 * gamma**2))

    def apply_to_dist(self, dist, gamma=None):
        if gamma is None:
            gamma = self.gamma
        K = -dist**2 / (2 * gamma**2)
        K = np.exp(K)
        return K


class MaternKernel():

    def __init__(self, n=1, gamma=1):
        self.n = n
        self.gamma = gamma

    def apply(self, x1, x2, gamma=None, n=None):
        if gamma is None:
            gamma = self.gamma
        if n is None:
            n = self.n

        v = n + 1 / 2

        norm_ab = np.linalg.norm(x1 - x2)

        k = np.exp(-np.sqrt(2 * v) * norm_ab / gamma)
        k *= np.math.factorial(n + 1) / np.math.factorial(2 * n + 1)

        s = 0
        for i in range(0, n + 1):
            s += np.math.factorial(n + i) / (np.math.factorial(i) * np.math.factorial(n - i)) *\
                np.power(np.sqrt(8 * v) * norm_ab / gamma, n - i)
        k *= s

        return k

    def apply_to_dist(self, dist, gamma=None, n=None):
        if gamma is None:
            gamma = self.gamma
        if n is None:
            n = self.n

        v = n + 1 / 2

        K = -np.sqrt(2 * v) * dist / gamma
        K = np.exp(K)
        K *= np.math.factorial(n + 1) / np.math.factorial(2 * n + 1)

        s = 0
        for i in range(0, n + 1):
            s += np.math.factorial(n + i) / (np.math.factorial(i) * np.math.factorial(n - i)) *\
                np.power(np.sqrt(8 * v) * dist / gamma, n - i)
        K *= s

        return K
