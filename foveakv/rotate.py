"""Randomized Hadamard rotation R = H_d · diag(s), s in {±1} (seeded), H_d orthonormal Sylvester.
Row-vector convention: rotate(x) = x @ R.T, unrotate(y) = y @ R."""
import numpy as np


def hadamard(d):
    assert d & (d - 1) == 0, "d must be a power of two"
    h = np.array([[1.0]])
    while h.shape[0] < d:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(d)


def rht_matrix(d, seed=0, dtype=np.float64):
    s = np.random.default_rng(seed).choice([-1.0, 1.0], d)
    return (hadamard(d) * s[None, :]).astype(dtype)          # H @ diag(s)


def rotate(x, R):
    return x @ R.T


def unrotate(y, R):
    return y @ R
