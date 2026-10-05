"""Quantizers: Gaussian Lloyd-Max 4-bit with fine-first nested 2-bit coarse plane, log-uniform gain,
gain-shape encode/decode, KIVI-style baseline, int8 reference format. All codebooks analytic."""
from math import erf, sqrt, pi
import numpy as np

_Phi = np.vectorize(lambda x: 0.5 * (1.0 + erf(x / sqrt(2.0))))


def _phi(x):
    return np.exp(-x * x / 2.0) / sqrt(2.0 * pi)


def cond_mean(a, b):
    """E[X | a < X < b], X ~ N(0,1); a,b may be ±inf."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (_phi(a) - _phi(b)) / np.maximum(_Phi(b) - _Phi(a), 1e-300)


def gaussian_mse(bnd, lev):
    """Exact MSE of a scalar quantizer (boundaries bnd, levels lev) on N(0,1)."""
    e = np.concatenate([[-np.inf], bnd, [np.inf]])
    tot = 0.0
    for i, c in enumerate(lev):
        a, b = e[i], e[i + 1]
        p = _Phi(b) - _Phi(a)
        m2 = p - (b * _phi(b) if np.isfinite(b) else 0.0) + (a * _phi(a) if np.isfinite(a) else 0.0)
        tot += m2 - 2.0 * c * (_phi(a) - _phi(b)) + c * c * p
    return float(tot)


def lloyd_max(n, iters=500):
    lev = np.linspace(-2.5, 2.5, n)
    for _ in range(iters):
        bnd = (lev[1:] + lev[:-1]) / 2.0
        e = np.concatenate([[-np.inf], bnd, [np.inf]])
        lev = cond_mean(e[:-1], e[1:])
    return (lev[1:] + lev[:-1]) / 2.0, lev


class NestedCodebook:
    """Fine-first nested codebook: fine = n_fine-level Lloyd-Max for N(0,1); coarse cell c = union of
    `merge` consecutive fine cells, reconstructed at the Gaussian conditional mean.
    idx in [0, n_fine); hi = idx >> lo_bits, lo = idx & (merge-1)."""

    def __init__(self, fine_bits=4, coarse_bits=2):
        self.fine_bits, self.coarse_bits = fine_bits, coarse_bits
        self.lo_bits = fine_bits - coarse_bits
        self.merge = 1 << self.lo_bits
        self.bnd16, self.lut16 = lloyd_max(1 << fine_bits)
        e = np.concatenate([[-np.inf], self.bnd16, [np.inf]])
        ce = e[:: self.merge]
        ce = np.concatenate([ce, [np.inf]]) if ce[-1] != np.inf else ce
        self.bnd4 = ce[1:-1]
        self.lut4 = cond_mean(ce[:-1], ce[1:])
        self.mse_fine = gaussian_mse(self.bnd16, self.lut16)
        self.mse_coarse = gaussian_mse(self.bnd4, self.lut4)

    def encode(self, z):
        return np.searchsorted(self.bnd16, z).astype(np.uint8)

    def fine(self, idx):
        return self.lut16[idx]

    def coarse(self, idx):
        return self.lut4[idx >> self.lo_bits]

    def sign(self, idx):
        """MSB plane: +1 for the upper half of the cells (cells are symmetric about 0)."""
        return np.where(idx >= (1 << (self.fine_bits - 1)), 1.0, -1.0)


class LogGain:
    """Uniform quantizer of log g on [lo, hi] (calibrated on prefill), midpoint reconstruction."""

    def __init__(self, g_calib, bits=8, margin=0.0):
        lg = np.log(np.maximum(g_calib, 1e-30))
        self.lo, self.hi = float(lg.min()) - margin, float(lg.max()) + margin
        self.bits = bits
        self.n = 1 << bits

    def quant(self, g):
        lg = np.log(np.maximum(g, 1e-30))
        span = max(self.hi - self.lo, 1e-12)
        c = np.clip(np.floor((lg - self.lo) / span * self.n), 0, self.n - 1)
        return np.exp(self.lo + (c + 0.5) / self.n * span)


class GainShape:
    """Gain-shape quantizer for one KV head.
    fit(): mu, gain range from calibration (prefill) tokens. encode(x): idx (uint8, rotated shape codes),
    gq (quantized gains). decode(...): rotated-domain reconstruction g*LUT[idx]/sqrt(d) (mean NOT added)."""

    def __init__(self, R, cb, gain_bits=8, center=True):
        self.R, self.cb, self.gain_bits, self.center = R, cb, gain_bits, center
        self.d = R.shape[0]

    def fit(self, x_calib):
        self.mu = x_calib.mean(0) if self.center else np.zeros(self.d, x_calib.dtype)
        y = (x_calib - self.mu) @ self.R.T
        self.gain = LogGain(np.linalg.norm(y, axis=-1), self.gain_bits)
        return self

    def encode(self, x):
        y = (x - self.mu) @ self.R.T
        g = np.linalg.norm(y, axis=-1)
        u = y / np.maximum(g, 1e-30)[..., None]
        return self.cb.encode(u * np.sqrt(self.d)), self.gain.quant(g), g, u

    def decode_rot(self, idx, gq, fine=True):
        lev = self.cb.fine(idx) if fine else self.cb.coarse(idx)
        return gq[..., None] * lev / np.sqrt(self.d)

    def decode(self, idx, gq, fine=True, add_mean=True):
        x = self.decode_rot(idx, gq, fine) @ self.R
        return x + self.mu if add_mean else x

    def bits_per_elem(self, shape_bits):
        return shape_bits + self.gain_bits / self.d


# ------------------------------------------------------------------------- KIVI baseline
def _asym(x, bits, axis):
    lo = x.min(axis, keepdims=True)
    hi = x.max(axis, keepdims=True)
    s = (hi - lo) / ((1 << bits) - 1)
    s = np.where(s == 0, 1.0, s)
    c = np.clip(np.round((x - lo) / s), 0, (1 << bits) - 1)
    return c * s + lo


def kivi_keys(K, bits, group):
    """Per-channel quantization over groups of `group` consecutive tokens. K: (T, d).
    The trailing partial group stays at reference precision (as in KIVI's residual)."""
    T, d = K.shape
    n = (T // group) * group
    out = K.copy()
    out[:n] = _asym(K[:n].reshape(-1, group, d), bits, 1).reshape(n, d)
    return out, n


def kivi_values(V, bits, group):
    """Per-token quantization over groups of `group` channels. V: (T, d)."""
    T, d = V.shape
    return _asym(V.reshape(T, d // group, group), bits, 2).reshape(T, d)


def kivi_bits(bits, group, meta_bits=32):
    """fp16 scale + fp16 zero per group."""
    return bits + meta_bits / group


# ------------------------------------------------------------------------- int8 reference format
def int8_absmax(x):
    s = np.abs(x).max(-1, keepdims=True) / 127.0
    s = np.where(s == 0, 1.0, s)
    return np.clip(np.round(x / s), -127, 127) * s


INT8_BITS = 8 + 16 / 256
