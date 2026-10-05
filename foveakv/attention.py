"""Walk-forward decode attention over a (quantized) cache, GQA layout.
Shapes: q (E, Hq, d) queries at absolute positions qpos (E,); K, V (T, Hkv, d).
Scores are (Hkv, G, E, T) with keys t > qpos masked. hot masks are (Hkv, E, T) (shared by the group)
or (Hkv, G, E, T)."""
import numpy as np


def scores(q, K, qpos, scale=None):
    E, Hq, d = q.shape
    T, Hkv, _ = K.shape
    G = Hq // Hkv
    scale = d ** -0.5 if scale is None else scale
    qh = q.reshape(E, Hkv, G, d).transpose(1, 2, 0, 3)                 # (Hkv,G,E,d)
    s = np.matmul(qh, K.transpose(1, 2, 0)[:, None]) * scale            # (Hkv,G,E,T), BLAS
    s[..., np.arange(T)[None, :] > np.asarray(qpos)[:, None]] = -np.inf
    return s


def softmax(s):
    m = s.max(-1, keepdims=True)
    p = np.exp(s - m)
    return p / p.sum(-1, keepdims=True)


def logprob(s):
    m = s.max(-1, keepdims=True)
    return s - (m + np.log(np.exp(s - m).sum(-1, keepdims=True)))


def attend(p, V):
    """p (Hkv,G,E,T), V (T,Hkv,d) -> (E, Hq, d)."""
    Hkv, G, E, T = p.shape
    o = np.matmul(p, V.transpose(1, 0, 2)[:, None])                     # (Hkv,G,E,d)
    return o.transpose(2, 0, 1, 3).reshape(E, Hkv * G, V.shape[-1])


def exact(q, K, V, qpos, scale=None):
    p = softmax(scores(q, K, qpos, scale))
    return attend(p, V), p


def _bcast(hot, G):
    return hot[:, None] if hot.ndim == 3 else hot


def mixed(q, Kc, Kf, Vc, Vf, hot, qpos, scale=None):
    """Reference recompute: hot tokens use fine K/V, cold use coarse K/V."""
    h = _bcast(hot, None)
    sc, sf = scores(q, Kc, qpos, scale), scores(q, Kf, qpos, scale)
    p = softmax(np.where(h, sf, sc))
    Hkv, G, E, T = p.shape
    ph = p * h
    return attend(p - ph, Vc) + attend(ph, Vf)


def delta_corrected(q, Kc, Kf, Vc, Vf, hot, qpos, scale=None):
    """Online-softmax delta correction: one coarse pass over all tokens (m, l, acc), then a correction
    pass that touches only hot tokens: swap their coarse contribution for the fine one."""
    h = _bcast(hot, None)
    sc = scores(q, Kc, qpos, scale)
    m = sc.max(-1, keepdims=True)                       # coarse pass
    ec = np.exp(sc - m)
    l = ec.sum(-1, keepdims=True)
    acc = np.einsum("kget,tkd->kged", ec, Vc)
    sf = scores(q, Kf, qpos, scale)                     # correction pass (only hot entries matter)
    sf_hot = np.where(h, sf, -np.inf)
    m2 = np.maximum(m, sf_hot.max(-1, keepdims=True))
    r = np.exp(m - m2)                                  # rescale running stats if a hot logit is larger
    ef = np.exp(sf_hot - m2)
    ec_hot = np.where(h, ec * r, 0.0)
    l2 = l * r + (ef - ec_hot).sum(-1, keepdims=True)
    acc2 = acc * r + np.einsum("kget,tkd->kged", ef, Vf) - np.einsum("kget,tkd->kged", ec_hot, Vc)
    out = acc2 / l2
    Hkv, G, E, d = out.shape
    return out.transpose(2, 0, 1, 3).reshape(E, Hkv * G, d)
