"""Build quantized caches for one captured layer (walk-forward calibration on the prefill only).

All reconstructions are returned in the attention ("logit") domain (post-RoPE, mean added back), which
is mathematically identical to the kernel's rotated-domain / mean-free computation (orthogonal R;
q·mu_K is a per-query constant post-RoPE). The rotated-domain kernel is exercised in tests."""
import numpy as np
from .quant import NestedCodebook, GainShape, kivi_keys, kivi_values, kivi_bits
from .rotate import rht_matrix
from .model_hybrid import rope_cos_sin, apply_rope

CB = NestedCodebook(4, 2)
SIGN_LEVEL = np.sqrt(2 / np.pi)       # 1-bit Lloyd-Max level for N(0,1)


def _gs_head(x, P, R, gain_bits, center):
    q = GainShape(R, CB, gain_bits, center).fit(x[:P])
    idx, gq, g, u = q.encode(x)
    rot = {"fine": q.decode_rot(idx, gq, True), "coarse": q.decode_rot(idx, gq, False),
           "sign": gq[:, None] * CB.sign(idx) * SIGN_LEVEL / np.sqrt(q.d)}
    return {k: v @ R + q.mu for k, v in rot.items()}, dict(idx=idx, g=g, u=u, mu=q.mu)


def gain_shape_cache(L, P, rope="post", rht=True, gain_bits=8, center_k=True, center_v=True, seed=0):
    """L: dict from kvdata.load_layer. Returns dict of (T,Hkv,d) arrays:
    Kf, Kc, Ks (sign-plane estimate), Vf, Vc, plus per-head diagnostics."""
    T, Hkv, d = L["k"].shape
    R = rht_matrix(d, seed) if rht else np.eye(d)
    src = L["k"] if rope == "post" else L["k_pre"]
    out = {n: np.empty((T, Hkv, d), np.float32) for n in ("Kf", "Kc", "Ks", "Vf", "Vc")}
    diag = []
    for h in range(Hkv):
        rk, dk = _gs_head(src[:, h].astype(np.float64), P, R, gain_bits, center_k)
        rv, dv = _gs_head(L["v"][:, h].astype(np.float64), P, R, gain_bits, center_v)
        out["Kf"][:, h], out["Kc"][:, h], out["Ks"][:, h] = rk["fine"], rk["coarse"], rk["sign"]
        out["Vf"][:, h], out["Vc"][:, h] = rv["fine"], rv["coarse"]
        diag.append(dict(k=dk, v=dv))
    if rope == "pre":
        rot = L.get("rot_dim", 64)
        cos, sin = rope_cos_sin(np.arange(T), rot, L.get("theta", 1e7))
        for n in ("Kf", "Kc", "Ks"):
            out[n] = apply_rope(out[n], cos, sin, rot)
    out["diag"] = diag
    out["bits_coarse"] = 2 + gain_bits / d
    out["bits_fine"] = 4 + gain_bits / d
    return out


def kivi_cache(L, bits, group):
    """KIVI-style: keys (post-RoPE) per-channel over token groups, values per-token over channel groups.
    Returns Kq, Vq and partial_mask(qpos) -> (E,T) tokens still at reference precision (incomplete group)."""
    T, Hkv, d = L["k"].shape
    Kq = np.empty_like(L["k"])
    Vq = np.empty_like(L["v"])
    for h in range(Hkv):
        Kq[:, h], _ = kivi_keys(L["k"][:, h], bits, group)
        Vq[:, h] = kivi_values(L["v"][:, h], bits, min(group, d))

    def partial_mask(qpos):
        start = ((np.asarray(qpos) + 1) // group) * group
        return np.arange(T)[None, :] >= start[:, None]
    bits_k = kivi_bits(bits, group)
    bits_v = kivi_bits(bits, min(group, d))
    return dict(Kq=Kq, Vq=Vq, partial_mask=partial_mask, bits=(bits_k + bits_v) / 2)


def scheme_attention(cfg, L, P, q, qpos):
    """Attention output (E,Hq,d) for decode queries q at qpos over the cache in L (k_pre, k, v from the int8
    reference), under scheme cfg. cfg['kind']: 'ref' | 'gs' (bits 2/4) | 'dyn' (sel, k, S, W) | 'kivi' (bits, group).
    Also returns (resident_bits, read_bits)."""
    from .attention import scores, softmax, attend
    from .select import select
    from .quant import INT8_BITS
    kind = cfg["kind"]
    if kind == "ref":
        return attend(softmax(scores(q, L["k"], qpos)), L["v"]), (INT8_BITS, INT8_BITS)
    if kind == "kivi":
        kv = kivi_cache(L, cfg["bits"], cfg["group"])
        pm = kv["partial_mask"](qpos)
        h = np.broadcast_to(pm[None, None], (L["k"].shape[1], q.shape[1] // L["k"].shape[1]) + pm.shape)
        p = softmax(np.where(h, scores(q, L["k"], qpos), scores(q, kv["Kq"], qpos)))
        ph = p * h
        return attend(p - ph, kv["Vq"]) + attend(ph, L["v"]), (kv["bits"], kv["bits"])
    gs = gain_shape_cache(L, P, rope=cfg.get("rope", "post"), rht=cfg.get("rht", True), gain_bits=cfg.get("gain_bits", 8))
    if kind == "gs":
        f = "f" if cfg["bits"] == 4 else "c"
        b = gs["bits_fine"] if cfg["bits"] == 4 else gs["bits_coarse"]
        return attend(softmax(scores(q, gs["K" + f], qpos)), gs["V" + f]), (b, b)
    sc, sf = scores(q, gs["Kc"], qpos), scores(q, gs["Kf"], qpos)
    s_sel = sc if cfg.get("sel", "coarse") == "coarse" else scores(q, gs["Ks"], qpos)
    hot = select(s_sel, qpos, cfg["k"], cfg.get("S", 4), cfg.get("W", 64), mode=cfg.get("mode", "budget"))
    h = hot[:, None]
    p = softmax(np.where(h, sf, sc))
    ph = p * h
    f = hot.sum(-1).mean() / (np.mean(qpos) + 1)
    return attend(p - ph, gs["Vc"]) + attend(ph, gs["Vf"]), (gs["bits_coarse"], gs["bits_coarse"] + 2 * f)


# ----------------------------------------------------------------------------- improvements / extra baselines
def key_smoothing_scales(L, P, rope="post", eps=1e-6):
    """Closed-form per-channel key/query balancing (SmoothQuant-like) per KV head, from prefill statistics:
    s_c = (E[(k_c-mu_c)^2] / E[q_c^2])^(1/4) (q stats averaged over the GQA group), geometric mean normalized to 1.
    Minimizes ||q∘s||·||(k-mu)/s|| (Cauchy-Schwarz), i.e. the logit error of an isotropic quantizer of (k-mu)/s."""
    src = L["k"] if rope == "post" else L["k_pre"]
    q2 = L["q2_post"] if rope == "post" else L["q2_pre"]               # (Hq, d)
    T, Hkv, d = src.shape
    G = q2.shape[0] // Hkv
    S = np.empty((Hkv, d))
    for h in range(Hkv):
        x = src[:P, h].astype(np.float64)
        k2 = ((x - x.mean(0)) ** 2).mean(0)
        s = ((k2 + eps) / (q2[h * G:(h + 1) * G].mean(0) + eps)) ** 0.25
        S[h] = s / np.exp(np.log(s).mean())
    return S


def block_means(X, B):
    """Per-block (B consecutive tokens) per-channel means, fp16-rounded, broadcast back to (T,...).
    The trailing incomplete block uses its own partial mean (those tokens sit in the exact recent window anyway)."""
    T = X.shape[0]
    n = (T // B) * B
    M = np.zeros_like(X, dtype=np.float32)
    M[:n] = X[:n].reshape(-1, B, *X.shape[1:]).mean(1).repeat(B, 0)
    if T > n:
        M[n:] = X[n:].mean(0)
    return M.astype(np.float16).astype(np.float32)


def gain_shape_cache_v2(L, P, rht=True, gain_bits=8, smooth=False, block=None, seed=0):
    """Post-RoPE gain-shape cache with optional key smoothing and block-local centering (block = tokens per mean;
    costs 16/block bits/elem per tensor). Same outputs as gain_shape_cache (reconstructions include the means)."""
    T, Hkv, d = L["k"].shape
    Lk = dict(L)
    Mk = block_means(L["k"], block) if block else L["k"][:P].mean(0)[None]
    Mv = block_means(L["v"], block) if block else L["v"][:P].mean(0)[None]
    S = key_smoothing_scales(L, P, "post") if smooth else np.ones((Hkv, d))
    Lk["k"] = ((L["k"] - Mk) / S[None]).astype(np.float32)
    Lk["v"] = (L["v"] - Mv).astype(np.float32)
    out = gain_shape_cache(Lk, P, "post", rht, gain_bits, center_k=False, center_v=False, seed=seed)
    for n in ("Kf", "Kc", "Ks"):
        out[n] = (out[n] * S[None] + Mk).astype(np.float32)
    for n in ("Vf", "Vc"):
        out[n] = (out[n] + Mv).astype(np.float32)
    extra = 16 / block if block else 0.0
    out["bits_coarse"] += extra
    out["bits_fine"] += extra
    return out


def turboquant_mse(L, bits, seed=0, rht=True):
    """TurboQuant (MSE variant, no QJL): random rotation, per-coordinate Lloyd-Max codebook for the N(0,1/d)
    coordinate distribution, norm stored in fp16; no mean removal, data-oblivious. Applied to post-RoPE K and V.
    bits/elem = b + 16/d."""
    from .quant import lloyd_max
    T, Hkv, d = L["k"].shape
    R = rht_matrix(d, seed) if rht else np.eye(d)
    bnd, lev = lloyd_max(1 << bits)
    out = {}
    for name, X in (("K", L["k"]), ("V", L["v"])):
        Y = np.empty_like(X)
        for h in range(Hkv):
            y = X[:, h].astype(np.float64) @ R.T
            g = np.linalg.norm(y, axis=-1, keepdims=True).astype(np.float16).astype(np.float64)
            z = y / np.maximum(g, 1e-30) * np.sqrt(d)
            Y[:, h] = (lev[np.searchsorted(bnd, z)] / np.sqrt(d) * g) @ R
        out[name] = Y
    out["bits"] = bits + 16 / d
    return out


def tiered_attention(q, qpos, sc, sf, sr, Vc, Vf, Vr, hot, win):
    """3-tier mix: win (E,T) tokens use reference K/V (logits sr), hot (Hkv,E,T) use fine, others coarse."""
    from .attention import softmax, attend
    w = win[None, None]
    h = hot[:, None] & ~w
    p = softmax(np.where(w, sr, np.where(h, sf, sc)))
    pw, ph = p * w, p * h
    return attend(p - pw - ph, Vc) + attend(ph, Vf) + attend(pw, Vr)
