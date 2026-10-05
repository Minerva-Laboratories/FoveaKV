"""Access to the captured KV cache (data/kv/<corpus>/L<i>.npz). Reference = int8 (per-token/head absmax)."""
import json, os, numpy as np
from .model_hybrid import rope_cos_sin, apply_rope
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
KV_DIR = os.environ.get("FOVEAKV_KV_DIR", os.path.join(DATA, "kv"))     # captured caches (one folder per model if you set it)
ROT_DIM, THETA, HD = 64, 1e7, 256   # verified against config.json in tests


def corpora():
    return sorted(os.listdir(KV_DIR))


def meta(corpus):
    return json.load(open(os.path.join(KV_DIR, corpus, "meta.json")))


def load_layer(corpus, layer, ref="int8"):
    """Returns dict with fp32 arrays: k_pre, k (post-RoPE), v (T,Hkv,d); q (E,Hq,d) post-RoPE,
    q_pre, positions of the queries, plus raw extras. ref: 'int8' or 'fp16' (only where captured)."""
    z = np.load(os.path.join(KV_DIR, corpus, f"L{layer}.npz"))
    if ref == "fp16":
        k_pre, v = z["k16"].astype(np.float32), z["v16"].astype(np.float32)
    else:
        k_pre = z["k_q"].astype(np.float32) * z["k_s"].astype(np.float32)[..., None]
        v = z["v_q"].astype(np.float32) * z["v_s"].astype(np.float32)[..., None]
    T = k_pre.shape[0]
    q_pre = z["q"].astype(np.float32)
    E = q_pre.shape[0]
    qpos = np.arange(T - E, T)
    cos, sin = rope_cos_sin(np.arange(T), ROT_DIM, THETA)
    out = dict(k_pre=k_pre, v=v, k=apply_rope(k_pre, cos, sin, ROT_DIM), q_pre=q_pre,
               q=apply_rope(q_pre, cos[qpos], sin[qpos], ROT_DIM), qpos=qpos, T=T, E=E)
    for extra in ("q_spec", "attn_in", "gate", "q2_pre", "q2_post"):
        if extra in z:
            out[extra] = z[extra].astype(np.float32)
    if "q_spec" in out:
        out["q_spec"] = apply_rope(out["q_spec"], cos[qpos], sin[qpos], ROT_DIM)
    return out
