"""Access to the captured KV cache (data/kv/<corpus>/L<i>.npz). Reference = int8 (per-token/head absmax)."""
import json, os, numpy as np
from .model_hybrid import rope_cos_sin, apply_rope
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
KV_DIR = os.environ.get("FOVEAKV_KV_DIR", os.path.join(DATA, "kv"))     # captured caches (one folder per model if you set it)
ROT_DIM, THETA, HD = 64, 1e7, 256   # Qwen3.5 defaults (verified in tests); captures record their own in meta.json


def corpora():
    return sorted(os.listdir(KV_DIR))


def meta(corpus):
    return json.load(open(os.path.join(KV_DIR, corpus, "meta.json")))


REF = os.environ.get("FOVEAKV_REF", "int8")   # fp16 for models whose outlier channels make per-token int8 lossy (Qwen3)


def load_layer(corpus, layer, ref=None):
    """Returns dict with fp32 arrays: k_pre, k (post-RoPE), v (T,Hkv,d); q (E,Hq,d) post-RoPE,
    q_pre, positions of the queries, plus raw extras. ref: 'int8' or 'fp16' (only where captured)."""
    ref = ref or REF
    z = np.load(os.path.join(KV_DIR, corpus, f"L{layer}.npz"))
    if ref == "fp16":
        k_pre, v = z["k16"].astype(np.float32), z["v16"].astype(np.float32)
    else:
        k_pre = z["k_q"].astype(np.float32) * z["k_s"].astype(np.float32)[..., None]
        v = z["v_q"].astype(np.float32) * z["v_s"].astype(np.float32)[..., None]
    T = k_pre.shape[0]
    m = meta(corpus)
    rot, theta = m.get("rot_dim", ROT_DIM), m.get("theta", THETA)
    q_pre = z["q"].astype(np.float32)
    E = q_pre.shape[0]
    qpos = np.arange(T - E, T)
    cos, sin = rope_cos_sin(np.arange(T), rot, theta)
    out = dict(k_pre=k_pre, v=v, k=apply_rope(k_pre, cos, sin, rot), q_pre=q_pre,
               q=apply_rope(q_pre, cos[qpos], sin[qpos], rot), qpos=qpos, T=T, E=E, rot_dim=rot, theta=theta)
    for extra in ("q_spec", "attn_in", "gate", "q2_pre", "q2_post"):
        if extra in z:
            out[extra] = z[extra].astype(np.float32)
    if "q_spec" in out:
        out["q_spec"] = apply_rope(out["q_spec"], cos[qpos], sin[qpos], rot)
    return out


def weight_name(layer, what="self_attn.o_proj.weight"):
    """Checkpoint key of a layer weight (Qwen3.5 nests the language model under model.language_model)."""
    from .load import SafeTensors, snapshot_dir
    keys = SafeTensors(snapshot_dir()).keys()
    for pre in ("model.language_model.layers", "model.layers"):
        if f"{pre}.{layer}.{what}" in keys:
            return f"{pre}.{layer}.{what}"
    raise KeyError(what)
