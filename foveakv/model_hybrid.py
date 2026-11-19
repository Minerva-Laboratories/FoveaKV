"""Minimal NumPy forward pass for Qwen3.5 (hybrid: Gated DeltaNet + gated full attention).

Semantics follow transformers' modeling_qwen3_5.py (read, not imported):
  * RMSNorm scales by (1 + w); the DeltaNet output norm is gated RMSNorm with plain w.
  * Full attention: q_proj -> per head [query | gate]; q/k RMSNorm over head_dim; partial RoPE
    (rotate_half on the first rotary_dim dims); GQA; output * sigmoid(gate); o_proj.
  * Text-only interleaved mRoPE == plain 1-D RoPE (all three position ids are equal).
  * DeltaNet: causal depthwise conv(4)+SiLU over [q|k|v]; l2-normed q,k; q /= sqrt(dk);
    beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias); plain recurrent delta rule.

Weights stay mmapped as bf16 and are converted per layer when used (low-RAM host).

Hook for experiments: `attn_hook(layer_idx, q, k, v, k_pre, positions) -> out or None`
  q: (T, Hq, d) post-RoPE, k/v: (T, Hkv, d) post-RoPE, k_pre: pre-RoPE keys.
  If it returns an array (T, Hq, d) it replaces the exact attention output (before the gate).
"""
import numpy as np
from .load import SafeTensors, attention_geometry, load_config, snapshot_dir

P = "model.language_model."


def rmsnorm(x, w, eps, plus_one=True):
    x32 = x.astype(np.float32, copy=False)
    y = x32 / np.sqrt(np.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return y * ((1.0 + w) if plus_one else w)


def silu(x):
    return x / (1.0 + np.exp(-x))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.logaddexp(0.0, x)


def rope_cos_sin(positions, rot_dim, theta):
    inv = 1.0 / theta ** (np.arange(0, rot_dim, 2, dtype=np.float64) / rot_dim)
    fr = np.asarray(positions, dtype=np.float64)[:, None] * inv[None, :]
    emb = np.concatenate([fr, fr], axis=-1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def apply_rope(x, cos, sin, rot_dim, inverse=False):
    """x: (T, H, d). rotate_half RoPE on the first rot_dim dims. inverse=True undoes it."""
    if inverse:
        sin = -sin
    xr, xp = x[..., :rot_dim], x[..., rot_dim:]
    h = rot_dim // 2
    rh = np.concatenate([-xr[..., h:], xr[..., :h]], axis=-1)
    out = xr * cos[:, None, :] + rh * sin[:, None, :]
    return np.concatenate([out, xp], axis=-1)


def causal_attention(q, k, v, scale, block=256):
    """q: (T, Hq, d), k/v: (T, Hkv, d). Exact causal softmax attention, query-blocked."""
    T, Hq, d = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    out = np.empty_like(q)
    for s in range(0, T, block):
        e = min(T, s + block)
        qb = q[s:e].reshape(e - s, Hkv, G, d)
        sc = np.einsum("tkgd,skd->kgts", qb, k[:e]) * scale          # (Hkv,G,tb,e)
        mask = np.arange(e)[None, :] > np.arange(s, e)[:, None]
        sc[..., mask] = -np.inf
        sc -= sc.max(-1, keepdims=True)
        p = np.exp(sc)
        p /= p.sum(-1, keepdims=True)
        out[s:e] = np.einsum("kgts,skd->tkgd", p, v[:e]).reshape(e - s, Hq, d)
    return out


class Qwen35:
    def __init__(self, repo=None):
        self.dir = snapshot_dir(repo)
        self.cfg = c = load_config(self.dir)
        self.st = SafeTensors(self.dir)
        self.L = c["num_hidden_layers"]
        self.layer_types = c["layer_types"]
        self.eps = c["rms_norm_eps"]
        g = attention_geometry(c)
        self.hd, self.Hq, self.Hkv, self.rot_dim, self.theta = g["head_dim"], g["Hq"], g["Hkv"], g["rot_dim"], g["theta"]
        assert (c.get("rope_parameters") or {}).get("rope_type", "default") == "default"
        self.lin = dict(nk=c["linear_num_key_heads"], nv=c["linear_num_value_heads"],
                        dk=c["linear_key_head_dim"], dv=c["linear_value_head_dim"],
                        kconv=c["linear_conv_kernel_dim"])
        self.attn_layers = [i for i, t in enumerate(self.layer_types) if t == "full_attention"]
        self._cache = {}

    def w(self, name):
        if name in self._cache:
            return self._cache[name]
        return self.st.get(P + name)

    def cache_layer_weights(self, layers):
        """Optionally keep some layers' fp32 weights resident (costs RAM)."""
        for i in layers:
            for k in self.st.keys():
                if k.startswith(P + f"layers.{i}."):
                    self._cache[k[len(P):]] = self.st.get(k)

    # ---------------------------------------------------------------- embedding / head
    def embed(self, ids):
        raw, _ = self.st.get_raw(P + "embed_tokens.weight")
        from .load import bf16_to_f32
        return bf16_to_f32(raw[np.asarray(ids)])

    def logits_iter(self, h, vblock=32768):
        """Yields (v0, v1, logits[:, v0:v1]) for the tied LM head without materializing E."""
        from .load import bf16_to_f32
        raw, _ = self.st.get_raw(P + "embed_tokens.weight")
        hn = rmsnorm(h, self.w("norm.weight"), self.eps)
        for v0 in range(0, raw.shape[0], vblock):
            v1 = min(raw.shape[0], v0 + vblock)
            yield v0, v1, hn @ bf16_to_f32(raw[v0:v1]).T

    def nll_and_argmax(self, h, targets):
        """Per-position NLL of targets and argmax token, streaming over the vocab."""
        T = h.shape[0]
        m = np.full(T, -np.inf, np.float64)
        s = np.zeros(T, np.float64)
        tgt = np.zeros(T, np.float64)
        best = np.full(T, -np.inf)
        arg = np.zeros(T, np.int64)
        targets = np.asarray(targets)
        for v0, v1, lg in self.logits_iter(h):
            lg = lg.astype(np.float64)
            bm = lg.max(1)
            nm = np.maximum(m, bm)
            s = s * np.exp(m - nm) + np.exp(lg - nm[:, None]).sum(1)
            m = nm
            inb = (targets >= v0) & (targets < v1)
            tgt[inb] = lg[np.where(inb)[0], targets[inb] - v0]
            ba = lg.argmax(1)
            upd = bm > best
            arg[upd] = ba[upd] + v0
            best[upd] = bm[upd]
        return (m + np.log(s)) - tgt, arg

    # ---------------------------------------------------------------- layers
    def mlp(self, i, x):
        pre = f"layers.{i}.mlp."
        return (silu(x @ self.w(pre + "gate_proj.weight").T) * (x @ self.w(pre + "up_proj.weight").T)) \
            @ self.w(pre + "down_proj.weight").T

    def attn_qkv(self, i, x, positions):
        """Returns q (post-RoPE), gate, k (post-RoPE), v, k_pre (pre-RoPE, post k_norm), q_pre."""
        pre = f"layers.{i}.self_attn."
        T = x.shape[0]
        qg = (x @ self.w(pre + "q_proj.weight").T).reshape(T, self.Hq, 2 * self.hd)
        q, gate = qg[..., :self.hd], qg[..., self.hd:].reshape(T, -1)
        k = (x @ self.w(pre + "k_proj.weight").T).reshape(T, self.Hkv, self.hd)
        v = (x @ self.w(pre + "v_proj.weight").T).reshape(T, self.Hkv, self.hd)
        q = rmsnorm(q, self.w(pre + "q_norm.weight"), self.eps)
        k = rmsnorm(k, self.w(pre + "k_norm.weight"), self.eps)
        cos, sin = rope_cos_sin(positions, self.rot_dim, self.theta)
        return apply_rope(q, cos, sin, self.rot_dim), gate, apply_rope(k, cos, sin, self.rot_dim), v, k, q

    def full_attn(self, i, x, positions, hook=None, kv_state=None):
        q, gate, k, v, k_pre, q_pre = self.attn_qkv(i, x, positions)
        if kv_state is not None:                      # incremental decode: append to cache
            kv_state["k"] = np.concatenate([kv_state.get("k", k[:0]), k])
            kv_state["v"] = np.concatenate([kv_state.get("v", v[:0]), v])
        out = None
        if hook is not None:
            out = hook(i, q, k, v, k_pre, positions, q_pre=q_pre, x=x)
        if out is None:
            if kv_state is not None:
                out = decode_attention(q, kv_state["k"], kv_state["v"], self.hd ** -0.5)
            else:
                out = causal_attention(q, k, v, self.hd ** -0.5)
        out = out.reshape(x.shape[0], -1) * sigmoid(gate)
        return out @ self.w(f"layers.{i}.self_attn.o_proj.weight").T

    def deltanet(self, i, x, state=None):
        """Gated DeltaNet, plain recurrent loop. state: dict with 'conv' (C,k-1) and 'S' (H,dk,dv)."""
        pre = f"layers.{i}.linear_attn."
        p = self.lin
        T = x.shape[0]
        mixed = x @ self.w(pre + "in_proj_qkv.weight").T                 # (T, C)
        z = (x @ self.w(pre + "in_proj_z.weight").T).reshape(T, p["nv"], p["dv"])
        b = x @ self.w(pre + "in_proj_b.weight").T
        a = x @ self.w(pre + "in_proj_a.weight").T
        cw = self.w(pre + "conv1d.weight")[:, 0, :]                        # (C, K)
        K = cw.shape[1]
        prev = state["conv"] if state is not None and "conv" in state else np.zeros((K - 1, mixed.shape[1]), np.float32)
        buf = np.concatenate([prev, mixed])                                 # (K-1+T, C)
        conv = sum(buf[j:j + T] * cw[:, j] for j in range(K))
        if state is not None:
            state["conv"] = buf[-(K - 1):]
        conv = silu(conv)
        kd = p["nk"] * p["dk"]
        q = conv[:, :kd].reshape(T, p["nk"], p["dk"])
        k = conv[:, kd:2 * kd].reshape(T, p["nk"], p["dk"])
        v = conv[:, 2 * kd:].reshape(T, p["nv"], p["dv"])
        rep = p["nv"] // p["nk"]
        if rep > 1:
            q, k = np.repeat(q, rep, axis=1), np.repeat(k, rep, axis=1)
        q = q / np.sqrt((q * q).sum(-1, keepdims=True) + 1e-6) / np.sqrt(p["dk"])
        k = k / np.sqrt((k * k).sum(-1, keepdims=True) + 1e-6)
        beta = sigmoid(b)
        g = np.exp(-np.exp(self.w(pre + "A_log")) * softplus(a + self.w(pre + "dt_bias")))  # decay in (0,1)
        S = state["S"] if state is not None and "S" in state else np.zeros((p["nv"], p["dk"], p["dv"]), np.float32)
        out = np.empty((T, p["nv"], p["dv"]), np.float32)
        for t in range(T):
            S = S * g[t][:, None, None]
            kv_mem = np.einsum("hkv,hk->hv", S, k[t])
            delta = (v[t] - kv_mem) * beta[t][:, None]
            S = S + k[t][:, :, None] * delta[:, None, :]
            out[t] = np.einsum("hkv,hk->hv", S, q[t])
        if state is not None:
            state["S"] = S
        o = rmsnorm(out, self.w(pre + "norm.weight"), self.eps, plus_one=False) * silu(z)
        return o.reshape(T, -1) @ self.w(pre + "out_proj.weight").T

    # ---------------------------------------------------------------- full model
    def forward(self, ids, positions=None, hook=None, state=None, return_hidden=False, layer_cb=None):
        """ids: 1-D token ids. state: per-layer dict list for incremental decode (or None).
        layer_cb(i, h) is called with the residual stream after each layer (for E7 etc.)."""
        ids = np.asarray(ids)
        if positions is None:
            off = state["pos"] if state is not None else 0
            positions = np.arange(off, off + len(ids))
        h = self.embed(ids)
        for i in range(self.L):
            st = state["layers"][i] if state is not None else None
            x = rmsnorm(h, self.w(f"layers.{i}.input_layernorm.weight"), self.eps)
            if self.layer_types[i] == "full_attention":
                h = h + self.full_attn(i, x, positions, hook=hook, kv_state=st)
            else:
                h = h + self.deltanet(i, x, state=st)
            x = rmsnorm(h, self.w(f"layers.{i}.post_attention_layernorm.weight"), self.eps)
            h = h + self.mlp(i, x)
            if layer_cb is not None:
                layer_cb(i, h)
        if state is not None:
            state["pos"] = positions[-1] + 1
        return h

    def new_state(self):
        return {"pos": 0, "layers": [dict() for _ in range(self.L)]}


def decode_attention(q, k, v, scale, block=256):
    """q: (t, Hq, d) for the last t positions of the cache k/v: (T, Hkv, d). Query-blocked."""
    t, Hq, d = q.shape
    T, Hkv, _ = k.shape
    G = Hq // Hkv
    out = np.empty_like(q)
    for s in range(0, t, block):
        e = min(t, s + block)
        sc = np.einsum("tkgd,skd->kgts", q[s:e].reshape(e - s, Hkv, G, d), k) * scale
        mask = np.arange(T)[None, :] > (T - t + np.arange(s, e))[:, None]
        sc[..., mask] = -np.inf
        sc -= sc.max(-1, keepdims=True)
        p = np.exp(sc)
        p /= p.sum(-1, keepdims=True)
        out[s:e] = np.einsum("kgts,skd->tkgd", p, v).reshape(e - s, Hq, d)
    return out
