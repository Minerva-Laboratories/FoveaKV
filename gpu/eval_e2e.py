"""End-to-end GPU evaluation: Qwen3.5-0.8B (bf16 weights, torch AMP autocast) with the quantized KV cache in the
decode loop, using the custom CUDA kernels for every decode step of the 6 full-attention layers.

Protocol (shared by all schemes): exact (bf16) chunked prefill of the context -> snapshot raw K/V of the attention
layers and the DeltaNet states -> per scheme: build the compressed cache from the snapshot, then decode token by token
(teacher-forced text for perplexity; question + greedy answer for retrieval). The question is DECODED through the
quantized cache, not prefilled, so retrieval genuinely depends on the compressed cache.

Usage: <torch venv python> gpu/eval_e2e.py {check|ppl|retrieval} [--quick]"""
import argparse, copy, glob, json, math, os, random, sys, time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention.bias import causal_lower_right

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from gpu.foveakv_gpu import make, ext, Base

from foveakv.load import snapshot_dir
MDIR = snapshot_dir()          # set FOVEAKV_MODEL to use another Qwen3.5 checkpoint
DEV = "cuda"
SCHEMES = ["bf16", "tq2", "tq3", "tq4", "gs4", "kivi2", "kivi4", "quest5", "quest10", "dyn2", "dyn5", "dyn10"]


# ----------------------------------------------------------------------------------------------- model plumbing
class LinCache:
    """Minimal cache for the Gated DeltaNet layers (the API their forward uses)."""

    def __init__(self, layer_ids):
        self.layers = {i: SimpleNamespace(conv_states=[None], recurrent_states=[None], record_past=False) for i in layer_ids}

    def has_previous_state(self, layer_idx=None, state_idx=None):
        return self.layers[layer_idx].recurrent_states[0] is not None

    def update_conv_state(self, s, layer_idx, conv_kernel_size=None, **kw):
        """Newer transformers: called with the new conv inputs (B, C, T); returns them with the cached left context
        prepended and keeps the last conv_kernel_size columns (older versions passed the state itself)."""
        l = self.layers[layer_idx]
        if conv_kernel_size is None:                               # older API
            l.conv_states[0] = s
            return s
        full = s if l.conv_states[0] is None else torch.cat([l.conv_states[0], s], -1)
        if full.shape[-1] < conv_kernel_size:
            full = F.pad(full, (conv_kernel_size - full.shape[-1], 0))
        l.conv_states[0] = full[..., -conv_kernel_size:].contiguous()
        return full

    def update_recurrent_state(self, s, layer_idx, **kw):
        self.layers[layer_idx].recurrent_states[0] = s

    def clone(self):
        c = LinCache(list(self.layers))
        for i, l in self.layers.items():
            c.layers[i].conv_states[0] = l.conv_states[0].clone()
            c.layers[i].recurrent_states[0] = l.recurrent_states[0].clone()
        return c


class Recorder(Base):
    """Stores raw K/V only (used during the shared exact prefill); also accumulates E[q q^T] per KV head (the query
    metric used by query-metric selection studies, e.g. SparsePage's research tree), over every 4th prefill query."""
    full_raw = True
    qq, nq = None, 0

    def observe_q(self, q):
        """q (Hq, T, D) post-RoPE."""
        x = q[:, ::4].float().reshape(self.Hkv, -1, q.shape[-1])                # (Hkv, G*T/4, D)
        a = x.transpose(1, 2) @ x
        self.qq = a if self.qq is None else self.qq + a
        self.nq += x.shape[1]
        if not hasattr(self, "qbuf"):
            self.qbuf = []
        self.qbuf = (self.qbuf + [(self.n, q.detach().to(torch.bfloat16))])[-4:]   # last 4 chunks (positions n, n+T)


def load_model():
    from transformers import Qwen3_5ForCausalLM, AutoTokenizer
    import transformers.models.qwen3_5.modeling_qwen3_5 as M
    m = Qwen3_5ForCausalLM.from_pretrained(MDIR, dtype=torch.bfloat16, device_map=DEV).eval()
    tok = AutoTokenizer.from_pretrained(MDIR)
    return m, tok, M.apply_rotary_pos_emb


class Runner:
    def __init__(self):
        self.m, self.tok, self.rope = load_model()
        cfg = self.m.config
        self.types = cfg.layer_types
        self.attn_ids = [i for i, t in enumerate(self.types) if t == "full_attention"]
        self.lin_ids = [i for i, t in enumerate(self.types) if t != "full_attention"]
        self.hd, self.Hkv = cfg.head_dim, cfg.num_key_value_heads

    def _attn(self, a, x, cos, sin, cache, prefill):
        B, T, _ = x.shape
        hd = self.hd
        qg = a.q_proj(x).view(B, T, -1, 2 * hd)
        q, gate = torch.chunk(qg, 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = a.q_norm(q).transpose(1, 2)
        k = a.k_norm(a.k_proj(x).view(B, T, -1, hd)).transpose(1, 2)
        v = a.v_proj(x).view(B, T, -1, hd).transpose(1, 2)
        q, k = self.rope(q, k, cos, sin)
        kk, vv = k[0].transpose(0, 1), v[0].transpose(0, 1)                    # (T, Hkv, D)
        if prefill:
            if hasattr(cache, "observe_q"):
                cache.observe_q(q[0])
            n0 = cache.n
            cache._store(kk, vv)
            K = cache.kraw[:cache.n].transpose(0, 1)[None]                     # (1, Hkv, n, D)
            V = cache.vraw[:cache.n].transpose(0, 1)[None]
            # lower-right causal bias: fused SDPA kernels, no materialized mask or score matrix
            o = F.scaled_dot_product_attention(q.to(torch.bfloat16), K, V, attn_mask=causal_lower_right(T, cache.n), enable_gqa=True)
            o = o.transpose(1, 2).reshape(B, T, -1)
        else:
            outs = []
            with torch.autocast("cuda", enabled=False):          # cache encode/attend run in fp32 (own kernels)
                for t in range(T):
                    cache.append(kk[t:t + 1].float(), vv[t:t + 1].float())
                    outs.append(cache.attend(q[0, :, t].float()).reshape(1, -1))
            o = torch.stack(outs, 1).to(x.dtype)
        o = o * torch.sigmoid(gate)
        return a.o_proj(o)

    @torch.no_grad()
    def forward(self, ids, pos0, caches, lin, prefill):
        """ids (T,) long. Returns final hidden states (T, H)."""
        mm = self.m.model
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = mm.embed_tokens(ids[None])
            pos = torch.arange(pos0, pos0 + len(ids), device=DEV)[None]
            try:
                cos, sin = mm.rotary_emb(x, pos)
            except IndexError:                          # newer transformers: (3, B, T) multimodal position ids
                cos, sin = mm.rotary_emb(x, pos[None].expand(3, *pos.shape))
            h = x
            for i, layer in enumerate(mm.layers):
                r = h
                y = layer.input_layernorm(h)
                if self.types[i] == "full_attention":
                    y = self._attn(layer.self_attn, y, cos, sin, caches[i], prefill)
                else:
                    y = layer.linear_attn(y, cache_params=lin)
                h = r + y
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            return mm.norm(h)[0]

    def logits(self, h):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return self.m.lm_head(h).float()

    @torch.no_grad()
    def prefill(self, ids, chunk=None):
        T = len(ids)
        chunk = chunk or 512
        caches = {i: Recorder(self.Hkv, T + 512) for i in self.attn_ids}
        lin = LinCache(self.lin_ids)
        h = None
        for s in range(0, T, chunk):
            h = self.forward(ids[s:s + chunk], s, caches, lin, prefill=True)
        return dict(kv={i: (c.kraw[:T].clone(), c.vraw[:T].clone()) for i, c in caches.items()}, lin=lin, T=T,
                    qm={i: c.qq / max(c.nq, 1) for i, c in caches.items()},
                    qbuf={i: getattr(c, "qbuf", []) for i, c in caches.items()},
                    last_logits=self.logits(h[-1:])[0])

    def build(self, scheme, snap, extra=512):
        caches = {}
        for i, (k, v) in snap["kv"].items():
            c = make(scheme, self.Hkv, snap["T"] + extra)
            c.prefill(k, v)
            caches[i] = c
        return caches

    @torch.no_grad()
    def decode_teacher(self, scheme, snap, targets):
        """Feed targets[:-1] one at a time after the prefix; return NLL of targets[1:] and of targets[0] (from prefill logits)."""
        caches = self.build(scheme, snap)
        lin = snap["lin"].clone()
        nll = [float(-torch.log_softmax(snap["last_logits"], -1)[targets[0]])]
        pos = snap["T"]
        for t in range(len(targets) - 1):
            h = self.forward(targets[t:t + 1], pos, caches, lin, prefill=False)
            pos += 1
            nll.append(float(-torch.log_softmax(self.logits(h[-1:])[0], -1)[targets[t + 1]]))
        bits = np.mean([c.read_bits() for c in caches.values()])
        del caches
        return np.array(nll), float(bits)

    @torch.no_grad()
    def decode_answer(self, scheme, snap, question_ids, max_new=16):
        caches = self.build(scheme, snap)
        lin = snap["lin"].clone()
        pos = snap["T"]
        h = self.forward(question_ids, pos, caches, lin, prefill=False)            # token-by-token through the cache
        pos += len(question_ids)
        out = []
        nxt = int(self.logits(h[-1:])[0].argmax())
        for _ in range(max_new):
            out.append(nxt)
            if nxt == self.tok.eos_token_id or self.tok.decode([nxt]).strip() in (".", "\n") and len(out) > 3:
                break
            h = self.forward(torch.tensor([nxt], device=DEV), pos, caches, lin, prefill=False)
            pos += 1
            nxt = int(self.logits(h[-1:])[0].argmax())
        bits = np.mean([c.read_bits() for c in caches.values()])
        del caches
        return self.tok.decode(out), float(bits)


# ----------------------------------------------------------------------------------------------- data
def prose_text():
    from foveakv.text import ensure_wikitext
    ensure_wikitext()
    return open(os.path.join(ROOT, "data", "wikitext2_train.txt")).read()


def code_text():
    import transformers
    base = os.path.join(os.path.dirname(transformers.__file__), "models")    # real Python source, ~2M tokens
    files = sorted(glob.glob(os.path.join(base, "*", "modeling_*.py")))
    random.Random(0).shuffle(files)
    return "".join(f"# file: {os.path.relpath(f, base)}\n" + open(f).read() + "\n" for f in files[:120])


def doc_windows(tok, text, L, n_docs, seed):
    ids = tok(text[: 5 * L * (n_docs + 2)], add_special_tokens=False)["input_ids"]
    rng = random.Random(seed)
    starts = sorted(rng.sample(range(0, len(ids) - L), n_docs))
    return [ids[s:s + L] for s in starts]


WORDS = ["apple", "river", "falcon", "copper", "violet", "harbor", "meadow", "cobalt", "lantern", "glacier", "orchid",
         "saffron", "thunder", "willow", "quartz", "ember", "canyon", "marble", "nebula", "pepper"]


def retrieval_sample(tok, task, L, rng, hay_ids):
    """RULER-style synthetic tasks. Returns (context_ids, question_ids, answers list)."""
    def num():
        return str(rng.randint(1000000, 9999999))
    keys = rng.sample(WORDS, 4)
    if task == "niah":
        needles = [(keys[0], [num()])]
        ask = keys[0]
    elif task == "mk-niah":                       # 4 different keys, ask one (3 distractors)
        needles = [(k, [num()]) for k in keys]
        ask = rng.choice(keys)
    elif task == "mv-niah":                       # one key, 4 values, ask all
        needles = [(keys[0], [num() for _ in range(4)])]
        ask = keys[0]
    sents = []
    for k, vals in needles:
        for v in vals:
            sents.append(tok(f" One of the special magic numbers for {k} is: {v}.", add_special_tokens=False)["input_ids"])
    q_txt = (f"\n\nWhat are all the special magic numbers for {ask} mentioned in the text above? Answer: The special magic numbers for {ask} are:"
             if task == "mv-niah" else
             f"\n\nWhat is the special magic number for {ask} mentioned in the text above? Answer: The special magic number for {ask} is:")
    q = tok(q_txt, add_special_tokens=False)["input_ids"]
    budget = L - sum(len(s) for s in sents)
    start = rng.randint(0, len(hay_ids) - budget - 1)
    hay = hay_ids[start:start + budget]
    pos = sorted(rng.sample(range(budget), len(sents)))
    ctx, last = [], 0
    for p, s in zip(pos, sents):
        ctx += hay[last:p] + s
        last = p
    ctx += hay[last:]
    answers = [v for k, vals in needles if k == ask for v in vals]
    return ctx, q, answers


# ----------------------------------------------------------------------------------------------- experiments
def check(R):
    """My forward (bf16 dense scheme) must reproduce HF's teacher-forced logits."""
    text = prose_text()[:20000]
    ids = R.tok(text, add_special_tokens=False)["input_ids"][:1200]
    x = torch.tensor(ids, device=DEV)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        hf = R.m(input_ids=x[None]).logits[0, 1000:1199].float()
    snap = R.prefill(x[:1000])
    nll, _ = R.decode_teacher("bf16", snap, x[1000:1200])
    hf_nll = -torch.log_softmax(hf, -1).gather(1, x[1001:1200, None])[:, 0].cpu().numpy()
    hf_nll = np.concatenate([[float(-torch.log_softmax(snap["last_logits"], -1)[x[1000]])], hf_nll])
    print(f"mine (prefill+decode, bf16 dense) PPL {np.exp(nll.mean()):.4f} | HF single forward PPL {np.exp(hf_nll.mean()):.4f} | "
          f"max |dNLL| {np.abs(nll - hf_nll).max():.3f}, mean {np.abs(nll - hf_nll).mean():.4f}")
    for sc in ("tq4", "dyn5", "kivi4", "quest5"):
        n2, b = R.decode_teacher(sc, snap, x[1000:1200])
        print(f"  {sc}: PPL {np.exp(n2.mean()):.4f} bits {b:.2f}")


def ppl(R, quick, schemes_override=None, tag=""):
    lens = [16384] if quick else [16384, 32768]
    ndoc = 1 if quick else 3
    N = 64 if quick else 256
    schemes = schemes_override or (["bf16", "tq4", "dyn5"] if quick else SCHEMES)
    out = []
    t0 = time.time()
    for src, text in (("prose", prose_text()), ("code", code_text())):
        for L in lens:
            for d, ids in enumerate(doc_windows(R.tok, text, L, ndoc, seed=L)):
                x = torch.tensor(ids, device=DEV)
                snap = R.prefill(x[:L - N])
                for sc in schemes:
                    nll, bits = R.decode_teacher(sc, snap, x[L - N:])
                    out.append(dict(src=src, L=L, doc=d, scheme=sc, nll=float(nll.mean()), bits=bits))
                    json.dump(out, open(os.path.join(ROOT, "results", "gpu_ppl" + ("_quick" if quick else "") + tag + ".json"), "w"), indent=1)
                    print(f"[{time.time()-t0:5.0f}s] {src} L={L} doc{d} {sc:8s} PPL {math.exp(nll.mean()):8.4f} bits {bits:.2f}", flush=True)
                del snap
                torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(ROOT, "results", "gpu_ppl" + ("_quick" if quick else "") + tag + ".json"), "w"), indent=1)


def retrieval(R, quick, schemes_override=None, tag=""):
    lens = [8192] if quick else [8192, 16384, 32768]
    tasks = ["niah", "mk-niah", "mv-niah"]
    nsamp = 2 if quick else 8
    schemes = schemes_override or (["bf16", "tq4", "dyn5"] if quick else SCHEMES)
    hay = R.tok(prose_text()[:2_000_000], add_special_tokens=False)["input_ids"]
    rng = random.Random(1)
    out = []
    t0 = time.time()
    for L in lens:
        for task in tasks:
            for si in range(nsamp):
                ctx, q, ans = retrieval_sample(R.tok, task, L, rng, hay)
                snap = R.prefill(torch.tensor(ctx, device=DEV))
                qx = torch.tensor(q, device=DEV)
                for sc in schemes:
                    txt, bits = R.decode_answer(sc, snap, qx, max_new=40 if task == "mv-niah" else 12)
                    score = sum(a in txt for a in ans) / len(ans)
                    out.append(dict(L=L, task=task, sample=si, scheme=sc, score=score, bits=bits, out=txt, ans=ans))
                    json.dump(out, open(os.path.join(ROOT, "results", "gpu_retrieval" + ("_quick" if quick else "") + tag + ".json"), "w"), indent=1)
                    print(f"[{time.time()-t0:5.0f}s] L={L} {task:8s} s{si} {sc:8s} score {score:.2f}  {txt.strip()[:60]!r}", flush=True)
                del snap
                torch.cuda.empty_cache()
    json.dump(out, open(os.path.join(ROOT, "results", "gpu_retrieval" + ("_quick" if quick else "") + tag + ".json"), "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "ppl", "retrieval"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--schemes", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    sch = a.schemes.split(",") if a.schemes else None
    ext()
    R = Runner()
    {"check": lambda: check(R), "ppl": lambda: ppl(R, a.quick, sch, a.tag), "retrieval": lambda: retrieval(R, a.quick, sch, a.tag)}[a.what]()
