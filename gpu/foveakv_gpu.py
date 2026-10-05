"""GPU KV-cache schemes for batch-1 decode with custom CUDA kernels (gpu/kernels.cu).

Every cache stores one attention layer: Hkv KV heads, D=256, GQA group G=4. Keys are post-RoPE.
Common interface:  prefill(k, v) ; append(k, v) ; attend(q) -> (Hq, D) fp32 ; read_bits() ; reference(q)
Compressed schemes keep an exact bf16 recent window: tokens are encoded in 128-token blocks once the whole block is
older than W=128 tokens (so the window holds 128..255 tokens). Raw bf16 K/V are kept for re-encoding/window only;
the attention kernels never read them for compressed tokens.

Schemes: Dense (torch SDPA or our bf16 kernel), TurboQuant-MSE (2/3/4 bit, no QJL), GS uniform (2/4 bit),
FoveaKV (2-bit coarse + 2-bit residual for top-k hot tokens, delta-corrected), KIVI (2/4 bit), Quest (bf16 pages)."""
import math, os, sys
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from foveakv.quant import NestedCodebook, lloyd_max          # analytic codebooks (NumPy)
from foveakv.rotate import rht_matrix

D, G, BLK, W = 256, 4, 128, 128
_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.9")
        bd = os.path.join(ROOT, "gpu", "build")
        os.makedirs(bd, exist_ok=True)
        _EXT = load("foveakv_kernels", [os.path.join(ROOT, "gpu", "kernels.cu")], build_directory=bd,
                    extra_cuda_cflags=["-O3", "--use_fast_math", "-allow-unsupported-compiler"], verbose=False)
    return _EXT


_CB = NestedCodebook(4, 2)
LUT16 = torch.tensor(_CB.lut16, dtype=torch.float32)
LUT4N = torch.tensor(_CB.lut4, dtype=torch.float32)          # nested coarse levels
BND16 = torch.tensor(_CB.bnd16, dtype=torch.float32)
_b, _l = lloyd_max(4); BND_TQ2, LUT_TQ2 = torch.tensor(_b, dtype=torch.float32), torch.tensor(_l, dtype=torch.float32)
_b, _l = lloyd_max(8); BND_TQ3, LUT_TQ3 = torch.tensor(_b, dtype=torch.float32), torch.tensor(_l, dtype=torch.float32)


_PASS_FIELDS = ["kb", "vb", "khi", "klo", "vhi", "vlo", "kg8", "vg8", "kglut", "vglut", "kg16", "vg16", "kmu", "vmu",
                "mu_block", "klev", "vlev", "kfine", "vfine", "kc", "vc", "kbits", "ksc", "kzp", "vsc", "vzp", "logits"]
_INT_DEFAULT = {"mu_block": 1, "kfine": 0, "vfine": 0, "kbits": 0}


def run_pass(fmt, q, Hkv, Tcap, t0, n, idx, tpb, boff, part, **kw):
    """fmt 0 = bf16, 1 = gain-shape planes, 2 = KIVI. Writes partials into part=(pm, pl, pacc) from block `boff`."""
    unknown = set(kw) - set(_PASS_FIELDS)
    assert not unknown, unknown
    ext().pass_(fmt, q, Hkv, Tcap, t0, n, idx, tpb, boff, *part, *[kw.get(f, _INT_DEFAULT.get(f)) for f in _PASS_FIELDS])


def tpb_for(n, Hkv, target_blocks=160):
    """tokens per thread block: multiple of 128 (4 warps x 32-token tiles, never straddling a 128-token block)."""
    t = max(128, -(-n * Hkv // target_blocks))
    return -(-t // 128) * 128


def pack2(c):
    """c: (..., D) uint8 in [0,4) -> (..., D/4) uint8, dim 4j+i at bits 2i of byte j."""
    c = c.to(torch.uint8).view(*c.shape[:-1], D // 4, 4)
    return c[..., 0] | (c[..., 1] << 2) | (c[..., 2] << 4) | (c[..., 3] << 6)


def pack4(c):
    c = c.to(torch.uint8).view(*c.shape[:-1], D // 2, 2)
    return c[..., 0] | (c[..., 1] << 4)


def softmax_attend_ref(q, K, V, scale):
    """q (Hq,D), K,V (n,Hkv,D) -> (Hq,D) in fp64."""
    Hkv = K.shape[1]
    qh = q.double().view(Hkv, G, D)
    s = torch.einsum("kgd,nkd->kgn", qh, K.double()) * scale
    p = torch.softmax(s, -1)
    return torch.einsum("kgn,nkd->kgd", p, V.double()).reshape(-1, D)


class Base:
    rotated = False
    name = "base"

    def __init__(self, Hkv, Tmax, device="cuda"):
        self.Hkv, self.Tmax, self.dev = Hkv, Tmax, device
        self.kraw = torch.zeros(Tmax, Hkv, D, dtype=torch.bfloat16, device=device)
        self.vraw = torch.zeros_like(self.kraw)
        self.n = 0
        self.n_comp = 0
        self.scale = D ** -0.5

    def _store(self, k, v):
        t = k.shape[0]
        self.kraw[self.n:self.n + t] = k.to(torch.bfloat16)
        self.vraw[self.n:self.n + t] = v.to(torch.bfloat16)
        self.n += t

    def prefill(self, k, v):
        self._store(k, v)
        self._calibrate()
        self._encode_ready()

    def append(self, k, v):
        self._store(k, v)
        self._encode_ready()

    def _calibrate(self):
        pass

    def _encode_ready(self):
        target = max(0, ((self.n - W) // BLK) * BLK)
        if target > self.n_comp:
            self._encode(self.n_comp, target)
            self.n_comp = target

    def _encode(self, a, b):
        pass

    # ---- window (exact bf16), possibly rotated, as a contiguous (Hkv, nw, D) buffer for the bf16 kernel
    def _window(self):
        kw = self.kraw[self.n_comp:self.n].transpose(0, 1).contiguous()
        vw = self.vraw[self.n_comp:self.n].transpose(0, 1).contiguous()
        return kw, vw

    def _partials(self, nb):
        HG = self.Hkv * G
        return (torch.empty(nb, self.Hkv, G, device=self.dev), torch.empty(nb, self.Hkv, G, device=self.dev),
                torch.empty(nb, self.Hkv, G, D, device=self.dev))

    def _window_pass(self, qk, part, boff, tpb=64):
        kw, vw = self._window()
        nw = kw.shape[1]
        run_pass(0, qk, self.Hkv, nw, 0, nw, None, tpb, boff, part, kb=kw, vb=vw)
        return -(-nw // tpb)

    def _window_stats(self, q, tpb=64):
        """(m, l, acc) of the exact bf16 window in the original domain (unrotated q)."""
        nw = self.n - self.n_comp
        if nw == 0:
            return None
        part = self._partials(-(-nw // tpb))
        qk = (q.float() * self.scale).view(self.Hkv, G, D).contiguous()
        nb = self._window_pass(qk, part, 0, tpb)
        _, m, l, a = ext().combine_partials(*part, nb, False)
        return m, l, a

    @staticmethod
    def _merge(seg_a, seg_b):
        """Merge two softmax segments (m, l, acc) in the same (original) domain -> normalized output."""
        if seg_b is None:
            m, l, a = seg_a
            return a / l[:, None]
        (ma, la, aa), (mb, lb, ab) = seg_a, seg_b
        M = torch.maximum(ma, mb)
        fa, fb = torch.exp(ma - M), torch.exp(mb - M)
        return (aa * fa[:, None] + ab * fb[:, None]) / (la * fa + lb * fb)[:, None]

    def _qk(self, q):
        qs = q.float() * self.scale
        if self.rotated:
            qs = qs @ self.Rt
        return qs.view(self.Hkv, G, D).contiguous()


class Dense(Base):
    """bf16 dense attention. impl='sdpa' (PyTorch flash/efficient SDPA) or 'kernel' (our bf16 kernel)."""

    def __init__(self, Hkv, Tmax, device="cuda", impl="sdpa"):
        super().__init__(Hkv, Tmax, device)
        self.impl = impl
        self.name = f"bf16 dense ({impl})"

    def _encode_ready(self):
        pass

    def attend(self, q):
        n = self.n
        if self.impl == "sdpa":
            K = self.kraw[:n].transpose(0, 1)                                # (Hkv, n, D)
            V = self.vraw[:n].transpose(0, 1)
            qh = q.to(torch.bfloat16).view(self.Hkv, G, 1, D)
            o = torch.nn.functional.scaled_dot_product_attention(qh, K[:, None].expand(-1, G, -1, -1),
                                                                 V[:, None].expand(-1, G, -1, -1))
            return o.view(-1, D).float()
        K = self.kraw[:n].transpose(0, 1).contiguous()
        V = self.vraw[:n].transpose(0, 1).contiguous()
        tpb = tpb_for(n, self.Hkv)
        nb = -(-n // tpb)
        part = self._partials(nb)
        run_pass(0, self._qk(q), self.Hkv, n, 0, n, None, tpb, 0, part, kb=K, vb=V)
        return ext().combine_partials(*part, nb, True)[0]

    def read_bits(self):
        return 16.0

    def reference(self, q):
        return softmax_attend_ref(q, self.kraw[:self.n].float(), self.vraw[:self.n].float(), self.scale)


class GainShape(Base):
    """Gain-shape cache in the rotated domain (shared by TurboQuant-MSE, uniform GS and FoveaKV).
    mode='tq'  : TurboQuant-MSE — no mean, fp16 norm, Lloyd-Max b-bit codebook (b in 2,3,4)
    mode='gs'  : mean removal (global prefill mean or 128-token block means), 8-bit log gain, nested 4/2-bit codebook."""
    rotated = True

    def __init__(self, Hkv, Tmax, device="cuda", mode="gs", bits=4, block_mean=True, gain_bits=8, seed=0, shrink=None):
        super().__init__(Hkv, Tmax, device)
        self.mode, self.bits, self.block_mean, self.gain_bits = mode, bits, block_mean, gain_bits
        # shrinkage correction (paper §nested): key shapes renormalized (x 1/sqrt(1-D)), value residuals unbiased (x 1/(1-D));
        # default on for our gain-shape code, off for TurboQuant (published method)
        self.shrink = (mode == "gs") if shrink is None else shrink
        R = torch.tensor(rht_matrix(D, seed), dtype=torch.float32, device=device)
        self.Rt = R.t().contiguous()                                   # y = x @ R^T
        self.R = R
        cap = math.ceil(Tmax / BLK) * BLK
        self.cap = cap
        z8 = lambda *s: torch.zeros(*s, dtype=torch.uint8, device=device)
        self.khi, self.klo, self.vhi, self.vlo = z8(Hkv, cap, D // 4), z8(Hkv, cap, D // 4), z8(Hkv, cap, D // 4), z8(Hkv, cap, D // 4)
        self.k3 = self.v3 = None                                       # 3-bit TurboQuant: reference path only
        if mode == "tq":
            self.kg16 = torch.zeros(Hkv, cap, dtype=torch.float16, device=device)
            self.vg16 = torch.zeros_like(self.kg16)
            self.kmu = self.vmu = None
        else:
            self.kg8, self.vg8 = z8(Hkv, cap), z8(Hkv, cap)
            nmu = cap // BLK if block_mean else 1
            self.kmu = torch.zeros(Hkv, nmu, D, dtype=torch.float16, device=device)
            self.vmu = torch.zeros_like(self.kmu)
        dev = device
        self.lut16, self.lut4n = LUT16.to(dev), LUT4N.to(dev)
        self.bnd16 = BND16.to(dev)
        Dc, Df = float(_CB.mse_coarse), float(_CB.mse_fine)
        sk = (lambda D: (1 - D) ** -0.5) if self.shrink else (lambda D: 1.0)
        sv = (lambda D: 1 / (1 - D)) if self.shrink else (lambda D: 1.0)
        self.k4, self.k16 = (self.lut4n * sk(Dc)).contiguous(), (self.lut16 * sk(Df)).contiguous()
        self.v4, self.v16 = (self.lut4n * sv(Dc)).contiguous(), (self.lut16 * sv(Df)).contiguous()
        self.name = (f"TurboQuant-MSE {bits}b" if mode == "tq" else f"GS {bits}b" + (" blkmean" if block_mean else ""))

    def _calibrate(self):
        if self.mode == "tq":
            return
        x = {"k": self.kraw[:self.n].float(), "v": self.vraw[:self.n].float()}
        self.mu_glob = {n: t.mean(0) for n, t in x.items()}             # (Hkv, D)
        self.glo, self.ghi = {}, {}
        for n, t in x.items():
            mu = self._means(n, t, 0, self.n) if self.block_mean else self.mu_glob[n][None]
            g = torch.linalg.norm((t - mu), dim=-1)                       # rotation preserves norm
            lg = torch.log(g.clamp_min(1e-30))
            self.glo[n], self.ghi[n] = lg.min(0).values, lg.max(0).values  # (Hkv,)
        nlev = 1 << self.gain_bits
        self.glut = {n: (torch.exp(self.glo[n][:, None] + (torch.arange(nlev, device=self.dev)[None] + 0.5) / nlev
                                   * (self.ghi[n] - self.glo[n])[:, None]) / 16.0).float().contiguous() for n in x}
        if not self.block_mean:
            self.kmu[:, 0] = (self.mu_glob["k"] @ self.Rt).half()
            self.vmu[:, 0] = (self.mu_glob["v"] @ self.Rt).half()

    def _means(self, name, t, a, b):
        """per-token block means for tokens [a,b) (fp16-rounded), t = raw tokens [a,b)."""
        nbk = (b - a) // BLK
        M = torch.empty_like(t)
        if nbk:
            M[:nbk * BLK] = t[:nbk * BLK].view(nbk, BLK, self.Hkv, D).mean(1).repeat_interleave(BLK, 0)
        if (b - a) % BLK:
            M[nbk * BLK:] = t[nbk * BLK:].mean(0)
        return M.half().float()

    def _encode(self, a, b):
        for name in ("k", "v"):
            x = (self.kraw if name == "k" else self.vraw)[a:b].float()          # (n, Hkv, D)
            if self.mode == "tq":
                y = x @ self.Rt
            else:
                if self.block_mean:
                    M = self._means(name, x, a, b)
                    mus = (M[::BLK] @ self.Rt).half()                             # (nblocks, Hkv, D) rotated
                    (self.kmu if name == "k" else self.vmu)[:, a // BLK:b // BLK] = mus.transpose(0, 1)
                    y = (x - M) @ self.Rt
                else:
                    y = (x - self.mu_glob[name]) @ self.Rt
            g = torch.linalg.norm(y, dim=-1)                                       # (n, Hkv)
            z = y / g.clamp_min(1e-30)[..., None] * 16.0
            if self.mode == "tq":
                gg = g.half()
                (self.kg16 if name == "k" else self.vg16)[:, a:b] = gg.t()
                if self.bits == 2:
                    c = torch.bucketize(z, BND_TQ2.to(self.dev))
                    hi, lo = c, torch.zeros_like(c)
                elif self.bits == 4:
                    c = torch.bucketize(z, self.bnd16)
                    hi, lo = c >> 2, c & 3
                else:   # 3-bit: keep dequantized values for the reference path
                    c = torch.bucketize(z, BND_TQ3.to(self.dev))
                    rec = LUT_TQ3.to(self.dev)[c] / 16.0 * gg.float()[..., None]
                    buf = torch.zeros(self.cap, self.Hkv, D, device=self.dev) if (self.k3 if name == "k" else self.v3) is None \
                        else (self.k3 if name == "k" else self.v3)
                    buf[a:b] = rec
                    if name == "k": self.k3 = buf
                    else: self.v3 = buf
                    continue
            else:
                n_lev = 1 << self.gain_bits
                lg = torch.log(g.clamp_min(1e-30))
                lo_, hi_ = self.glo[name], self.ghi[name]
                gc = ((lg - lo_) / (hi_ - lo_).clamp_min(1e-12) * n_lev).floor().clamp(0, n_lev - 1).to(torch.uint8)
                (self.kg8 if name == "k" else self.vg8)[:, a:b] = gc.t()
                c = torch.bucketize(z, self.bnd16)
                hi, lo = c >> 2, c & 3
            (self.khi if name == "k" else self.vhi)[:, a:b] = pack2(hi).transpose(0, 1)
            (self.klo if name == "k" else self.vlo)[:, a:b] = pack2(lo).transpose(0, 1)

    # ------------------------------------------------------------------ kernel launch helpers
    def _gs_args(self, kfine, vfine, klev, vlev):
        a = dict(khi=self.khi, klo=self.klo, vhi=self.vhi, vlo=self.vlo, klev=klev, vlev=vlev, kfine=kfine, vfine=vfine)
        if self.mode == "tq":
            a.update(kg16=self.kg16, vg16=self.vg16, mu_block=self.cap)
        else:
            a.update(kg8=self.kg8, vg8=self.vg8, kglut=self.glut["k"], vglut=self.glut["v"], kmu=self.kmu, vmu=self.vmu,
                     mu_block=BLK if self.block_mean else self.cap)
        return a

    def _levels(self, fine):
        """-> (key LUT, value LUT, fine flag)"""
        if self.mode == "tq":
            lv = LUT_TQ2.to(self.dev) if self.bits == 2 else self.lut16
            return lv, lv, int(self.bits == 4)
        return (self.k16, self.v16, 1) if fine else (self.k4, self.v4, 0)

    def attend(self, q, tpb=None):
        tpb = tpb or tpb_for(self.n_comp, self.Hkv)
        if self.mode == "tq" and self.bits == 3:
            return self.reference(q).float()
        qk = self._qk(q)
        klev, vlev, fine = self._levels(self.bits == 4)
        nb_c = -(-self.n_comp // tpb)
        win = self._window_stats(q)
        if nb_c == 0:
            return self._merge(win, None)
        part = self._partials(nb_c)
        run_pass(1, qk, self.Hkv, self.cap, 0, self.n_comp, None, tpb, 0, part, **self._gs_args(fine, fine, klev, vlev))
        _, m, l, a = ext().combine_partials(*part, nb_c, False)
        return self._merge((m, l, a @ self.R), win)

    def read_bits(self):
        nw = self.n - self.n_comp
        if self.mode == "tq":
            b = self.bits + 16 / D
        else:
            b = self.bits + self.gain_bits / D + (16 / BLK if self.block_mean else 0)
        return (b * self.n_comp + 16 * nw) / self.n

    # ------------------------------------------------------------------ dequantized reference
    def dequant(self, fine_mask=None):
        """Full K,V (n_comp, Hkv, D) fp32 in the ORIGINAL domain. fine_mask (Hkv, n_comp) bool: fine for those,
        coarse otherwise (FoveaKV); None -> this scheme's own precision."""
        n = self.n_comp
        out = []
        for name in ("k", "v"):
            if self.mode == "tq" and self.bits == 3:
                out.append((self.k3 if name == "k" else self.v3)[:n] @ self.R)
                continue
            hi = getattr(self, name + "hi")[:, :n]
            lo = getattr(self, name + "lo")[:, :n]
            unpack = lambda p: torch.stack([(p >> (2 * i)) & 3 for i in range(4)], -1).view(self.Hkv, n, D).long()
            ch, cl = unpack(hi), unpack(lo)
            if self.mode == "tq":
                lev = LUT_TQ2.to(self.dev)[ch] if self.bits == 2 else self.lut16[(ch << 2) | cl]
                g = getattr(self, name + "g16")[:, :n].float()[..., None] / 16.0
                y = g * lev
            else:
                fine = (self.k16 if name == "k" else self.v16)[(ch << 2) | cl]
                coarse = (self.k4 if name == "k" else self.v4)[ch]
                if fine_mask is None:
                    lev = fine if self.bits == 4 else coarse
                else:
                    lev = torch.where(fine_mask[..., None], fine, coarse)
                g = self.glut[name][torch.arange(self.Hkv, device=self.dev)[:, None], getattr(self, name + "g8")[:, :n].long()][..., None]
                mu = getattr(self, name + "mu")
                mu = mu.float().repeat_interleave(BLK, 1)[:, :n] if self.block_mean else mu.float()
                y = g * lev + mu
            out.append((y @ self.R).transpose(0, 1))                         # (n, Hkv, D)
        return out

    def reference(self, q, fine_mask=None):
        K, V = self.dequant(fine_mask)
        K = torch.cat([K, self.kraw[self.n_comp:self.n].float()])
        V = torch.cat([V, self.vraw[self.n_comp:self.n].float()])
        return softmax_attend_ref(q, K, V, self.scale)


class FoveaKV(GainShape):
    """FoveaKV: 2-bit coarse plane for all compressed tokens + 2-bit residual plane for the top-k hot tokens (per KV head,
    shared by the GQA group: max over the group of estimated log-prob). Exact bf16 recent window outside the budget.
    Two kernels: coarse pass (writes logits for selection) and a delta-correction pass over hot tokens."""

    def __init__(self, Hkv, Tmax, device="cuda", k_frac=0.05, block_mean=True, **kw):
        super().__init__(Hkv, Tmax, device, mode="gs", bits=2, block_mean=block_mean, **kw)
        self.name_suffix = "" if self.shrink else " noshrink"
        self.k_frac = k_frac
        self.logits = torch.empty(Hkv, G, self.cap, device=device)
        self.name = f"FoveaKV k={k_frac*100:g}%" + (" blkmean" if block_mean else "") + self.name_suffix
        self.last_hot = None

    def attend(self, q, tpb=None):
        tpb = tpb or tpb_for(self.n_comp, self.Hkv)
        e = ext()
        qk = self._qk(q)
        nb_c = -(-self.n_comp // tpb)
        win = self._window_stats(q)
        self.last_hot = None
        if nb_c == 0:
            return self._merge(win, None)
        part = self._partials(nb_c)
        run_pass(1, qk, self.Hkv, self.cap, 0, self.n_comp, None, tpb, 0, part,
                 logits=self.logits, **self._gs_args(0, 0, self.k4, self.v4))
        _, m1, l1, acc1 = e.combine_partials(*part, nb_c, False)
        k = min(self.n_comp, math.ceil(self.k_frac * self.n))
        if k:
            # estimated log-prob uses the full normalizer (compressed coarse + exact window)
            if win is not None:
                M = torch.maximum(m1, win[0])
                lse = M + torch.log(l1 * torch.exp(m1 - M) + win[1] * torch.exp(win[0] - M))
            else:
                lse = m1 + torch.log(l1)
            sel = (self.logits[:, :, :self.n_comp] - lse.view(self.Hkv, G, 1)).amax(1)     # (Hkv, n_comp)
            idx = torch.topk(sel, k, dim=-1, sorted=False).indices.int().contiguous()
            self.last_hot = idx
            tpb_d = 64
            nbd = -(-k // tpb_d)
            _, pl, pacc = self._partials(nbd)
            e.delta(qk, self.Hkv, self.cap, idx, tpb_d, m1.view(self.Hkv, G).contiguous(), pl, pacc,
                    self.khi, self.klo, self.vhi, self.vlo, self.kg8, self.vg8, self.glut["k"], self.glut["v"],
                    self.kmu, self.vmu, BLK if self.block_mean else self.cap, self.k4, self.k16, self.v4, self.v16)
            l1 = l1 + pl.view(nbd, -1).sum(0)
            acc1 = acc1 + pacc.view(nbd, -1, D).sum(0)
        return self._merge((m1, l1, acc1 @ self.R), win)

    def read_bits(self):
        nw = self.n - self.n_comp
        b = 2 + self.gain_bits / D + (16 / BLK if self.block_mean else 0)
        k = min(self.n_comp, math.ceil(self.k_frac * self.n))
        return (b * self.n_comp + 2 * k + 16 * nw) / self.n

    def reference(self, q):
        mask = torch.zeros(self.Hkv, self.n_comp, dtype=torch.bool, device=self.dev)
        if self.last_hot is not None:
            mask.scatter_(1, self.last_hot.long(), True)
        return GainShape.reference(self, q, mask)


class KIVI(Base):
    """KIVI: keys per-channel asymmetric over 128-token blocks, values per-token asymmetric over 128-channel groups,
    b in {2,4}; fp16 scale+zero; recent window bf16 (KIVI's full-precision residual)."""

    def __init__(self, Hkv, Tmax, device="cuda", bits=2):
        super().__init__(Hkv, Tmax, device)
        self.bits = bits
        cap = math.ceil(Tmax / BLK) * BLK
        self.cap = cap
        nbyte = D * bits // 8
        self.kc = torch.zeros(Hkv, cap, nbyte, dtype=torch.uint8, device=device)
        self.vc = torch.zeros_like(self.kc)
        self.ksc = torch.zeros(Hkv, cap // BLK, D, dtype=torch.float16, device=device)
        self.kzp = torch.zeros_like(self.ksc)
        self.vsc = torch.zeros(Hkv, cap, D // 128, dtype=torch.float16, device=device)
        self.vzp = torch.zeros_like(self.vsc)
        self.name = f"KIVI {bits}b"

    def _encode(self, a, b):
        L = (1 << self.bits) - 1
        pk = pack2 if self.bits == 2 else pack4
        x = self.kraw[a:b].float().view(-1, BLK, self.Hkv, D)                   # (nb, BLK, Hkv, D)
        lo, hi = x.amin(1), x.amax(1)
        s = ((hi - lo) / L).clamp_min(1e-8)
        s16, z16 = s.half(), lo.half()
        c = torch.round((x - z16.float()[:, None]) / s16.float()[:, None]).clamp(0, L)
        self.kc[:, a:b] = pk(c.view(-1, self.Hkv, D)).transpose(0, 1)
        self.ksc[:, a // BLK:b // BLK] = s16.transpose(0, 1)
        self.kzp[:, a // BLK:b // BLK] = z16.transpose(0, 1)
        v = self.vraw[a:b].float().view(-1, self.Hkv, D // 128, 128)
        lo, hi = v.amin(-1), v.amax(-1)
        s = ((hi - lo) / L).clamp_min(1e-8)
        s16, z16 = s.half(), lo.half()
        c = torch.round((v - z16.float()[..., None]) / s16.float()[..., None]).clamp(0, L)
        self.vc[:, a:b] = pk(c.view(-1, self.Hkv, D)).transpose(0, 1)
        self.vsc[:, a:b] = s16.transpose(0, 1)
        self.vzp[:, a:b] = z16.transpose(0, 1)

    def attend(self, q, tpb=None):
        tpb = tpb or tpb_for(self.n_comp, self.Hkv)
        qk = self._qk(q)
        nb_c = -(-self.n_comp // tpb)
        nw = self.n - self.n_comp
        part = self._partials(nb_c + -(-nw // 64) + 1)
        if nb_c:
            run_pass(2, qk, self.Hkv, self.cap, 0, self.n_comp, None, tpb, 0, part, kc=self.kc, vc=self.vc, kbits=self.bits,
                     ksc=self.ksc, kzp=self.kzp, vsc=self.vsc, vzp=self.vzp)
        nb_w = self._window_pass(qk, part, nb_c)
        return ext().combine_partials(*part, nb_c + nb_w, True)[0]

    def read_bits(self):
        b = self.bits + 0.5 * (32 / BLK) + 0.5 * (32 / 128)
        return (b * self.n_comp + 16 * (self.n - self.n_comp)) / self.n

    def reference(self, q):
        n = self.n_comp
        unp = (lambda p: torch.stack([(p >> (2 * i)) & 3 for i in range(4)], -1)) if self.bits == 2 else \
              (lambda p: torch.stack([(p >> (4 * i)) & 15 for i in range(2)], -1))
        ck = unp(self.kc[:, :n]).view(self.Hkv, n, D).float()
        K = ck * self.ksc.float().repeat_interleave(BLK, 1)[:, :n] + self.kzp.float().repeat_interleave(BLK, 1)[:, :n]
        cv = unp(self.vc[:, :n]).view(self.Hkv, n, D // 128, 128).float()
        V = (cv * self.vsc[:, :n].float()[..., None] + self.vzp[:, :n].float()[..., None]).view(self.Hkv, n, D)
        K = torch.cat([K.transpose(0, 1), self.kraw[n:self.n].float()])
        V = torch.cat([V.transpose(0, 1), self.vraw[n:self.n].float()])
        return softmax_attend_ref(q, K, V, self.scale)


class Quest(Base):
    """Quest: bf16 KV in 16-token pages, per-page channel min/max of keys; per step score pages by the upper bound
    sum_c max(q_c*kmax_c, q_c*kmin_c) (max over the GQA group), attend over the top pages (token budget k) + window."""
    PAGE = 16

    def __init__(self, Hkv, Tmax, device="cuda", k_frac=0.05):
        super().__init__(Hkv, Tmax, device)
        self.k_frac = k_frac
        cap = math.ceil(Tmax / BLK) * BLK
        self.kmax = torch.zeros(Hkv, cap // self.PAGE, D, dtype=torch.bfloat16, device=device)
        self.kmin = torch.zeros_like(self.kmax)
        self.Kt = torch.zeros(Hkv, cap, D, dtype=torch.bfloat16, device=device)   # head-major copy for the kernel
        self.Vt = torch.zeros_like(self.Kt)
        self.name = f"Quest k={k_frac*100:g}%"
        self.last_idx = None

    def _encode(self, a, b):
        x = self.kraw[a:b].view(-1, self.PAGE, self.Hkv, D)
        self.kmax[:, a // self.PAGE:b // self.PAGE] = x.amax(1).transpose(0, 1)
        self.kmin[:, a // self.PAGE:b // self.PAGE] = x.amin(1).transpose(0, 1)
        self.Kt[:, a:b] = self.kraw[a:b].transpose(0, 1)
        self.Vt[:, a:b] = self.vraw[a:b].transpose(0, 1)

    def attend(self, q, tpb=64):
        qk = self._qk(q)
        npg = self.n_comp // self.PAGE
        kp = min(npg, max(1, math.ceil(self.k_frac * self.n / self.PAGE))) if npg else 0
        nw = self.n - self.n_comp
        nb_s = -(-(kp * self.PAGE) // tpb)
        part = self._partials(nb_s + -(-nw // 64) + 1)
        if kp:
            # upper bound sum_c max(q_c kmax_c, q_c kmin_c) = kmax . relu(q) + kmin . min(q, 0)  (two small GEMMs)
            qp, qn = qk.clamp_min(0).transpose(1, 2), qk.clamp_max(0).transpose(1, 2)        # (Hkv, D, G)
            ub = (torch.bmm(self.kmax[:, :npg].float(), qp) + torch.bmm(self.kmin[:, :npg].float(), qn)).transpose(1, 2)
            pages = torch.topk(ub.amax(1), kp, dim=-1, sorted=False).indices      # (Hkv, kp)
            idx = (pages[..., None] * self.PAGE + torch.arange(self.PAGE, device=self.dev)).view(self.Hkv, -1).int().contiguous()
            self.last_idx = idx
            run_pass(0, qk, self.Hkv, self.Kt.shape[1], 0, idx.shape[1], idx, tpb, 0, part, kb=self.Kt, vb=self.Vt)
        nb_w = self._window_pass(qk, part, nb_s)
        return ext().combine_partials(*part, nb_s + nb_w, True)[0]

    def read_bits(self):
        # metadata (min+max bf16 per page per channel, keys only) + selected pages bf16 (K and V) + window
        npg = self.n_comp // self.PAGE
        kp = min(npg, max(1, math.ceil(self.k_frac * self.n / self.PAGE))) if npg else 0
        meta = 0.5 * 2 * 16 / self.PAGE * self.n_comp
        return (meta + 16 * kp * self.PAGE + 16 * (self.n - self.n_comp)) / self.n

    def reference(self, q):
        if self.last_idx is None:
            K, V = self.kraw[self.n_comp:self.n].float(), self.vraw[self.n_comp:self.n].float()
            return softmax_attend_ref(q, K, V, self.scale)
        outs = []
        for h in range(self.Hkv):
            ii = self.last_idx[h].long()
            K = torch.cat([self.kraw[ii, h], self.kraw[self.n_comp:self.n, h]]).float()[:, None]
            V = torch.cat([self.vraw[ii, h], self.vraw[self.n_comp:self.n, h]]).float()[:, None]
            outs.append(softmax_attend_ref(q.view(self.Hkv, G, D)[h].reshape(G, D), K, V, self.scale))
        return torch.cat(outs)


def make(name, Hkv, Tmax, device="cuda"):
    """Factory for benchmark/eval scheme names."""
    table = {
        "bf16": lambda: Dense(Hkv, Tmax, device, "sdpa"),
        "bf16-kernel": lambda: Dense(Hkv, Tmax, device, "kernel"),
        "tq2": lambda: GainShape(Hkv, Tmax, device, "tq", 2),
        "tq3": lambda: GainShape(Hkv, Tmax, device, "tq", 3),
        "tq4": lambda: GainShape(Hkv, Tmax, device, "tq", 4),
        "gs2": lambda: GainShape(Hkv, Tmax, device, "gs", 2),
        "gs4": lambda: GainShape(Hkv, Tmax, device, "gs", 4),
        "gs4-noshrink": lambda: GainShape(Hkv, Tmax, device, "gs", 4, shrink=False),
        "kivi2": lambda: KIVI(Hkv, Tmax, device, 2),
        "kivi4": lambda: KIVI(Hkv, Tmax, device, 4),
    }
    for kf in (0.02, 0.05, 0.10):
        table[f"dyn{int(kf*100)}"] = (lambda kf=kf: FoveaKV(Hkv, Tmax, device, kf))
        table[f"dyn{int(kf*100)}-gmean"] = (lambda kf=kf: FoveaKV(Hkv, Tmax, device, kf, block_mean=False))
        table[f"dyn{int(kf*100)}-noshrink"] = (lambda kf=kf: FoveaKV(Hkv, Tmax, device, kf, shrink=False))
        table[f"quest{int(kf*100)}"] = (lambda kf=kf: Quest(Hkv, Tmax, device, kf))
    return table[name]()
