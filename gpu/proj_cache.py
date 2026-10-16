"""Sequence-fitted low-rank cache projections in the model (accuracy reference in PyTorch; E29 in the decode loop).

Per attention layer and KV head, at the end of prefill: the position-free key dims (64..255; RoPE dims 0..63 stay as
they are) of tokens in [SINK, T - W) are replaced by their rank-rk reconstruction, values of the same tokens by their
rank-rv reconstruction. Sinks, the last W prefill tokens and every decoded token stay exact (bf16).
Bases:
  keys   'static' (key PCA of a calibration document), 'kpca' (key PCA of this prompt), 'qw' (whitened by this prompt's
         query metric E[q q^T]: SVD of M^1/2 S_k^1/2, encoder U^T M^1/2, decoder M^-1/2 U)
  values 'static' (value PCA of the calibration document), 'vpca' (this prompt), 'opca' (PCA of this prompt's attention
         OUTPUTS: the last 2048 prefill queries, exact attention)
codes: None (bf16 latent) or (bits_latent, bits_rope) for the equal-bytes arms: the latent (and the RoPE dims of keys)
coded with FoveaKV's gain-shape codes (random rotation, Lloyd-Max 4-bit fine / nested 2-bit coarse, fp32 gain)."""
import math
import torch
from gpu.foveakv_gpu import Base, D, G, LUT16, LUT4N, BND16

ROT, SINK, W_ = 64, 4, 128


def _psd_pow(A, p, eps=1e-6):
    w, U = torch.linalg.eigh((A + A.T) / 2)
    w = w.clamp_min(eps * float(w.max()))
    return (U * w ** p) @ U.T


def _top(S, r):
    w, U = torch.linalg.eigh((S + S.T) / 2)
    return U[:, -r:].flip(1)


def gs_code(X, bits, seed):
    """Gain-shape code of the rows of X (n, d) fp32: random rotation, unit shape scaled by sqrt(d), Lloyd-Max levels
    (4-bit fine, or the nested 2-bit coarse cells), exact gain. Returns the reconstruction."""
    n, d = X.shape
    g = torch.Generator(device="cpu").manual_seed(seed)
    R = torch.linalg.qr(torch.randn(d, d, generator=g))[0].to(X.device)
    mu = X.mean(0)
    Y = (X - mu) @ R.T
    gain = Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
    z = Y / gain * math.sqrt(d)
    idx = torch.bucketize(z, BND16.to(X.device))
    lev = LUT16.to(X.device)[idx] if bits == 4 else LUT4N.to(X.device)[idx >> 2]
    return (lev * gain / math.sqrt(d)) @ R + mu


class Proj(Base):
    full_raw = True

    def __init__(self, Hkv, Tmax, device="cuda", kb=None, rk=96, vb=None, rv=128, codes=None):
        super().__init__(Hkv, Tmax, device)
        self.kb, self.rk, self.vb, self.rv, self.codes = kb, rk, vb, rv, codes
        self.name = f"proj k={kb}:{rk} v={vb}:{rv}" + (f" codes={codes}" if codes else "")

    def build(self, qm, qbuf, static=None, layer_seed=0):
        """qm (Hkv, D, D) prompt E[q q^T]; qbuf [(pos0, q (Hq, T, D) bf16)] recent prefill queries; static {h: (Uk, muk,
        Uv, muv)} calibration bases."""
        T0 = self.n
        a, b = SINK, T0 - W_
        for h in range(self.Hkv):
            K = self.kraw[:T0, h].float()
            V = self.vraw[:T0, h].float()
            Kn = K[:, ROT:]
            if self.kb is not None:
                if self.kb == "static":
                    Uk, muk = static[h][0][:, :self.rk], static[h][1]
                    E_, D_ = Uk.T, Uk
                else:
                    muk = Kn.mean(0)
                    Sk = torch.cov((Kn - muk).T, correction=0)
                    if self.kb == "kpca":
                        Uk = _top(Sk, self.rk); E_, D_ = Uk.T, Uk
                    else:
                        M = qm[h][ROT:, ROT:].float()
                        Mh, Mi = _psd_pow(M, .5), _psd_pow(M, -.5)
                        Uk = _top(Mh @ Sk @ Mh, self.rk)
                        E_, D_ = Uk.T @ Mh, Mi @ Uk
                Z = (Kn[a:b] - muk) @ E_.T
                if self.codes:
                    Z = gs_code(Z, self.codes[0], 1000 * layer_seed + h)
                Kh = Z @ D_.T + muk
                Kr = K[a:b, :ROT]
                if self.codes:
                    Kr = gs_code(Kr, self.codes[1], 1000 * layer_seed + 100 + h)
                self.kraw[a:b, h] = torch.cat([Kr, Kh], 1).to(self.kraw.dtype)
            if self.vb is not None:
                if self.vb == "static":
                    Uv, muv = static[h][2][:, :self.rv], static[h][3]
                else:
                    muv = V.mean(0)
                    if self.vb == "vpca":
                        Uv = _top(torch.cov((V - muv).T, correction=0), self.rv)
                    else:
                        O = []
                        for p0, q in qbuf:                                         # q (Hq, T, D)
                            qh = q[h * G:(h + 1) * G].float()                      # (G, T, D)
                            n = p0 + qh.shape[1]
                            s = qh @ K[:n].T * self.scale
                            pos = torch.arange(p0, n, device=s.device)[:, None]
                            s = s.masked_fill(torch.arange(n, device=s.device)[None] > pos, -math.inf)
                            O.append((s.softmax(-1) @ V[:n]).reshape(-1, D))
                        O = torch.cat(O)
                        Uv = _top(torch.cov((O - muv).T, correction=0), self.rv)
                Z = (V[a:b] - muv) @ Uv
                if self.codes:
                    Z = gs_code(Z, self.codes[0], 1000 * layer_seed + 200 + h)
                self.vraw[a:b, h] = (Z @ Uv.T + muv).to(self.vraw.dtype)

    def attend(self, q):
        n = self.n
        K = self.kraw[:n].transpose(0, 1).float()                                 # (Hkv, n, D)
        V = self.vraw[:n].transpose(0, 1).float()
        qh = q.float().view(self.Hkv, G, D)
        p = (qh @ K.transpose(1, 2) * self.scale).softmax(-1)
        return (p @ V).reshape(-1, D)

    def read_bits(self):
        return 16.0


def static_bases(kv, Hkv, rmax=256):
    """{layer: {h: (Uk (192, rmax), muk, Uv (256, rmax), muv)}} from a calibration prefill's raw K/V."""
    out = {}
    for i, (k, v) in kv.items():
        out[i] = {}
        for h in range(Hkv):
            Kn = k[:, h, ROT:].float(); V = v[:, h].float()
            muk, muv = Kn.mean(0), V.mean(0)
            out[i][h] = (_top(torch.cov((Kn - muk).T, correction=0), min(rmax, D - ROT)), muk,
                         _top(torch.cov((V - muv).T, correction=0), min(rmax, D)), muv)
    return out
