"""Check that importance weights Σ_i p_i² x_i (and hence the weighted-MSE error predictor) are computable with FlashAttention:
   Σ_i p_i² x_i = exp(lse(2s) - 2 lse(s)) * FA(q, k, x; scale = 2/sqrt(d))
using the logsumexp that the flash kernel returns. Compared with a direct fp64 computation."""
import torch, math
torch.manual_seed(0)
dev = "cuda"
B, H, Lq, T, d = 1, 8, 16, 32768, 256
q = torch.randn(B, H, Lq, d, device=dev, dtype=torch.bfloat16) * 1.5
k = torch.randn(B, H, T, d, device=dev, dtype=torch.bfloat16)
v = torch.randn(B, H, T, d, device=dev, dtype=torch.bfloat16)
x = torch.zeros_like(v)
x[..., 0] = (v.float() ** 2).sum(-1).to(torch.bfloat16)          # per-token ||v||^2
x[..., 1] = 1.0                                                   # ones -> Σ p²
x[..., 2:] = v[..., 2:]                                           # other columns: Σ p² v
fa = torch.ops.aten._scaled_dot_product_flash_attention
s1 = 1 / math.sqrt(d)
out1, lse1 = fa(q, k, v, 0.0, False, False, scale=s1)[:2]
out2, lse2 = fa(q, k, x, 0.0, False, False, scale=2 * s1)[:2]
est = torch.exp(lse2 - 2 * lse1)[..., None] * out2.float()        # Σ p² x
s = (q.double() @ k.double().transpose(-1, -2)) * s1
p = torch.softmax(s, -1)
ref = (p ** 2) @ x.double()
rel = ((est.double() - ref).norm() / ref.norm()).item()
print(f"flash identity: rel err {rel:.2e} (bf16 inputs/outputs); Σp² example {est[0,0,0,1].item():.4e} vs {ref[0,0,0,1].item():.4e}")
# full weighted-MSE ingredient Σ p² ||v - o||² from flash calls (scale 2) with V = v and V = [||v||², 1, 0...]
o = out1.float()
pv = torch.exp(lse2 - 2 * lse1)[..., None] * fa(q, k, v, 0.0, False, False, scale=2 * s1)[0].float()   # Σ p² v
w_fa = est[..., 0] - 2 * (o * pv).sum(-1) + (o ** 2).sum(-1) * est[..., 1]
od = p @ v.double()
w_ref = (p ** 2) @ (v.double() ** 2).sum(-1, keepdim=True)
w_ref = w_ref[..., 0] - 2 * (od * ((p ** 2) @ v.double())).sum(-1) + (od ** 2).sum(-1) * (p ** 2).sum(-1)
print(f"Σ p²||v-o||² via 3 flash calls vs fp64: rel err {((w_fa.double() - w_ref).norm() / w_ref.norm()).item():.2e}")
