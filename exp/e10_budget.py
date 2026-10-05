"""E10 Budget model: per-step PCIe bytes/time for residual fetches vs an estimated (memory-bound) decode step,
for 32K/128K/256K contexts and the local Qwen3.5 configs (0.8B measured; 2B/9B extrapolated from config.json).
MEASURED inputs: hit rates from E6/E7 (results/e06_07_raw.npz). ESTIMATED: everything else (bandwidths, step time)."""
from common import *
import glob, json

PCIE = 25e9                 # B/s, PCIe 4.0 x16 (assumption)
HBM = 192e9                 # B/s, RTX 4050 Laptop (this machine) — assumption for step time
W_BITS = {"bf16": 16, "Q4": 4.5}
CTX = [32768, 131072, 262144]
KF = 0.05
G_BITS = 8

raw = np.load(os.path.join(RES, "e06_07_raw.npz"), allow_pickle=True)
A6, A7 = raw["e6"], raw["e7"]
a6 = A6[A6[:, 3] == KF]
a7 = A7[A7[:, 3] == KF]
HITS = {"no reuse (all dynamic hot fetched)": 0.0,
        "LRU 2×k (E6, measured)": float(a6[:, 6].astype(float).mean()),
        "LRU 4×k (E6, measured)": float(a6[:, 7].astype(float).mean()),
        "prefetch (a)∪(c) (E7, measured)": float(a7[:, 8].astype(float).mean()),
        "target": 0.85}


def cfg(name):
    if name == "0.8B":
        f = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*/config.json"))[0]
    else:
        f = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/models--Qwen--Qwen3.5-{name}-Base/snapshots/*/config.json"))[0]
    c = json.load(open(f))
    return c.get("text_config", c)


def model_numbers(c):
    H, I, V = c["hidden_size"], c["intermediate_size"], c["vocab_size"]
    hd, Hq, Hkv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
    nk, nv, dk, dv = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"]
    n_attn = sum(t == "full_attention" for t in c["layer_types"])
    n_lin = c["num_hidden_layers"] - n_attn
    attn = 2 * Hq * hd * H + 2 * Hkv * hd * H + Hq * hd * H
    lin = (2 * nk * dk + nv * dv) * H + 2 * nv * dv * H + 2 * nv * H
    mlp = 3 * H * I
    params = n_attn * attn + n_lin * lin + c["num_hidden_layers"] * mlp + V * H     # tied head read each step
    state_bytes = n_lin * nv * dk * dv * 4                                          # fp32 recurrent states
    return dict(params=params, n_attn=n_attn, Hkv=Hkv, hd=hd, state=state_bytes)


rows = []
for mname in ("0.8B", "2B", "9B"):
    m = model_numbers(cfg(mname))
    for T in CTX:
        elems = m["n_attn"] * m["Hkv"] * m["hd"] * T * 2                            # K and V elements
        coarse_B = elems * (2 + G_BITS / m["hd"]) / 8
        u4_B = elems * (4 + G_BITS / m["hd"]) / 8
        hot_B = elems * KF * 2 / 8                                                  # residual plane of hot tokens
        for wq, wb in W_BITS.items():
            w_B = m["params"] * wb / 8
            step_dyn = (w_B + coarse_B + hot_B + m["state"]) / HBM
            step_u4 = (w_B + u4_B + m["state"]) / HBM
            for hname, hit in HITS.items():
                pcie_B = hot_B * (1 - hit)
                t_pcie = pcie_B / PCIE
                rows.append((mname, f"{T//1024}K", wq, hname, coarse_B / 2**20, u4_B / 2**20, pcie_B / 2**20,
                             t_pcie * 1e3, step_dyn * 1e3, step_u4 * 1e3, t_pcie / step_dyn))

sel = [r for r in rows if r[2] == "Q4" and r[3] in ("LRU 4×k (E6, measured)", "prefetch (a)∪(c) (E7, measured)", "target")]
key = [r for r in rows if r[0] == "0.8B" and r[1] == "128K" and r[2] == "Q4"]
best = max(HITS[k] for k in HITS if "measured" in k)
r128 = [r for r in rows if r[1] == "128K" and r[2] == "Q4" and HITS[r[3]] == best]
write("10_budget.md", f"""# E10 Budget model (PCIe residual fetches vs estimated decode step)

**TL;DR:** measured hit rates at k=5%: LRU 2× {HITS['LRU 2×k (E6, measured)']:.3f}, LRU 4× {HITS['LRU 4×k (E6, measured)']:.3f}, prefetch (a)∪(c) {HITS['prefetch (a)∪(c) (E7, measured)']:.3f} (target 0.85).
With the best measured hit rate ({best:.3f}) at 128K and Q4 weights, PCIe time is {', '.join(f'{r[0]}: {r[10]*100:.0f}%' for r in r128)} of the estimated decode step (criterion ≤ 20%).
Hit rates are measured at 8K context and assumed to hold at 32K–256K — this is the main extrapolation.

## Config (ESTIMATED unless noted)
- PCIe {PCIE/1e9:.0f} GB/s; HBM {HBM/1e9:.0f} GB/s (RTX 4050 Laptop); step time = bytes read / HBM (memory-bound decode, batch 1).
- Step bytes (DYN): weights + coarse K/V (2 + {G_BITS}/{256} bits) + hot residual planes (2 bits × k={KF*100:.0f}%) + fp32 DeltaNet states.
- PCIe bytes = (1 − hit) × hot residual bytes. Hit rates MEASURED on Qwen3.5-0.8B at 8K (E6/E7); applied unchanged to 2B/9B.
- Model sizes from local config.json (2B/9B: only configs are on disk; their attention statistics are not measured).

## Main table (Q4 weights)
""" + md_table(["model", "ctx", "weights", "hit source", "coarse KV MiB", "uniform-4b KV MiB", "PCIe MiB/step", "PCIe ms", "DYN step ms", "U4 step ms", "PCIe/step"], sel, "{:.2f}") + """
## All rows
""" + md_table(["model", "ctx", "weights", "hit source", "coarse KV MiB", "uniform-4b KV MiB", "PCIe MiB/step", "PCIe ms", "DYN step ms", "U4 step ms", "PCIe/step"], rows, "{:.2f}") + """
## Caveats
- PCIe transfers can overlap with compute of the 3 linear-attention layers between attention layers only if they are predicted in time (E7); misses are counted as fully serialized here (pessimistic) but PCIe latency/small-transfer inefficiency is ignored (optimistic).
- k=5% of a 128K context is ~6.5K hot tokens per head; whether 5% suffices at long context is not measured (E2 is at 8K).
""")
