# E9 End-to-end (quantized cache in the loop, Qwen3.5-0.8B NumPy forward)

**TL;DR:** window PPL (prose / code): int8 ref 21.255 / 4.265; GS-4b 21.412 / 4.254; DYN k=5% 21.386 / 4.286 (-0.12% / +0.74% vs GS-4b); DYN k=10% -0.20% / +0.24%.
Go/no-go "+1% PPL vs uniform 4-bit at ≤10% hot": PASS (worst corpus +0.24% at k=10%). Needle exact-match (10/50/90% depth): GS-4b ✓ ✓ ✓, DYN k=5% ✓ ✓ ✓.
Only 512 decode positions per corpus at 8K context — small-sample PPL; differences < ~0.5% are within noise.

## Config
- Prefill P = N − window tokens with exact attention (shared by all schemes); window = 512 (prose, code) or question+answer (needle).
- Reference cache = int8 per-token/head absmax of the NumPy model's K/V. Schemes quantize from it; calibration (mu, gain range) on prefill only.
- "fp32 (no cache quant)" row: window K/V are fp32, prefill K/V are the int8 reference (fp32 prefill KV not stored) — near-reference sanity row.
- DYN: coarse/sign selection, budget mode (GQA-shared), S=4, W=64. KIVI: per-channel K / per-token V, group 128.

| scheme | prose PPL | Δ vs GS-4b % | code PPL | Δ vs GS-4b % | needle 10/50/90 | answer NLL |
|---|---|---|---|---|---|---|
| fp32 (no cache quant) | 21.2424 | -0.79 | 4.2694 | +0.35 | ✓ ✓ ✓ | 0.017 |
| int8 (reference) | 21.2552 | -0.73 | 4.2654 | +0.26 | ✓ ✓ ✓ | 0.017 |
| GS-4b | 21.4116 | +0.00 | 4.2545 | +0.00 | ✓ ✓ ✓ | 0.020 |
| GS-2b | 23.1720 | +8.22 | 4.6751 | +9.89 | ✓ ✓ ✓ | 0.045 |
| DYN coarse k=2% | 21.5091 | +0.46 | 4.2578 | +0.08 | ✓ ✓ ✓ | 0.019 |
| DYN coarse k=5% | 21.3865 | -0.12 | 4.2860 | +0.74 | ✓ ✓ ✓ | 0.019 |
| DYN coarse k=10% | 21.3693 | -0.20 | 4.2647 | +0.24 | ✓ ✓ ✓ | 0.019 |
| DYN sign k=5% | 21.4081 | -0.02 | 4.2629 | +0.20 | ✓ ✓ ✓ | 0.019 |
| KIVI-4b g128 | 21.2329 | -0.83 | 4.3007 | +1.09 | ✓ ✓ ✓ | 0.018 |
| KIVI-2b g128 | 31.0356 | +44.95 | 6.0912 | +43.17 | ✗ ✗ ✓ | 0.240 |

## Needle greedy outputs
- needle10 / fp32 (no cache quant): `' 7-4-1-9-2'` (NLL 0.017)
- needle10 / int8 (reference): `' 7-4-1-9-2'` (NLL 0.017)
- needle10 / GS-4b: `' 7-4-1-9-2'` (NLL 0.018)
- needle10 / GS-2b: `' 7-4-1-9-2'` (NLL 0.042)
- needle10 / DYN coarse k=2%: `' 7-4-1-9-2'` (NLL 0.017)
- needle10 / DYN coarse k=5%: `' 7-4-1-9-2'` (NLL 0.018)
- needle10 / DYN coarse k=10%: `' 7-4-1-9-2'` (NLL 0.018)
- needle10 / DYN sign k=5%: `' 7-4-1-9-2'` (NLL 0.019)
- needle10 / KIVI-4b g128: `' 7-4-1-9-2'` (NLL 0.017)
- needle10 / KIVI-2b g128: `' 7-4-1-9.2'` (NLL 0.270)
- needle50 / fp32 (no cache quant): `' 7-4-1-9-2'` (NLL 0.016)
- needle50 / int8 (reference): `' 7-4-1-9-2'` (NLL 0.016)
- needle50 / GS-4b: `' 7-4-1-9-2'` (NLL 0.019)
- needle50 / GS-2b: `' 7-4-1-9-2'` (NLL 0.042)
- needle50 / DYN coarse k=2%: `' 7-4-1-9-2'` (NLL 0.018)
- needle50 / DYN coarse k=5%: `' 7-4-1-9-2'` (NLL 0.018)
- needle50 / DYN coarse k=10%: `' 7-4-1-9-2'` (NLL 0.017)
- needle50 / DYN sign k=5%: `' 7-4-1-9-2'` (NLL 0.017)
- needle50 / KIVI-4b g128: `' 7-4-1-9-2'` (NLL 0.015)
- needle50 / KIVI-2b g128: `' **7-4-1-9-2'` (NLL 0.281)
- needle90 / fp32 (no cache quant): `' 7-4-1-9-2'` (NLL 0.017)
- needle90 / int8 (reference): `' 7-4-1-9-2'` (NLL 0.017)
- needle90 / GS-4b: `' 7-4-1-9-2'` (NLL 0.023)
- needle90 / GS-2b: `' 7-4-1-9-2'` (NLL 0.052)
- needle90 / DYN coarse k=2%: `' 7-4-1-9-2'` (NLL 0.021)
- needle90 / DYN coarse k=5%: `' 7-4-1-9-2'` (NLL 0.022)
- needle90 / DYN coarse k=10%: `' 7-4-1-9-2'` (NLL 0.022)
- needle90 / DYN sign k=5%: `' 7-4-1-9-2'` (NLL 0.021)
- needle90 / KIVI-4b g128: `' 7-4-1-9-2'` (NLL 0.022)
- needle90 / KIVI-2b g128: `' 7-4-1-9-2'` (NLL 0.167)

## Caveats
- Single model (0.8B), 8K context, 2 text corpora × 512 positions: PPL deltas below ~0.5% are not significant.
- Needle at 8K is easy for this model; harder retrieval (longer context, multiple needles) is untested.
runtime 1398s
