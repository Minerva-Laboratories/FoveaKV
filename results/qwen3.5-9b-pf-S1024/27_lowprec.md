# E27 Dense low-precision attention (SageAttention-style) and an exact-refinement ladder, prefill tiles

compute = relative tensor-core time vs bf16 (int8 2x, int4 4x, fp8 2x; QK and PV half each) + exact refined pairs.

| arm | compute | attn err | resid err |
|---|---|---|---|
| int8 QK / fp8 PV | 0.5000 | 0.0165 | 0.0245 |
| int8 QK / bf16 PV | 0.7500 | 0.0084 | 0.0137 |
| bf16 QK / fp8 PV | 0.7500 | 0.0135 | 0.0201 |
| int8 QK / int8 PV (Ampere) | 0.5000 | 0.0768 | 0.0807 |
| int8 QK / int8 PV + exact top 0.02 (+window) | 0.5470 | 0.0181 | 0.0227 |
| int8 QK / int8 PV + exact top 0.05 (+window) | 0.5771 | 0.0104 | 0.0143 |
| int8 QK / int8 PV + exact top 0.1 (+window) | 0.6271 | 0.0058 | 0.0086 |
| int8 QK / int8 PV + exact 0.02 by error model (+window) | 0.5470 | 0.0142 | 0.0175 |
| int8 QK / int8 PV + exact 0.05 by error model (+window) | 0.5771 | 0.0072 | 0.0092 |
| int8 QK / int8 PV + exact 0.1 by error model (+window) | 0.6271 | 0.0035 | 0.0047 |
| int4 QK / fp8 PV | 0.3750 | 0.1771 | 0.2912 |
| int4 QK / fp8 PV + exact top 0.02 (+window) | 0.4220 | 0.0597 | 0.1081 |
| int4 QK / fp8 PV + exact top 0.05 (+window) | 0.4521 | 0.0360 | 0.0696 |
| int4 QK / fp8 PV + exact top 0.1 (+window) | 0.5021 | 0.0218 | 0.0443 |

runtime 656s
