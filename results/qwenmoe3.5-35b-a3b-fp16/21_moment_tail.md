# E21 Query-dependent moment summaries for the attention tail (block 64)

Exact tier: 4 sinks + first block, 128-token window, tokens outside full blocks. Every other block is either read
exactly (refined, chosen per query and shared by the GQA group) or served by its summary evaluated for that query.
'bytes' = K/V values moved per decode step relative to a full fp16 read (summaries included); 'refined' = fraction of
blocks read exactly; err = relative error of the attention output vs exact attention.

## Mass estimate (mass-weighted mean |log mass error| of the block summaries, nats)
| summary | |d log mass| | signed (+ = over) |
|---|---|---|
| mean | 2.1701 | -2.1701 |
| mom0 | 1.2118 | -1.1158 |
| mom2 | 0.8651 | -0.4111 |
| mom8 | 0.6950 | -0.0241 |

## All runs
| summary | selector | budget | bytes | refined | attn err |
|---|---|---|---|---|---|
| drop | bound | rho=0.03125 | 0.1035 | 0.0342 | 0.1529 |
| drop | bound | rho=0.0625 | 0.1276 | 0.0598 | 0.1302 |
| drop | bound | rho=0.125 | 0.1922 | 0.1282 | 0.0909 |
| drop | bound | rho=0.25 | 0.3051 | 0.2479 | 0.0577 |
| mean | mass | rho=0.03125 | 0.1035 | 0.0342 | 0.0947 |
| mean | mass | rho=0.0625 | 0.1276 | 0.0598 | 0.0796 |
| mean | mass | rho=0.125 | 0.1922 | 0.1282 | 0.0582 |
| mean | mass | rho=0.25 | 0.3051 | 0.2479 | 0.0391 |
| mom0 | mass | rho=0.03125 | 0.1182 | 0.0342 | 0.0860 |
| mom0 | mass | rho=0.0625 | 0.1424 | 0.0598 | 0.0730 |
| mom0 | mass | rho=0.125 | 0.2069 | 0.1282 | 0.0539 |
| mom0 | mass | rho=0.25 | 0.3198 | 0.2479 | 0.0366 |
| mom0 | err | rho=0.03125 | 0.1182 | 0.0342 | 0.0858 |
| mom0 | err | rho=0.0625 | 0.1424 | 0.0598 | 0.0728 |
| mom0 | err | rho=0.125 | 0.2069 | 0.1282 | 0.0537 |
| mom0 | err | rho=0.25 | 0.3198 | 0.2479 | 0.0362 |
| mom0 | err | tau=0.1 | 0.2185 | 0.1404 | 0.0777 |
| mom0 | err | tau=0.03 | 0.3167 | 0.2445 | 0.0492 |
| mom0 | err | tau=0.01 | 0.4127 | 0.3462 | 0.0319 |
| mom0 | err | tau=0.003 | 0.5218 | 0.4619 | 0.0206 |
| mom0 no v-tilt | mass | rho=0.03125 | 0.1182 | 0.0342 | 0.0860 |
| mom0 no v-tilt | mass | rho=0.0625 | 0.1424 | 0.0598 | 0.0730 |
| mom0 no v-tilt | mass | rho=0.125 | 0.2069 | 0.1282 | 0.0539 |
| mom0 no v-tilt | mass | rho=0.25 | 0.3198 | 0.2479 | 0.0366 |
| mom2 | mass | rho=0.03125 | 0.1478 | 0.0342 | 0.0923 |
| mom2 | mass | rho=0.0625 | 0.1720 | 0.0598 | 0.0712 |
| mom2 | mass | rho=0.125 | 0.2365 | 0.1282 | 0.0487 |
| mom2 | mass | rho=0.25 | 0.3494 | 0.2479 | 0.0321 |
| mom2 | err | rho=0.03125 | 0.1478 | 0.0342 | 0.0928 |
| mom2 | err | rho=0.0625 | 0.1720 | 0.0598 | 0.0716 |
| mom2 | err | rho=0.125 | 0.2365 | 0.1282 | 0.0490 |
| mom2 | err | rho=0.25 | 0.3494 | 0.2479 | 0.0324 |
| mom2 | err | tau=0.1 | 0.2648 | 0.1582 | 0.0740 |
| mom2 | err | tau=0.03 | 0.3600 | 0.2591 | 0.0513 |
| mom2 | err | tau=0.01 | 0.4489 | 0.3533 | 0.0376 |
| mom2 | err | tau=0.003 | 0.5470 | 0.4573 | 0.0268 |
| mom2 no v-tilt | mass | rho=0.03125 | 0.1330 | 0.0342 | 0.0950 |
| mom2 no v-tilt | mass | rho=0.0625 | 0.1572 | 0.0598 | 0.0771 |
| mom2 no v-tilt | mass | rho=0.125 | 0.2217 | 0.1282 | 0.0541 |
| mom2 no v-tilt | mass | rho=0.25 | 0.3346 | 0.2479 | 0.0359 |
| mom8 | mass | rho=0.03125 | 0.2364 | 0.0342 | 0.0907 |
| mom8 | mass | rho=0.0625 | 0.2606 | 0.0598 | 0.0681 |
| mom8 | mass | rho=0.125 | 0.3251 | 0.1282 | 0.0446 |
| mom8 | mass | rho=0.25 | 0.4380 | 0.2479 | 0.0283 |
| mom8 | err | rho=0.03125 | 0.2364 | 0.0342 | 0.0903 |
| mom8 | err | rho=0.0625 | 0.2606 | 0.0598 | 0.0679 |
| mom8 | err | rho=0.125 | 0.3251 | 0.1282 | 0.0444 |
| mom8 | err | rho=0.25 | 0.4380 | 0.2479 | 0.0280 |
| mom8 | err | tau=0.1 | 0.3766 | 0.1828 | 0.0681 |
| mom8 | err | tau=0.03 | 0.4742 | 0.2862 | 0.0473 |
| mom8 | err | tau=0.01 | 0.5636 | 0.3810 | 0.0345 |
| mom8 | err | tau=0.003 | 0.6608 | 0.4840 | 0.0243 |
| mom8 no v-tilt | mass | rho=0.03125 | 0.1774 | 0.0342 | 0.1025 |
| mom8 no v-tilt | mass | rho=0.0625 | 0.2016 | 0.0598 | 0.0823 |
| mom8 no v-tilt | mass | rho=0.125 | 0.2661 | 0.1282 | 0.0564 |
| mom8 no v-tilt | mass | rho=0.25 | 0.3790 | 0.2479 | 0.0366 |
| mom8 | oracle | rho=0.03125 | 0.2364 | 0.0342 | 0.0805 |
| mom8 | oracle | rho=0.0625 | 0.2606 | 0.0598 | 0.0562 |
| mom8 | oracle | rho=0.125 | 0.3251 | 0.1282 | 0.0326 |
| mom8 | oracle | rho=0.25 | 0.4380 | 0.2479 | 0.0177 |
| mom8+o1 | mass | rho=0.03125 | 0.2506 | 0.0342 | 0.0872 |
| mom8+o1 | mass | rho=0.0625 | 0.2745 | 0.0598 | 0.0658 |
| mom8+o1 | mass | rho=0.125 | 0.3380 | 0.1282 | 0.0428 |
| mom8+o1 | mass | rho=0.25 | 0.4491 | 0.2479 | 0.0273 |
| mom8+o1 | err | rho=0.03125 | 0.2506 | 0.0342 | 0.0867 |
| mom8+o1 | err | rho=0.0625 | 0.2745 | 0.0598 | 0.0654 |
| mom8+o1 | err | rho=0.125 | 0.3380 | 0.1282 | 0.0425 |
| mom8+o1 | err | rho=0.25 | 0.4491 | 0.2479 | 0.0270 |
| mom8+o1 | err | tau=0.1 | 0.3868 | 0.1808 | 0.0659 |
| mom8+o1 | err | tau=0.03 | 0.4829 | 0.2842 | 0.0451 |
| mom8+o1 | err | tau=0.01 | 0.5710 | 0.3791 | 0.0324 |
| mom8+o1 | err | tau=0.003 | 0.6668 | 0.4822 | 0.0220 |
| mom8+o1 no v-tilt | mass | rho=0.03125 | 0.1917 | 0.0342 | 0.0989 |
| mom8+o1 no v-tilt | mass | rho=0.0625 | 0.2155 | 0.0598 | 0.0796 |
| mom8+o1 no v-tilt | mass | rho=0.125 | 0.2790 | 0.1282 | 0.0544 |
| mom8+o1 no v-tilt | mass | rho=0.25 | 0.3901 | 0.2479 | 0.0354 |

runtime 4645s
