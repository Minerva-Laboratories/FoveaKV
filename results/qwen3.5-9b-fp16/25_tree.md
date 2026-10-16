# E25 Two-level query-metric tree over keys + precision ladder (sigma = 0.5, rank-2 nodes)

compute = (exact + opened tokens [coarse at 1/4] + node evaluations) / dense, decode, per query head; 'nodes evaluated'
= share of leaf-level nodes whose summary was computed (flat = 1). rho = opened-token budget, fraction of the region.

| mode | budget | compute | nodes evaluated | attn err | resid err |
|---|---|---|---|---|---|
| flat | rho=0.02 | 0.1296 | 1.0000 | 0.0545 | 0.0787 |
| flat +ladder | rho=0.02 | 0.1228 | 1.0000 | 0.0645 | 0.0926 |
| flat | rho=0.05 | 0.1577 | 1.0000 | 0.0322 | 0.0443 |
| flat +ladder | rho=0.05 | 0.1403 | 1.0000 | 0.0385 | 0.0533 |
| flat | rho=0.1 | 0.2050 | 1.0000 | 0.0195 | 0.0253 |
| flat +ladder | rho=0.1 | 0.1695 | 1.0000 | 0.0241 | 0.0319 |
| flat | rho=0.2 | 0.2994 | 1.0000 | 0.0104 | 0.0134 |
| flat +ladder | rho=0.2 | 0.2286 | 1.0000 | 0.0133 | 0.0173 |
| tree Pp=2 | rho=0.02 | 0.0857 | 0.2571 | 0.0643 | 0.0891 |
| tree Pp=2 +ladder | rho=0.02 | 0.0789 | 0.2571 | 0.0738 | 0.1021 |
| tree Pp=2 | rho=0.05 | 0.1137 | 0.2571 | 0.0458 | 0.0614 |
| tree Pp=2 +ladder | rho=0.05 | 0.0964 | 0.2571 | 0.0505 | 0.0679 |
| tree Pp=2 | rho=0.1 | 0.1573 | 0.2571 | 0.0381 | 0.0515 |
| tree Pp=2 +ladder | rho=0.1 | 0.1246 | 0.2571 | 0.0405 | 0.0543 |
| tree Pp=2 | rho=0.2 | 0.2116 | 0.2571 | 0.0352 | 0.0485 |
| tree Pp=2 +ladder | rho=0.2 | 0.1709 | 0.2571 | 0.0358 | 0.0490 |
| tree Pp=4 | rho=0.02 | 0.0976 | 0.4590 | 0.0576 | 0.0819 |
| tree Pp=4 +ladder | rho=0.02 | 0.0909 | 0.4590 | 0.0675 | 0.0957 |
| tree Pp=4 | rho=0.05 | 0.1258 | 0.4590 | 0.0359 | 0.0489 |
| tree Pp=4 +ladder | rho=0.05 | 0.1083 | 0.4590 | 0.0419 | 0.0573 |
| tree Pp=4 | rho=0.1 | 0.1730 | 0.4590 | 0.0242 | 0.0316 |
| tree Pp=4 +ladder | rho=0.1 | 0.1376 | 0.4590 | 0.0282 | 0.0373 |
| tree Pp=4 | rho=0.2 | 0.2668 | 0.4590 | 0.0165 | 0.0221 |
| tree Pp=4 +ladder | rho=0.2 | 0.1964 | 0.4590 | 0.0188 | 0.0248 |
| tree Pp=6 | rho=0.02 | 0.1093 | 0.6569 | 0.0555 | 0.0796 |
| tree Pp=6 +ladder | rho=0.02 | 0.1025 | 0.6569 | 0.0655 | 0.0935 |
| tree Pp=6 | rho=0.05 | 0.1375 | 0.6569 | 0.0333 | 0.0456 |
| tree Pp=6 +ladder | rho=0.05 | 0.1200 | 0.6569 | 0.0396 | 0.0545 |
| tree Pp=6 | rho=0.1 | 0.1847 | 0.6569 | 0.0208 | 0.0270 |
| tree Pp=6 +ladder | rho=0.1 | 0.1493 | 0.6569 | 0.0253 | 0.0334 |
| tree Pp=6 | rho=0.2 | 0.2792 | 0.6569 | 0.0119 | 0.0153 |
| tree Pp=6 +ladder | rho=0.2 | 0.2083 | 0.6569 | 0.0147 | 0.0190 |

runtime 1573s
