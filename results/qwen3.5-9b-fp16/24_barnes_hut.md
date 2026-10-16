# E24 One-level Barnes-Hut for exp-dot attention: node construction x opening rule (rank-2 second-cumulant nodes)

compute = (exact + opened tokens + node evaluations) / dense, decode, per query head. rho = token budget for opened
nodes as a fraction of the summarized region.

| nodes | opening | budget | compute | attn err | resid err |
|---|---|---|---|---|---|
| pos | mass | rho=0.02 | 0.1314 | 0.1363 | 0.1655 |
| pos | mass | rho=0.05 | 0.1556 | 0.1080 | 0.1211 |
| pos | mass | rho=0.1 | 0.2040 | 0.0860 | 0.0897 |
| pos | mass | rho=0.2 | 0.3008 | 0.0633 | 0.0624 |
| pos | err | rho=0.02 | 0.1314 | 0.1741 | 0.2391 |
| pos | err | rho=0.05 | 0.1556 | 0.1328 | 0.1788 |
| pos | err | rho=0.1 | 0.2040 | 0.0949 | 0.1282 |
| pos | err | rho=0.2 | 0.3008 | 0.0558 | 0.0757 |
| pos | oracle | rho=0.02 | 0.1314 | 0.1150 | 0.1427 |
| pos | oracle | rho=0.05 | 0.1556 | 0.0711 | 0.0877 |
| pos | oracle | rho=0.1 | 0.2040 | 0.0407 | 0.0510 |
| pos | oracle | rho=0.2 | 0.3008 | 0.0223 | 0.0273 |
| kmeans | mass | rho=0.02 | 0.1289 | 0.0676 | 0.1080 |
| kmeans | mass | rho=0.05 | 0.1571 | 0.0347 | 0.0495 |
| kmeans | mass | rho=0.1 | 0.2043 | 0.0221 | 0.0301 |
| kmeans | mass | rho=0.2 | 0.2990 | 0.0129 | 0.0170 |
| kmeans | err | rho=0.02 | 0.1284 | 0.0991 | 0.1515 |
| kmeans | err | rho=0.05 | 0.1569 | 0.0524 | 0.0789 |
| kmeans | err | rho=0.1 | 0.2041 | 0.0322 | 0.0480 |
| kmeans | err | rho=0.2 | 0.2988 | 0.0184 | 0.0265 |
| kmeans | oracle | rho=0.02 | 0.1290 | 0.0553 | 0.0824 |
| kmeans | oracle | rho=0.05 | 0.1571 | 0.0266 | 0.0354 |
| kmeans | oracle | rho=0.1 | 0.2043 | 0.0164 | 0.0210 |
| kmeans | oracle | rho=0.2 | 0.2989 | 0.0093 | 0.0118 |
| gain-shape | mass | rho=0.02 | 0.1388 | 0.0875 | 0.1312 |
| gain-shape | mass | rho=0.05 | 0.1667 | 0.0476 | 0.0707 |
| gain-shape | mass | rho=0.1 | 0.2139 | 0.0299 | 0.0435 |
| gain-shape | mass | rho=0.2 | 0.3085 | 0.0169 | 0.0239 |
| gain-shape | err | rho=0.02 | 0.1382 | 0.1340 | 0.2025 |
| gain-shape | err | rho=0.05 | 0.1666 | 0.0741 | 0.1130 |
| gain-shape | err | rho=0.1 | 0.2138 | 0.0462 | 0.0712 |
| gain-shape | err | rho=0.2 | 0.3084 | 0.0257 | 0.0393 |
| gain-shape | oracle | rho=0.02 | 0.1389 | 0.0715 | 0.1034 |
| gain-shape | oracle | rho=0.05 | 0.1669 | 0.0337 | 0.0453 |
| gain-shape | oracle | rho=0.1 | 0.2139 | 0.0201 | 0.0265 |
| gain-shape | oracle | rho=0.2 | 0.3085 | 0.0112 | 0.0148 |

runtime 1932s
