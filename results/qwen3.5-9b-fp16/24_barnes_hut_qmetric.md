# E24 One-level Barnes-Hut for exp-dot attention: node construction x opening rule (rank-2 second-cumulant nodes)

compute = (exact + opened tokens + node evaluations) / dense, decode, per query head. rho = token budget for opened
nodes as a fraction of the summarized region.

| nodes | opening | budget | compute | attn err | resid err |
|---|---|---|---|---|---|
| kmeans-q | mass | rho=0.02 | 0.1297 | 0.0525 | 0.0784 |
| kmeans-q | mass | rho=0.05 | 0.1579 | 0.0317 | 0.0459 |
| kmeans-q | mass | rho=0.1 | 0.2051 | 0.0200 | 0.0284 |
| kmeans-q | mass | rho=0.2 | 0.2994 | 0.0112 | 0.0160 |
| kmeans-q | err | rho=0.02 | 0.1294 | 0.1032 | 0.1621 |
| kmeans-q | err | rho=0.05 | 0.1577 | 0.0810 | 0.1294 |
| kmeans-q | err | rho=0.1 | 0.2049 | 0.0619 | 0.1037 |
| kmeans-q | err | rho=0.2 | 0.2994 | 0.0389 | 0.0704 |
| kmeans-q | oracle | rho=0.02 | 0.1297 | 0.0451 | 0.0626 |
| kmeans-q | oracle | rho=0.05 | 0.1579 | 0.0256 | 0.0335 |
| kmeans-q | oracle | rho=0.1 | 0.2050 | 0.0163 | 0.0212 |
| kmeans-q | oracle | rho=0.2 | 0.2993 | 0.0094 | 0.0126 |
| kmeans-q s=0.5 | mass | rho=0.02 | 0.1297 | 0.0539 | 0.0769 |
| kmeans-q s=0.5 | mass | rho=0.05 | 0.1579 | 0.0320 | 0.0442 |
| kmeans-q s=0.5 | mass | rho=0.1 | 0.2051 | 0.0194 | 0.0256 |
| kmeans-q s=0.5 | mass | rho=0.2 | 0.2994 | 0.0102 | 0.0131 |
| kmeans-q s=0.5 | err | rho=0.02 | 0.1294 | 0.0968 | 0.1416 |
| kmeans-q s=0.5 | err | rho=0.05 | 0.1576 | 0.0777 | 0.1155 |
| kmeans-q s=0.5 | err | rho=0.1 | 0.2049 | 0.0582 | 0.0910 |
| kmeans-q s=0.5 | err | rho=0.2 | 0.2994 | 0.0365 | 0.0624 |
| kmeans-q s=0.5 | oracle | rho=0.02 | 0.1298 | 0.0452 | 0.0612 |
| kmeans-q s=0.5 | oracle | rho=0.05 | 0.1580 | 0.0245 | 0.0306 |
| kmeans-q s=0.5 | oracle | rho=0.1 | 0.2051 | 0.0146 | 0.0176 |
| kmeans-q s=0.5 | oracle | rho=0.2 | 0.2994 | 0.0077 | 0.0091 |
| kmeans-q s=0.25 | mass | rho=0.02 | 0.1297 | 0.0638 | 0.0860 |
| kmeans-q s=0.25 | mass | rho=0.05 | 0.1579 | 0.0389 | 0.0503 |
| kmeans-q s=0.25 | mass | rho=0.1 | 0.2051 | 0.0243 | 0.0300 |
| kmeans-q s=0.25 | mass | rho=0.2 | 0.2994 | 0.0129 | 0.0153 |
| kmeans-q s=0.25 | err | rho=0.02 | 0.1294 | 0.1122 | 0.1577 |
| kmeans-q s=0.25 | err | rho=0.05 | 0.1577 | 0.0914 | 0.1297 |
| kmeans-q s=0.25 | err | rho=0.1 | 0.2049 | 0.0694 | 0.1021 |
| kmeans-q s=0.25 | err | rho=0.2 | 0.2994 | 0.0447 | 0.0706 |
| kmeans-q s=0.25 | oracle | rho=0.02 | 0.1297 | 0.0550 | 0.0704 |
| kmeans-q s=0.25 | oracle | rho=0.05 | 0.1580 | 0.0313 | 0.0368 |
| kmeans-q s=0.25 | oracle | rho=0.1 | 0.2051 | 0.0194 | 0.0217 |
| kmeans-q s=0.25 | oracle | rho=0.2 | 0.2994 | 0.0105 | 0.0115 |

runtime 1555s
