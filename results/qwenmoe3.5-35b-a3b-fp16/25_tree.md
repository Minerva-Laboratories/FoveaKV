# E25 Two-level query-metric tree over keys + precision ladder (sigma = 0.5, rank-2 nodes)

compute = (exact + opened tokens [coarse at 1/4] + node evaluations) / dense, decode, per query head; 'nodes evaluated'
= share of leaf-level nodes whose summary was computed (flat = 1). rho = opened-token budget, fraction of the region.

| mode | budget | compute | nodes evaluated | attn err | resid err |
|---|---|---|---|---|---|
| flat | rho=0.02 | 0.1296 | 1.0000 | 0.0535 | 0.0917 |
| flat +ladder | rho=0.02 | 0.1228 | 1.0000 | 0.0603 | 0.1005 |
| flat | rho=0.05 | 0.1580 | 1.0000 | 0.0353 | 0.0612 |
| flat +ladder | rho=0.05 | 0.1403 | 1.0000 | 0.0403 | 0.0686 |
| flat | rho=0.1 | 0.2052 | 1.0000 | 0.0245 | 0.0408 |
| flat +ladder | rho=0.1 | 0.1698 | 1.0000 | 0.0281 | 0.0474 |
| flat | rho=0.2 | 0.2995 | 1.0000 | 0.0153 | 0.0252 |
| flat +ladder | rho=0.2 | 0.2287 | 1.0000 | 0.0180 | 0.0298 |
| tree Pp=2 | rho=0.02 | 0.0868 | 0.2756 | 0.0630 | 0.1029 |
| tree Pp=2 +ladder | rho=0.02 | 0.0800 | 0.2756 | 0.0693 | 0.1110 |
| tree Pp=2 | rho=0.05 | 0.1153 | 0.2756 | 0.0479 | 0.0784 |
| tree Pp=2 +ladder | rho=0.05 | 0.0975 | 0.2756 | 0.0516 | 0.0836 |
| tree Pp=2 | rho=0.1 | 0.1593 | 0.2756 | 0.0411 | 0.0666 |
| tree Pp=2 +ladder | rho=0.1 | 0.1263 | 0.2756 | 0.0430 | 0.0701 |
| tree Pp=2 | rho=0.2 | 0.2173 | 0.2756 | 0.0373 | 0.0607 |
| tree Pp=2 +ladder | rho=0.2 | 0.1738 | 0.2756 | 0.0382 | 0.0621 |
| tree Pp=4 | rho=0.02 | 0.0992 | 0.4850 | 0.0572 | 0.0960 |
| tree Pp=4 +ladder | rho=0.02 | 0.0924 | 0.4850 | 0.0638 | 0.1045 |
| tree Pp=4 | rho=0.05 | 0.1276 | 0.4850 | 0.0397 | 0.0669 |
| tree Pp=4 +ladder | rho=0.05 | 0.1099 | 0.4850 | 0.0443 | 0.0738 |
| tree Pp=4 | rho=0.1 | 0.1747 | 0.4850 | 0.0299 | 0.0492 |
| tree Pp=4 +ladder | rho=0.1 | 0.1394 | 0.4850 | 0.0330 | 0.0548 |
| tree Pp=4 | rho=0.2 | 0.2676 | 0.4850 | 0.0226 | 0.0371 |
| tree Pp=4 +ladder | rho=0.2 | 0.1979 | 0.4850 | 0.0245 | 0.0403 |
| tree Pp=6 | rho=0.02 | 0.1109 | 0.6821 | 0.0549 | 0.0932 |
| tree Pp=6 +ladder | rho=0.02 | 0.1041 | 0.6821 | 0.0616 | 0.1019 |
| tree Pp=6 | rho=0.05 | 0.1393 | 0.6821 | 0.0369 | 0.0631 |
| tree Pp=6 +ladder | rho=0.05 | 0.1215 | 0.6821 | 0.0418 | 0.0704 |
| tree Pp=6 | rho=0.1 | 0.1864 | 0.6821 | 0.0263 | 0.0435 |
| tree Pp=6 +ladder | rho=0.1 | 0.1511 | 0.6821 | 0.0298 | 0.0498 |
| tree Pp=6 | rho=0.2 | 0.2807 | 0.6821 | 0.0178 | 0.0292 |
| tree Pp=6 +ladder | rho=0.2 | 0.2100 | 0.6821 | 0.0202 | 0.0332 |

runtime 1584s
