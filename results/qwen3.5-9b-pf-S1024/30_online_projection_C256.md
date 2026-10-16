# E30 Online learning of the cache projection (causal, chunks of 256), keys r = 64 / 192, values r = 128 / 256

Relative attention-output error (pre-gate), mean over ['prose', 'code'] x layers [11, 27] x heads [0, 1, 2, 3]; evaluation as E29.
"own" = each chunk read with the basis it was written with; "latest" = all codes read through the newest basis.
Per-chunk bases cost r x d bf16 per chunk and KV head: keys 96 B/token on top of 128 B of codes,
values 256 B/token on top of 256 B ("local" = exact PCA of each chunk alone, the limit of fast Oja).

|  | arm | attn err |
|---|---|---|
| k | batch prompt (E29 qw) | 0.0854 |
| k | static (other corpus, pca) | 0.1933 |
| k | ideal gamma=0.97 own | 0.0745 |
| k | ideal gamma=0.97 latest | 0.4504 |
| k | power gamma=0.97 own | 0.0786 |
| k | power gamma=0.97 latest | 0.2653 |
| k | local gamma=0.97 own | 0.0343 |
| k | local gamma=0.97 latest | 0.6014 |
| k | oja eta=0.5 gamma=0.97 own | 0.0937 |
| k | oja eta=0.5 gamma=0.97 latest | 0.5125 |
| k | oja eta=2 gamma=0.97 own | 0.0650 |
| k | oja eta=2 gamma=0.97 latest | 0.5162 |
| k | oja eta=8 gamma=0.97 own | 0.0519 |
| k | oja eta=8 gamma=0.97 latest | 0.5047 |
| k | oja_s eta=0.5 gamma=0.97 own | 0.1018 |
| k | oja_s eta=0.5 gamma=0.97 latest | 0.4386 |
| k | oja_s eta=2 gamma=0.97 own | 0.0703 |
| k | oja_s eta=2 gamma=0.97 latest | 0.4996 |
| k | oja_s eta=8 gamma=0.97 own | 0.0557 |
| k | oja_s eta=8 gamma=0.97 latest | 0.5100 |
| k | ideal gamma=1 own | 0.0781 |
| k | ideal gamma=1 latest | 0.4170 |
| k | power gamma=1 own | 0.0820 |
| k | power gamma=1 latest | 0.2610 |
| k | oja eta=0.5 gamma=1 own | 0.0941 |
| k | oja eta=0.5 gamma=1 latest | 0.5640 |
| k | oja eta=2 gamma=1 own | 0.0653 |
| k | oja eta=2 gamma=1 latest | 0.5509 |
| k | oja eta=8 gamma=1 own | 0.0523 |
| k | oja eta=8 gamma=1 latest | 0.5480 |
| k | oja_s eta=0.5 gamma=1 own | 0.1030 |
| k | oja_s eta=0.5 gamma=1 latest | 0.5131 |
| k | oja_s eta=2 gamma=1 own | 0.0712 |
| k | oja_s eta=2 gamma=1 latest | 0.5341 |
| k | oja_s eta=8 gamma=1 own | 0.0564 |
| k | oja_s eta=8 gamma=1 latest | 0.5425 |
| v | batch prompt (E29 opca) | 0.0587 |
| v | static (other corpus, pca) | 0.2109 |
| v | ideal gamma=0.97 own | 0.0570 |
| v | ideal gamma=0.97 latest | 0.3429 |
| v | power gamma=0.97 own | 0.0643 |
| v | power gamma=0.97 latest | 0.2826 |
| v | local gamma=0.97 own | 0.0214 |
| v | local gamma=0.97 latest | 0.3843 |
| v | oja eta=0.5 gamma=0.97 own | 0.0518 |
| v | oja eta=0.5 gamma=0.97 latest | 0.2997 |
| v | oja eta=2 gamma=0.97 own | 0.0409 |
| v | oja eta=2 gamma=0.97 latest | 0.3132 |
| v | oja eta=8 gamma=0.97 own | 0.0331 |
| v | oja eta=8 gamma=0.97 latest | 0.3197 |
| v | oja_s eta=0.5 gamma=0.97 own | 0.0521 |
| v | oja_s eta=0.5 gamma=0.97 latest | 0.2906 |
| v | oja_s eta=2 gamma=0.97 own | 0.0410 |
| v | oja_s eta=2 gamma=0.97 latest | 0.3106 |
| v | oja_s eta=8 gamma=0.97 own | 0.0332 |
| v | oja_s eta=8 gamma=0.97 latest | 0.3191 |
| v | ideal gamma=1 own | 0.0614 |
| v | ideal gamma=1 latest | 0.3429 |
| v | power gamma=1 own | 0.0685 |
| v | power gamma=1 latest | 0.2559 |
| v | oja eta=0.5 gamma=1 own | 0.0519 |
| v | oja eta=0.5 gamma=1 latest | 0.3012 |
| v | oja eta=2 gamma=1 own | 0.0410 |
| v | oja eta=2 gamma=1 latest | 0.3172 |
| v | oja eta=8 gamma=1 own | 0.0332 |
| v | oja eta=8 gamma=1 latest | 0.3219 |
| v | oja_s eta=0.5 gamma=1 own | 0.0522 |
| v | oja_s eta=0.5 gamma=1 latest | 0.3030 |
| v | oja_s eta=2 gamma=1 own | 0.0411 |
| v | oja_s eta=2 gamma=1 latest | 0.3160 |
| v | oja_s eta=8 gamma=1 own | 0.0332 |
| v | oja_s eta=8 gamma=1 latest | 0.3202 |

runtime 870s
