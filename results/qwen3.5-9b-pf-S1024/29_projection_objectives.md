# E29 Sequence-fitted low-rank cache projections: objectives and rank x bits at equal bytes

9B prefill capture, ['prose', 'code'] x layers [11, 27] x KV heads [0, 1, 2, 3]; prompt [0, 7680), evaluation = 512 queries after it;
sinks 4 + window 128 exact. Errors: relative attention-output error (pre-gate) and residual error (gate, o_proj),
mean over evaluation queries, then over (corpus, layer, head). Keys: 192 position-free dims projected (RoPE 64 exact).

## Key objectives (bf16 latent)
| arm | r | bits | attn err | resid err |
|---|---|---|---|---|
| kpca | 32 | 16 | 0.1591 | 0.2410 |
| static | 32 | 16 | 0.2577 | 0.3798 |
| qw | 32 | 16 | 0.1407 | 0.1982 |
| jdiag | 32 | 16 | 0.1293 | 0.1846 |
| jfull | 32 | 16 | 0.1265 | 0.1803 |
| kpca | 64 | 16 | 0.1134 | 0.1674 |
| static | 64 | 16 | 0.1933 | 0.2763 |
| qw | 64 | 16 | 0.0854 | 0.1191 |
| jdiag | 64 | 16 | 0.0820 | 0.1161 |
| jfull | 64 | 16 | 0.0811 | 0.1150 |
| kpca | 96 | 16 | 0.0777 | 0.1139 |
| static | 96 | 16 | 0.1419 | 0.1976 |
| qw | 96 | 16 | 0.0557 | 0.0765 |
| jdiag | 96 | 16 | 0.0542 | 0.0761 |
| jfull | 96 | 16 | 0.0530 | 0.0744 |

## Value objectives (bf16 latent)
| arm | r | bits | attn err | resid err |
|---|---|---|---|---|
| vpca | 32 | 16 | 0.1744 | 0.2834 |
| vstatic | 32 | 16 | 0.3072 | 0.4722 |
| opca | 32 | 16 | 0.1269 | 0.1981 |
| orecent | 32 | 16 | 0.1291 | 0.1999 |
| ow | 32 | 16 | 0.1303 | 0.1962 |
| oracle | 32 | 16 | 0.0935 | 0.1469 |
| vpca | 64 | 16 | 0.1337 | 0.2214 |
| vstatic | 64 | 16 | 0.2730 | 0.4148 |
| opca | 64 | 16 | 0.0901 | 0.1419 |
| orecent | 64 | 16 | 0.0923 | 0.1433 |
| ow | 64 | 16 | 0.0935 | 0.1389 |
| oracle | 64 | 16 | 0.0585 | 0.0929 |
| vpca | 128 | 16 | 0.0827 | 0.1443 |
| vstatic | 128 | 16 | 0.2109 | 0.3168 |
| opca | 128 | 16 | 0.0517 | 0.0823 |
| orecent | 128 | 16 | 0.0539 | 0.0844 |
| ow | 128 | 16 | 0.0553 | 0.0793 |
| oracle | 128 | 16 | 0.0283 | 0.0452 |

## Equal bytes: keys (bits per head-token for the 192 dims)
| arm | r | bits | attn err | resid err |
|---|---|---|---|---|
| eqbytes384:full | 192 | 2 | 0.0971 | 0.1502 |
| eqbytes384:kpca | 96 | 4 | 0.0812 | 0.1198 |
| eqbytes384:qw | 96 | 4 | 0.0735 | 0.1032 |
| eqbytes384:jfull | 96 | 4 | 0.0681 | 0.1023 |
| eqbytes384:kpca | 48 | 8 | 0.1318 | 0.1960 |
| eqbytes384:qw | 48 | 8 | 0.1084 | 0.1515 |
| eqbytes384:jfull | 48 | 8 | 0.0996 | 0.1415 |
| eqbytes384:kpca | 24 | 16 | 0.1778 | 0.2731 |
| eqbytes384:qw | 24 | 16 | 0.1610 | 0.2282 |
| eqbytes384:jfull | 24 | 16 | 0.1481 | 0.2117 |
| eqbytes768:full | 192 | 4 | 0.0176 | 0.0271 |
| eqbytes768:kpca | 96 | 8 | 0.0777 | 0.1138 |
| eqbytes768:qw | 96 | 8 | 0.0561 | 0.0776 |
| eqbytes768:jfull | 96 | 8 | 0.0530 | 0.0745 |
| eqbytes768:kpca | 48 | 16 | 0.1318 | 0.1961 |
| eqbytes768:qw | 48 | 16 | 0.1084 | 0.1514 |
| eqbytes768:jfull | 48 | 16 | 0.0995 | 0.1415 |

## Equal bytes: values (bits per head-token for 256 dims)
| arm | r | bits | attn err | resid err |
|---|---|---|---|---|
| eqbytes512:full | 256 | 2 | 0.0696 | 0.1247 |
| eqbytes512:vpca | 128 | 4 | 0.0840 | 0.1467 |
| eqbytes512:opca | 128 | 4 | 0.0542 | 0.0883 |
| eqbytes512:vpca | 64 | 8 | 0.1337 | 0.2214 |
| eqbytes512:opca | 64 | 8 | 0.0902 | 0.1419 |
| eqbytes512:vpca | 32 | 16 | 0.1744 | 0.2834 |
| eqbytes512:opca | 32 | 16 | 0.1269 | 0.1981 |
| eqbytes1024:full | 256 | 4 | 0.0148 | 0.0261 |
| eqbytes1024:vpca | 128 | 8 | 0.0827 | 0.1444 |
| eqbytes1024:opca | 128 | 8 | 0.0517 | 0.0825 |
| eqbytes1024:vpca | 64 | 16 | 0.1337 | 0.2214 |
| eqbytes1024:opca | 64 | 16 | 0.0901 | 0.1419 |

## First-order check: J_q (I - B) q vs measured output error (key truncation, r = 64)
| corpus | layer | head | basis | median cos(pred, meas) | median |pred|/|meas| |
|---|---|---|---|---|---|
| prose | 11 | 0 | kpca | 0.9654 | 1.5497 |
| prose | 11 | 0 | kpca | 0.9754 | 1.4567 |
| prose | 11 | 0 | kpca | 0.9644 | 1.6678 |
| prose | 11 | 0 | kpca | 0.9488 | 2.9917 |
| prose | 11 | 0 | jfull | 0.9849 | 1.3034 |
| prose | 11 | 0 | jfull | 0.9904 | 1.3753 |
| prose | 11 | 0 | jfull | 0.9894 | 1.4458 |
| prose | 11 | 0 | jfull | 0.9947 | 1.3232 |
| prose | 11 | 1 | kpca | 0.9430 | 1.4806 |
| prose | 11 | 1 | kpca | 0.9433 | 1.5358 |
| prose | 11 | 1 | kpca | 0.9831 | 1.7771 |
| prose | 11 | 1 | kpca | 0.9744 | 1.2796 |
| prose | 11 | 1 | jfull | 0.9918 | 1.1752 |
| prose | 11 | 1 | jfull | 0.9869 | 1.2850 |
| prose | 11 | 1 | jfull | 0.9907 | 1.5232 |
| prose | 11 | 1 | jfull | 0.9875 | 1.2200 |
| prose | 11 | 2 | kpca | 0.7995 | 3.6877 |
| prose | 11 | 2 | kpca | 0.8723 | 2.9842 |
| prose | 11 | 2 | kpca | 0.9202 | 2.8490 |
| prose | 11 | 2 | kpca | 0.8135 | 2.5215 |
| prose | 11 | 2 | jfull | 0.9864 | 1.4520 |
| prose | 11 | 2 | jfull | 0.9770 | 1.2052 |
| prose | 11 | 2 | jfull | 0.9913 | 1.7305 |
| prose | 11 | 2 | jfull | 0.9931 | 1.3020 |
| prose | 11 | 3 | kpca | 0.9192 | 4.5787 |
| prose | 11 | 3 | kpca | 0.8948 | 4.3557 |
| prose | 11 | 3 | kpca | 0.8565 | 2.1064 |
| prose | 11 | 3 | kpca | 0.9065 | 2.7265 |
| prose | 11 | 3 | jfull | 0.9930 | 1.2886 |
| prose | 11 | 3 | jfull | 0.9887 | 1.4458 |
| prose | 11 | 3 | jfull | 0.9862 | 1.2723 |
| prose | 11 | 3 | jfull | 0.9820 | 1.2635 |
| prose | 27 | 0 | kpca | 0.9910 | 1.1358 |
| prose | 27 | 0 | kpca | 0.9867 | 1.1988 |
| prose | 27 | 0 | kpca | 0.9901 | 1.2728 |
| prose | 27 | 0 | kpca | 0.9880 | 1.1420 |
| prose | 27 | 0 | jfull | 0.9877 | 1.2607 |
| prose | 27 | 0 | jfull | 0.9898 | 1.5446 |
| prose | 27 | 0 | jfull | 0.9851 | 1.5419 |
| prose | 27 | 0 | jfull | 0.9901 | 1.3398 |
| prose | 27 | 1 | kpca | 0.9811 | 1.3488 |
| prose | 27 | 1 | kpca | 0.9791 | 1.2589 |
| prose | 27 | 1 | kpca | 0.9779 | 1.2968 |
| prose | 27 | 1 | kpca | 0.9972 | 1.1939 |
| prose | 27 | 1 | jfull | 0.9869 | 1.1544 |
| prose | 27 | 1 | jfull | 0.9878 | 1.1388 |
| prose | 27 | 1 | jfull | 0.9898 | 1.1329 |
| prose | 27 | 1 | jfull | 0.9764 | 1.2373 |
| prose | 27 | 2 | kpca | 0.9869 | 1.5323 |
| prose | 27 | 2 | kpca | 0.9981 | 1.4720 |
| prose | 27 | 2 | kpca | 0.9892 | 1.4613 |
| prose | 27 | 2 | kpca | 0.9887 | 1.4779 |
| prose | 27 | 2 | jfull | 0.9877 | 1.2989 |
| prose | 27 | 2 | jfull | 0.9832 | 1.3847 |
| prose | 27 | 2 | jfull | 0.9886 | 1.3260 |
| prose | 27 | 2 | jfull | 0.9885 | 1.2905 |
| prose | 27 | 3 | kpca | 0.9665 | 1.5568 |
| prose | 27 | 3 | kpca | 0.9700 | 1.6769 |
| prose | 27 | 3 | kpca | 0.9650 | 1.3615 |
| prose | 27 | 3 | kpca | 0.9655 | 1.7886 |
| prose | 27 | 3 | jfull | 0.9842 | 1.2516 |
| prose | 27 | 3 | jfull | 0.9890 | 1.3412 |
| prose | 27 | 3 | jfull | 0.9728 | 1.3192 |
| prose | 27 | 3 | jfull | 0.9822 | 1.2378 |
| code | 11 | 0 | kpca | 0.9779 | 1.4017 |
| code | 11 | 0 | kpca | 0.9671 | 1.3745 |
| code | 11 | 0 | kpca | 0.9713 | 1.4415 |
| code | 11 | 0 | kpca | 0.9686 | 2.5556 |
| code | 11 | 0 | jfull | 0.9797 | 1.2689 |
| code | 11 | 0 | jfull | 0.9843 | 1.3476 |
| code | 11 | 0 | jfull | 0.9836 | 1.3551 |
| code | 11 | 0 | jfull | 0.9920 | 1.2965 |
| code | 11 | 1 | kpca | 0.9797 | 1.3296 |
| code | 11 | 1 | kpca | 0.9738 | 1.4171 |
| code | 11 | 1 | kpca | 0.9779 | 1.3416 |
| code | 11 | 1 | kpca | 0.9758 | 1.3264 |
| code | 11 | 1 | jfull | 0.9891 | 1.1524 |
| code | 11 | 1 | jfull | 0.9865 | 1.2569 |
| code | 11 | 1 | jfull | 0.9870 | 1.2889 |
| code | 11 | 1 | jfull | 0.9858 | 1.2719 |
| code | 11 | 2 | kpca | 0.8773 | 2.1877 |
| code | 11 | 2 | kpca | 0.9281 | 2.2095 |
| code | 11 | 2 | kpca | 0.9282 | 2.8971 |
| code | 11 | 2 | kpca | 0.8741 | 2.5320 |
| code | 11 | 2 | jfull | 0.9820 | 1.2210 |
| code | 11 | 2 | jfull | 0.9865 | 1.2370 |
| code | 11 | 2 | jfull | 0.9893 | 1.5332 |
| code | 11 | 2 | jfull | 0.9931 | 1.2743 |
| code | 11 | 3 | kpca | 0.9907 | 3.2837 |
| code | 11 | 3 | kpca | 0.9530 | 2.5134 |
| code | 11 | 3 | kpca | 0.9630 | 1.7107 |
| code | 11 | 3 | kpca | 0.9810 | 2.0540 |
| code | 11 | 3 | jfull | 0.9899 | 1.3066 |
| code | 11 | 3 | jfull | 0.9839 | 1.4325 |
| code | 11 | 3 | jfull | 0.9858 | 1.3835 |
| code | 11 | 3 | jfull | 0.9826 | 1.3846 |
| code | 27 | 0 | kpca | 0.9864 | 1.2622 |
| code | 27 | 0 | kpca | 0.9921 | 1.5140 |
| code | 27 | 0 | kpca | 0.9899 | 1.3191 |
| code | 27 | 0 | kpca | 0.9885 | 1.2986 |
| code | 27 | 0 | jfull | 0.9838 | 1.2922 |
| code | 27 | 0 | jfull | 0.9906 | 1.4848 |
| code | 27 | 0 | jfull | 0.9818 | 1.3586 |
| code | 27 | 0 | jfull | 0.9914 | 1.3623 |
| code | 27 | 1 | kpca | 0.9944 | 1.2981 |
| code | 27 | 1 | kpca | 0.9940 | 1.3197 |
| code | 27 | 1 | kpca | 0.9953 | 1.2131 |
| code | 27 | 1 | kpca | 0.9986 | 1.2131 |
| code | 27 | 1 | jfull | 0.9802 | 1.2205 |
| code | 27 | 1 | jfull | 0.9771 | 1.1889 |
| code | 27 | 1 | jfull | 0.9779 | 1.1869 |
| code | 27 | 1 | jfull | 0.9669 | 1.3190 |
| code | 27 | 2 | kpca | 0.9919 | 1.2770 |
| code | 27 | 2 | kpca | 0.9926 | 1.3281 |
| code | 27 | 2 | kpca | 0.9914 | 1.2800 |
| code | 27 | 2 | kpca | 0.9929 | 1.2116 |
| code | 27 | 2 | jfull | 0.9846 | 1.3149 |
| code | 27 | 2 | jfull | 0.9785 | 1.2833 |
| code | 27 | 2 | jfull | 0.9803 | 1.3028 |
| code | 27 | 2 | jfull | 0.9778 | 1.3238 |
| code | 27 | 3 | kpca | 0.9837 | 1.3189 |
| code | 27 | 3 | kpca | 0.9927 | 1.4647 |
| code | 27 | 3 | kpca | 0.9733 | 1.6295 |
| code | 27 | 3 | kpca | 0.9803 | 1.3603 |
| code | 27 | 3 | jfull | 0.9829 | 1.2158 |
| code | 27 | 3 | jfull | 0.9940 | 1.5069 |
| code | 27 | 3 | jfull | 0.9793 | 1.3089 |
| code | 27 | 3 | jfull | 0.9814 | 1.2261 |

## Error vs n_eff = 1 / sum p^2 (quartiles; keys, 384-bit arms; first query head of each group)
| arm | n_eff quartile | median n_eff | attn err (rel. to |o|) | |do| / rms|v| |
|---|---|---|---|---|
| full r192 2b | 1 | 10.9175 | 0.1053 | 0.0780 |
| full r192 2b | 2 | 21.1194 | 0.1095 | 0.0742 |
| full r192 2b | 3 | 33.6631 | 0.1083 | 0.0697 |
| full r192 2b | 4 | 58.6063 | 0.1020 | 0.0619 |
| jfull r96 4b | 1 | 10.9175 | 0.0732 | 0.0566 |
| jfull r96 4b | 2 | 21.1194 | 0.0747 | 0.0515 |
| jfull r96 4b | 3 | 33.6631 | 0.0715 | 0.0461 |
| jfull r96 4b | 4 | 58.6063 | 0.0736 | 0.0443 |
| jfull r24 16b | 1 | 10.9175 | 0.1669 | 0.1300 |
| jfull r24 16b | 2 | 21.1194 | 0.1628 | 0.1131 |
| jfull r24 16b | 3 | 33.6631 | 0.1639 | 0.1071 |
| jfull r24 16b | 4 | 58.6063 | 0.1571 | 0.0955 |

runtime 815s
