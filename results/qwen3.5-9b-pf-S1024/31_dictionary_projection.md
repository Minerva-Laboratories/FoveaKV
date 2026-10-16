# E31 Local bases without per-chunk storage: keys r = 64 / 192 (query-whitened), values r = 128 / 256

Relative attention-output error (pre-gate), mean over ['prose', 'code'] x layers [11, 27] x heads [0, 1, 2, 3]; evaluation as E29/E30.
"extra B/token" = storage beyond the r codes per token and head (per-chunk bases, indices, or the dictionary amortized
over this 8192-token sequence; the dictionary is a constant, so its share falls with context). Codes: 128 B (keys),
256 B (values) per token and head in bf16.

|  | chunk | arm | attn err | extra B/token |
|---|---|---|---|---|
| k | - | static (other corpus) | 0.1933 | 0.0000 |
| k | - | global (initial basis) | 0.0854 | 0.0000 |
| k | 256 | local (basis per chunk) | 0.0325 | 96.0000 |
| k | 256 | reencode (current basis only) | 0.1443 | 0.0000 |
| k | 256 | coordsel (initial full basis) | 0.0841 | 0.2500 |
| k | 256 | swap4 (initial basis + swap log) | 0.0777 | 6.0000 |
| k | 256 | swap8 (initial basis + swap log) | 0.0723 | 12.0000 |
| k | 256 | swap16 (initial basis + swap log) | 0.0637 | 24.0000 |
| k | 256 | dict2d (current dictionary) | 0.0811 | 17.6133 |
| k | 256 | dict4d (current dictionary) | 0.0748 | 19.8926 |
| k | 1024 | local (basis per chunk) | 0.0572 | 24.0000 |
| k | 1024 | reencode (current basis only) | 0.0835 | 0.0000 |
| k | 1024 | coordsel (initial full basis) | 0.0845 | 0.0625 |
| k | 1024 | swap4 (initial basis + swap log) | 0.0812 | 1.5000 |
| k | 1024 | swap8 (initial basis + swap log) | 0.0781 | 3.0000 |
| k | 1024 | swap16 (initial basis + swap log) | 0.0748 | 6.0000 |
| k | 1024 | dict2d (current dictionary) | 0.0805 | 10.6318 |
| k | 1024 | dict4d (current dictionary) | 0.0805 | 10.6318 |
| v | - | static (other corpus) | 0.2109 | 0.0000 |
| v | - | global (initial basis) | 0.0517 | 0.0000 |
| v | 256 | local (basis per chunk) | 0.0139 | 256.0000 |
| v | 256 | reencode (current basis only) | 0.1969 | 0.0000 |
| v | 256 | coordsel (initial full basis) | 0.0503 | 0.5000 |
| v | 256 | swap4 (initial basis + swap log) | 0.0501 | 8.0000 |
| v | 256 | swap8 (initial basis + swap log) | 0.0493 | 16.0000 |
| v | 256 | swap16 (initial basis + swap log) | 0.0457 | 32.0000 |
| v | 256 | dict2d (current dictionary) | 0.0490 | 32.0000 |
| v | 256 | dict4d (current dictionary) | 0.0442 | 44.2148 |
| v | 1024 | local (basis per chunk) | 0.0334 | 64.0000 |
| v | 1024 | reencode (current basis only) | 0.1069 | 0.0000 |
| v | 1024 | coordsel (initial full basis) | 0.0511 | 0.1250 |
| v | 1024 | swap4 (initial basis + swap log) | 0.0490 | 2.0000 |
| v | 1024 | swap8 (initial basis + swap log) | 0.0481 | 4.0000 |
| v | 1024 | swap16 (initial basis + swap log) | 0.0466 | 8.0000 |
| v | 1024 | dict2d (current dictionary) | 0.0469 | 22.6602 |
| v | 1024 | dict4d (current dictionary) | 0.0469 | 22.6602 |

runtime 992s
