# FoveaKV

FoveaKV is a KV-cache format for long-context LLM decoding on small GPUs. Every token is kept as a 2-bit code. At each
decoding step, the few tokens the current query attends to the most also get their 2-bit residual, which turns them into
4-bit tokens. The 2-bit code does three jobs at once: it is a usable approximation for the many tokens that get little
attention, it is the signal that picks the important tokens, and it is the first half of their 4-bit code. An exact
correction merges both precisions inside one softmax. On Qwen3.5-0.8B at 16K–32K context, FoveaKV stays within 0.03% of
bf16 perplexity (4-bit TurboQuant: 0.46%). With the residuals kept in CPU memory and read over PCIe, the cache uses 213 MiB of
GPU memory at 128K context instead of 1536 MiB for bf16, and a full decoding step at 64K is faster than with a bf16 cache
(16.9 ms vs 20.4 ms).

The name comes from the fovea: the eye sees everything at low resolution and only the point it looks at in detail.

FoveaKV builds on two published ideas. Keeping a low-bit copy of every token and refining the tokens selected from it comes
from SpeCache (Jie et al., ICML 2025), which fetches full 16-bit copies of the selected tokens from CPU memory. FoveaKV
instead fetches only a 2-bit residual of the same code, so it needs no full-precision copy and moves 8 times less data per
refined token. The error model below comes from Laus et al. (2026). See [Related work](#related-work).

![Attention error versus bits read per step](paper/figs/frontier.png)

## Results

Qwen3.5-0.8B on an RTX 4050 Laptop GPU, with the compressed cache used at every decoding step.

- **Perplexity** is the mean change vs bf16 over prose and code at 16K and 32K context (256 decoded tokens, 3 documents each).
- **Retrieval** is a RULER-style needle test at 8K-32K where the question itself is decoded through the compressed cache.
  "Same answer" counts generations identical to the bf16 generation.
- **GPU memory** is the measured size of the whole KV cache (6 layers) at 128K context; "host" is pinned CPU memory.
- **Step** is the time of a full decoding step (all 24 layers, one token). **Attention** is one attention layer at 128K.
  Rows marked "in CPU memory" read their residuals (or exact copies) over PCIe during the step. This laptop's PCIe link
  gives 6.7 GB/s for copies and 3.4–5.3 GB/s for the scattered 64-byte rows FoveaKV reads.

| Method | Bits read / value | Perplexity vs bf16 | Retrieval | Same answer | GPU memory @128K | Step @32K | Step @64K | Attention @128K |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 (reference) | 16 | 0 | 93.8% | 100% | 1536 MiB | 14.9 ms | 20.4 ms | 3774 µs |
| **FoveaKV, 5% hot, residuals in CPU memory** | **2.42** | **+0.03%** | **94.4%** | **93%** | 213 MiB + 192 host | 16.2 ms | 16.9 ms | 1164 µs |
| FoveaKV, 5% hot, all in GPU memory | 2.42 | +0.03% | 94.4% | 93% | 405 MiB | 16.3 ms | 16.2 ms | 1024 µs |
| FoveaKV, 5% hot, two-stage, residuals in CPU memory | 2.47 | +0.01% | 94.4% | 94% | 213 MiB + 192 host | 18.9 ms | 21.1 ms | 2310 µs |
| FoveaKV, 2% hot | 2.36 | +0.12% | 93.8% | 94% | 405 MiB | 16.2 ms | 16.2 ms | 934 µs |
| FoveaKV, 10% hot | 2.52 | +0.09% | 94.4% | 93% | 405 MiB | 16.2 ms | 16.3 ms | 1178 µs |
| TurboQuant-MSE, 4-bit | 4.20 | +0.46% | 94.8% | 88% | 396 MiB | 15.6 ms | 15.3 ms | 745 µs |
| TurboQuant-MSE, 3-bit | 3.21 | +1.33% | 93.8% | 79% | 396 MiB | 15.3 ms | 15.3 ms | 750 µs |
| TurboQuant-MSE, 2-bit | 2.23 | +13.0% | 90.6% | 72% | 204 MiB | 15.3 ms | 15.2 ms | 722 µs |
| KIVI, 4-bit | 4.39 | +0.26% | 94.4% | 88% | 411 MiB | 14.5 ms | 14.4 ms | 603 µs |
| KIVI, 2-bit | 2.41 | +29.0% | 77.4% | 60% | 219 MiB | 14.4 ms | 14.4 ms | 510 µs |
| Quest, 5% of pages | 1.99 | +3.0% | 90.6% | 83% | 1635 MiB | 15.1 ms | 15.2 ms | 901 µs |
| Quest, 10% of pages | 2.78 | +1.8% | 93.8% | 88% | 1635 MiB | 15.0 ms | 15.3 ms | 991 µs |
| SpeCache-style, 5% exact tokens, copies in CPU memory | 3.21 | +0.07% | 94.1% | 96% | 219 MiB + 1536 host | 18.2 ms | 22.3 ms | 5006 µs |
| SpeCache-style, same traffic as FoveaKV 5%, in CPU memory | 2.51 | +2.07% | 93.8% | 86% | 219 MiB + 1536 host | 17.6 ms | 17.6 ms | 1335 µs |

How to read it:

- Bits include gains, scales, block means and the exact recent window at this context length. Every compressed method gets
  the same exact 128-token recent window.
- Quest keeps the full bf16 cache in GPU memory. SpeCache-style keeps a 16-bit copy of every token in CPU memory; its bits
  read include the 16-bit refined tokens.
- Retrieval uses 8 samples per cell, so small accuracy differences are noise.
- Keeping residuals in CPU memory changes where data lives, not the result: the offloaded and in-GPU versions give the same
  output (checked in `gpu/test_kernels.py`), so they share their quality numbers.
- Up to 32K the decoding step is dominated by the rest of the model and by kernel launch overhead in our PyTorch loop, so
  cache compression does not make it faster; FoveaKV adds about 1 ms for selection and the correction pass. At 64K the bf16
  cache starts to dominate (20.4 ms) and FoveaKV with residuals in CPU memory is faster (16.9 ms). Reading 16-bit copies over
  PCIe instead (SpeCache-style, 5%) takes 22.3 ms.
- Our gain-shape kernels are compute-bound on this GPU, so reading 2 bits does not make FoveaKV faster than 4-bit kernels.
  Its advantage today is memory: about half the resident size of a 4-bit cache at the same quality. Fusing the selection
  step and using CUDA graphs would remove most of the extra milliseconds; we have not done this yet.

Offline findings on the captured cache (attention-output error, 5 texts x 6 layers, see `results/`):

- At the same refinement traffic, FoveaKV has 3 times lower error than SpeCache-style refinement (0.0175 vs 0.0537 at 5%).
- Choosing the 4-bit tokens again at every step matters: a fixed set chosen once from earlier queries, at the same number
  of 4-bit tokens, has 2 times higher error (0.0342 vs 0.0174).
- Selecting from 2-bit keys loses a little to selecting with the true attention (0.0175 vs 0.0169). Two-stage selection
  (2k candidates from 2-bit keys, re-ranked with their 4-bit keys) closes the gap (0.0168).

## How it works

For each token and KV head:

1. **Center.** Subtract the per-channel mean of its 128-token block (stored in fp16). The mean adds one known offset per
   block to the logits, so only the residual is quantized.
2. **Rotate.** Apply a fixed randomized Hadamard transform. Rotations keep dot products unchanged and make every
   coordinate close to a standard Gaussian, so one analytic codebook fits all heads.
3. **Split into gain and shape.** Store the length of the vector (8 bits, log scale) and quantize its direction per
   coordinate with a 16-level Gaussian Lloyd–Max codebook.
4. **Nest the code.** The 16 levels are grouped in 4 groups of 4. The group index is the 2-bit coarse code; the position
   inside the group is the 2-bit residual. Coarse codes are always on the GPU; residuals can live elsewhere (for example
   in CPU memory).
5. **Correct the shrinkage.** Centroid codebooks shrink large values by about (1 − D), where D is the distortion. This
   bias is known (TurboQuant removes it with an extra 1-bit code; llama.cpp rescales key norms). For attention it acts like
   a higher softmax temperature applied to all tokens at once. FoveaKV rescales key shapes by 1/√(1 − D) and value residuals
   by 1/(1 − D). The scales are folded into the codebook tables, so they cost nothing.

At each decoding step:

1. One pass over all coarse codes computes the attention logits and the softmax state (max, sum, weighted values).
2. The top-k tokens by estimated attention are selected (shared by the query heads of a GQA group).
3. A second pass reads only their residuals and swaps their coarse contribution for the fine one. This "delta
   correction" is exact: the result equals attention with fine values on the hot tokens and coarse values elsewhere.
4. The most recent 128 tokens are always exact.

Storage is 2 + 8/256 + 16/128 = 2.16 bits per value. Each step reads 2·k more bits per value for the hot tokens.

## Why it works

To first order, the attention-output error is a weighted mean squared error of the cache (this is Theorem 1 of Laus et al.,
2026, "KV Cache Compression Through the Lens of Transform Coding"):

```
E‖ô − o‖² ≈ Σᵢ pᵢ² · Dᵢ · ( ‖q‖² gₖ,ᵢ² ‖vᵢ − o‖² / d² + gᵥ,ᵢ² )
```

where pᵢ is the attention weight of token i, gₖ,ᵢ and gᵥ,ᵢ are the key and value gains, and Dᵢ is the codebook
distortion of that token (0 for exact, 0.0095 for 4-bit, 0.12 for 2-bit). Three things follow:

- Errors count with pᵢ², so tokens with little attention barely matter. A 2-bit code is enough for them.
- Giving 4 bits to the tokens with the largest pᵢ² removes most of the error. Attention is concentrated: in our data the
  top 5% of tokens hold about 92% of the weight.
- Every term can be computed with an unmodified FlashAttention call (our addition), using
  `Σ pᵢ² xᵢ = exp(lse(2s) − 2·lse(s)) · softmax(2s)·x` (attention at doubled scale plus the returned log-sum-exp).

The model assumes errors with zero mean. Centroid codebooks break that assumption (the shrinkage above), which is why the
correction matters. Selection from 2-bit keys also loses a little for a reason the model does not show: the coding error is
uncorrelated with the estimate but anti-correlated with the true score, so no re-scoring of the estimate can fix it, while a
few more bits of the selected candidates can.

The paper has the derivations, proofs and all measurements: [`paper/main.pdf`](paper/main.pdf).

## Quick start

```bash
pip install -r requirements.txt          # CPU part: NumPy, SciPy, safetensors, tokenizers
python -m pytest -q                      # codebooks, nesting, delta correction, rotation, loader

# GPU part (CUDA toolkit with nvcc needed; kernels compile on first use)
pip install -r requirements-gpu.txt
python gpu/test_kernels.py               # every kernel vs an fp64 reference (needs the captured cache below)
python gpu/bench_kernels.py              # per-layer attention latency of all methods, 8K-128K
python gpu/bench_memory.py               # measured GPU and CPU memory of every cache, 32K-128K
python gpu/bench_e2e.py                  # full decoding-step latency of all methods, 16K and 32K
BENCH_LENS=65536 BENCH_TAG=_64k python gpu/bench_e2e.py   # same at 64K
python gpu/eval_e2e.py check             # our decode loop vs HF transformers
python gpu/eval_e2e.py ppl               # long-context perplexity, all methods
python gpu/eval_e2e.py retrieval         # RULER-style retrieval, all methods
```

The model (`Qwen/Qwen3.5-0.8B`) and WikiText-2 are read from the local Hugging Face cache. Download them first with
`huggingface-cli download Qwen/Qwen3.5-0.8B` and `huggingface-cli download --repo-type dataset Salesforce/wikitext`.

## Reproduce the paper

```bash
python -m foveakv.text                   # build the 5 evaluation texts (data/corpus)
python tools/capture_kv.py               # one GPU forward pass per text: int8 KV cache + queries (data/kv, ~0.5 GB)
cd exp && python e01_gaussianization.py  # offline experiments, each writes results/NN_*.md (E1-E18)
cd ../paper && python make_tables.py && latexmk -pdf main.tex
```

Offline experiments run on the CPU from the captured cache: each decoding query only sees earlier tokens, and all
statistics come from the prefill part. Set `FOVEAKV_QUICK=1` to run any experiment on one text and two layers. Every
number in the paper is generated by `paper/make_tables.py` from the files in `results/`.

## Repository layout

```
foveakv/        NumPy reference: codebooks, gain-shape coding, rotation, attention, selection, Qwen3.5 forward pass
gpu/            CUDA kernels (kernels.cu), GPU cache classes, kernel tests, benchmarks, end-to-end evaluation
exp/            offline experiments E1-E18 (one script each)
tools/          KV capture, HF reference dump, log recovery
tests/          unit tests
results/        experiment reports (*.md), raw results (*.json, *.npz), plots
paper/          LaTeX source, generated tables and figures, citation ledger, main.pdf
reference/      the original synthetic tests that motivated the project
```

## Related work

The closest work, and how FoveaKV differs:

| Work | What it does | Difference |
|---|---|---|
| SpeCache (ICML 2025) | low-bit KIVI copy of all tokens; top-k replaced by 16-bit copies from CPU | FoveaKV refines with a 2-bit residual of a nested code: no 16-bit copy, 8x less transfer per token |
| Laus et al. (2026) | attention-aware distortion with p² weights, used for offline bit allocation | FoveaKV uses it online to pick tokens, and computes it with FlashAttention |
| TurboQuant (ICLR 2026) | rotation + per-coordinate Lloyd-Max + norm, QJL to remove inner-product bias | FoveaKV adds nesting, block means and a free rescaling instead of QJL |
| ZipCache, MiKV, HyQuant | more bits for important tokens, chosen once | FoveaKV re-chooses at every step from the current query |
| QuantSpec, Kitty, Lynx | bit-plane KV codes on uniform grids, all tokens refined | FoveaKV refines only the tokens the query looks at |
| Quest, Twilight, Self-Indexing Attention, RaBitQCache | select tokens from compressed keys, drop the rest | FoveaKV keeps every token at 2 bits |

The full list (90 references), with what each one supports and where the paper cites it, is in
[`paper/CITATIONS.md`](paper/CITATIONS.md).

## Limitations

- One small model (Qwen3.5-0.8B: 6 attention layers, 2 KV heads) and contexts up to 32K tokens.
- The SpeCache and static mixed-precision comparisons are offline emulations, not the original systems.
- Perplexity windows are short (256 tokens per document); differences below about 0.2% are within noise.
- Offloading is measured on one laptop (PCIe at 6.7 GB/s) up to 64K in the full model and 128K per layer. Residual reads
  are not yet overlapped with the work of other layers.
- The hot set changes quickly between steps (about 56% overlap), so prefetching is harder than it looks. Missed tokens
  carry little attention, so serving them coarse for one step costs little.
- The CUDA kernels are compute-bound on a laptop GPU, and selection adds kernel launches; a fused, faster kernel is the main
  open engineering task.

## Citation

```bibtex
@misc{foveakv2026,
  title  = {FoveaKV: Foveated KV Caches with Nested Codes and Query-Dependent Refinement},
  author = {TBD},
  year   = {2026},
  note   = {https://github.com/TBD/foveakv}
}
```
