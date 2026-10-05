# Next steps and handoff

State on 2026-10-04 (commit after `6333806`). Everything in the paper and README is generated from `results/` by
`paper/make_tables.py`, so new experiments only need to write their raw files and re-run that script.

## Where things stand

Done and measured on Qwen3.5-0.8B (RTX 4050 Laptop GPU, 6 GB, PCIe 6.7 GB/s):

- Offline study E1-E18 on captured caches (`results/NN_*.md`).
- CUDA kernels for every method, checked against fp64 references (`gpu/test_kernels.py`, 21 variants).
- End-to-end quality: perplexity at 16K/32K and RULER-style retrieval at 8K-32K (`gpu/eval_e2e.py`).
- Latency: per attention layer 8K-128K and full decoding step 16K/32K/64K, with and without offloading.
- Memory: measured GPU and host bytes of every cache (`gpu/bench_memory.py`).
- Real offloading: FoveaKV residual planes (and SpeCache-style copies) in pinned host memory, read over PCIe.
- Paper draft (`paper/main.pdf`) with a verified citation ledger (`paper/CITATIONS.md`, 90 references).

## Set up another machine

```bash
git clone git@github.com:Minerva-Laboratories/FoveaKV.git && cd FoveaKV
pip install -r requirements-gpu.txt              # torch with CUDA, transformers>=5; nvcc must match torch's CUDA major
huggingface-cli download Qwen/Qwen3.5-0.8B      # or the model you want (see below)
huggingface-cli download --repo-type dataset Salesforce/wikitext
python -m pytest -q                              # 11 tests
python -m foveakv.text                           # evaluation texts -> data/corpus
python tools/capture_kv.py                       # one forward pass per text -> data/kv (~0.5 GB for 0.8B)
python gpu/test_kernels.py                       # kernels compile for the local GPU on first use
```

`data/` is not in git; the two commands above rebuild it. Kernels are compiled for the GPU found at run time
(override with `TORCH_CUDA_ARCH_LIST`).

## Running on another model

- `FOVEAKV_MODEL` selects the checkpoint (default `Qwen/Qwen3.5-0.8B`), e.g. `FOVEAKV_MODEL=Qwen/Qwen3.5-2B-Base`.
- `FOVEAKV_KV_DIR` selects where captured caches go and are read from (default `data/kv`). Use one folder per model.
- Experiments write to `results/` and overwrite the 0.8B files. Run each model on its own branch, or copy `results/`
  to `results/<model>/` before switching.

What the code supports today:

| Model | Works as is? | Notes |
|---|---|---|
| Qwen3.5-0.8B / 2B / 9B | yes | same attention shape: head_dim 256, 4 query heads per KV head, RoPE on 64 dims. 9B needs about 18 GB in bf16. |
| Dense models (Qwen3, Llama) | no | kernels assume head_dim 256 and a GQA group of 4 (`D` and `G` in `gpu/kernels.cu`, `gpu/foveakv_gpu.py`), and `gpu/eval_e2e.py` drives Qwen3.5 modules directly. Needs a templated kernel and a small model adapter. |

## Remaining experiments, in priority order

1. **Bigger models (same family).** Qwen3.5-2B and 9B: capture, offline E1-E18, kernels, end-to-end quality, latency and
   memory. Commands are the same with `FOVEAKV_MODEL` and `FOVEAKV_KV_DIR` set.
2. **A dense model.** Many attention layers, where the KV cache dominates memory. Needs `D=128` support in the kernels (template
   `D` and `G`, keep `LD = D/32` dims per lane) and an adapter for the model's attention module in `gpu/eval_e2e.py`.
3. **Longer context.** Quality at 64K-128K and a standard benchmark (full RULER, LongBench). The prefill path already uses
   SDPA with a lower-right causal bias, so memory is not the limit; time is.
4. **Faster decoding step.** Overlap residual reads with other layers (CUDA streams), fuse selection into the coarse pass,
   capture the step in a CUDA graph. Today FoveaKV costs about 1 ms more per step than 4-bit kernels up to 32K.
5. **Statistics.** More documents per perplexity cell and more retrieval samples per cell; report confidence intervals.
6. **Original SpeCache.** Our SpeCache-style path is our own implementation of its attention and transfers (no speculative
   prefetching). Running their code would make that comparison stronger.

## Paper items still open

- Authors and the citation block (`paper/main.tex`, `README.md`).
- Three references marked "UNVERIFIED parts" in `paper/CITATIONS.md` (PolarQuant venue, Qwen3-Next blog title, Sabin & Gray
  full text) and quotes marked "(via fetch)" in `paper/citations/G1-G3.md`.
- A method figure (coarse plane, residual plane, selection, delta correction).
- After new experiments: add their numbers to `paper/make_tables.py`, then `python paper/citations/build_ledger.py` if
  citations changed, and `latexmk -pdf main.tex` in `paper/`.

## Repository conventions

- Commits carry no Claude trailer or co-author line.
- README: plain, short sentences; gist in the first paragraph; every number traceable to `results/`.
- Run heavy jobs under a memory cap on small machines (we used `systemd-run --user --scope -p MemoryMax=4G`).
