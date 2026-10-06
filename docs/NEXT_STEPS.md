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
- Offline study on Qwen3.5-2B, Qwen3.5-4B and dense Qwen3-0.6B (head_dim 128), summarized in `results/MODELS.md`.
  Key smoothing is needed on Qwen3 (E19 predicts when); the int8 reference is lossy there, so it uses fp16.
- Kernels support head_dim 128 (`FOVEAKV_HEAD_DIM=128`); all 21 variants pass on Qwen3-0.6B data.

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

Offline study for any Qwen3 / Qwen3.5 checkpoint (CPU, memory-guarded):

```bash
export FOVEAKV_MODEL=Qwen/Qwen3-0.6B FOVEAKV_CORPUS_DIR=data/qwen3-0.6b/corpus FOVEAKV_KV_DIR=data/qwen3-0.6b/kv
python -m foveakv.text                                         # tokenize the texts for this model
python tools/capture_kv_stream.py --device cpu --fp16_ref all  # one layer at a time; --layers to subset
FOVEAKV_RESULTS=$PWD/results/e19 FOVEAKV_KV_DIRS="qwen3=$FOVEAKV_KV_DIR@fp16" python exp/e19_smoothing.py   # pipeline check first
FOVEAKV_REF=fp16 FOVEAKV_LAYERS=0,9,18,27 tools/run_offline.sh qwen3-0.6b      # results/qwen3-0.6b/
FOVEAKV_SMOOTH=1 ... tools/run_offline.sh qwen3-0.6b-smooth e13_selection_criteria e16_selection_gap e17_specache e18_static_mixed
```

- `FOVEAKV_MODEL` selects the checkpoint (default `Qwen/Qwen3.5-0.8B`), e.g. `FOVEAKV_MODEL=Qwen/Qwen3.5-2B-Base`.
- `FOVEAKV_KV_DIR` selects where captured caches go and are read from (default `data/kv`). Use one folder per model.
- Experiments write to `results/` and overwrite the 0.8B files. Run each model on its own branch, or copy `results/`
  to `results/<model>/` before switching.

What the code supports today:

| Model | Works as is? | Notes |
|---|---|---|
| Qwen3.5-0.8B / 2B / 9B | yes | same attention shape: head_dim 256, 4 query heads per KV head, RoPE on 64 dims. 9B needs about 18 GB in bf16. |
| Dense models (Qwen3, Llama) | offline yes, end to end no | capture with `tools/capture_kv_stream.py`; kernels support head_dim 128 and 256 but only a GQA group of 4, have no key smoothing, and `gpu/eval_e2e.py` drives Qwen3.5 modules directly. |

## Remaining experiments, in priority order

1. **End to end on bigger models.** 2B, 4B (offline done) and 9B: perplexity, retrieval, latency and memory with `gpu/eval_e2e.py`.
2. **A dense model end to end.** Add key smoothing to the GPU path (scale q, divide keys at encode), GQA groups other than 4,
   and an adapter for the model's attention module in `gpu/eval_e2e.py`. Offline results on Qwen3-0.6B are in `results/MODELS.md`.
   For every new model, run the E19 pipeline check right after capture (int8 broke on Qwen3).
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
