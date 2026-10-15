# Roadmap: bigger models

This is a plan for testing FoveaKV on models larger than the ones we used (Qwen3.5-0.8B end to end; 2B, 4B and Qwen3-0.6B
offline only). Setup, the per-model commands and the conventions are in [`NEXT_STEPS.md`](NEXT_STEPS.md). This file only
gives the order of work and what to check at each step.

## Before you start: does the model fit the code?

Read the model's `config.json` and compare it with this table.

| Field | Supported today | If not |
|---|---|---|
| `head_dim` | 128 or 256 | add a `FKV_D` case in `gpu/kernels.cu` (offline code is generic) |
| query heads per KV head | 4 in the GPU kernels; any value offline | step 3 below |
| attention module | Qwen3.5 (`self_attn` plus `linear_attn`) in `gpu/eval_e2e.py` | step 3 below |
| RoPE | partial or full, read from the config | NoPE or MLA layers need new code |

Qwen3.5 models with 4 query heads per KV head and head_dim 256 (like 0.8B, 2B, 4B and 9B) work end to end without changes.

## Step 1: offline study (CPU, any size)

`tools/capture_kv_stream.py` loads one decoder layer at a time from the safetensors files, so the model never has to fit in
RAM. Use one folder per model.

1. Capture with `--fp16_ref all` and a subset of layers (`--layers`, e.g. four spread over the depth).
2. Run the E19 pipeline check first. Continue only if offline exact attention matches the model's own output within about
   1.5%. A larger error means the reference is lossy, as int8 was on Qwen3.
3. Run `tools/run_offline.sh <tag>`. If E19 predicts a gain of 2× or more from key smoothing, also run the smoothed variant.
4. Add the model to `results/MODELS.md`.

What to watch for:
- the gap between FoveaKV at 5% and GS-4b (21–24% on Qwen3.5; about 2× on Qwen3-0.6B, still unexplained);
- whether SpeCache-style refinement with 8× the traffic beats FoveaKV (it did on 4B only);
- the GQA union size for groups larger than 4.

## Step 2: end to end on Qwen3.5 (GPU)

This step needs no code changes. Set `FOVEAKV_MODEL`, then in order:

1. `gpu/test_kernels.py` on the new model's data;
2. `gpu/eval_e2e.py check`, then `ppl` and `retrieval` (start with `--quick`);
3. `gpu/bench_kernels.py`, `gpu/bench_e2e.py` (contexts via `BENCH_CTX` / `BENCH_LENS`) and `gpu/bench_memory.py`.

Copy `results/` to `results/<model>/` before switching models, because the scripts overwrite the 0.8B files. Then add the
numbers to `paper/make_tables.py`.

## Step 3: dense and other models end to end

1. **GQA groups other than 4.** Make `G` a compile-time parameter in `gpu/kernels.cu`, like `FKV_D`. Groups of 2, 4 and 8
   cover most models.
2. **Key smoothing in the GPU path.** Compute γ from the prefill, store keys divided by γ, and multiply the query by γ once
   per step. Logits do not change.
3. **Model adapter.** `gpu/eval_e2e.py` calls Qwen3.5 modules directly. Add a path for dense attention (Qwen3, Llama) with
   no output gate.
4. Then repeat step 2.

## Step 4: longer context and standard benchmarks

- Quality at 64K–128K, where the memory saving matters most.
- Full RULER or LongBench instead of our RULER-style retrieval.
- More documents per perplexity cell and more retrieval samples per cell, with confidence intervals.

## Step 5: speed

Today FoveaKV is about 1 ms per step slower than 4-bit kernels up to 32K. Ways to close that gap:

- overlap the residual reads with other layers (CUDA streams);
- fuse selection into the coarse pass;
- capture the decoding step in a CUDA graph.

## Done means

For each new model:
- a row in `results/MODELS.md`;
- end-to-end perplexity and retrieval against bf16 and 4-bit;
- latency and memory at the longest context that fits;
- the paper tables regenerated.

Report negative results the same way as positive ones.
