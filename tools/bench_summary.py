"""Summary of a benchmark JSONL (gpu/eval_bench.py): mean score per (length, scheme), the paired difference to bf16
over the same samples with a 95% bootstrap interval (1,000 resamples), and per-task means.
Usage: FOVEAKV_RESULTS_TAG=<tag> python tools/bench_summary.py ruler|longbench > results/<tag>/<suite>_summary.md"""
import collections, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv += [""] * 2
from gpu import eval_bench as eb

suite = sys.argv[1]
rs = (eb.score_longbench if suite == "longbench" else eb.score_ruler)(eb.out_path(suite))
by = collections.defaultdict(dict)
for r in rs:
    by[(r["L"], r["scheme"])][(r["task"], r["idx"])] = r["score"]
schemes = list(dict.fromkeys(r["scheme"] for r in rs))
rng = np.random.default_rng(0)
print(f"# {suite} summary ({os.environ.get('FOVEAKV_RESULTS_TAG')})\n")
print("| length | scheme | n | mean | diff to bf16 | 95% CI |\n|---|---|---|---|---|---|")
for L in sorted({k[0] for k in by}):
    base = by.get((L, "bf16"), {})
    for sc in schemes:
        d = by.get((L, sc), {})
        keys = sorted(set(d) & set(base))
        if not keys:
            continue
        x = np.array([d[k] for k in keys]); b = np.array([base[k] for k in keys])
        diff = x - b
        boot = [diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(1000)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        print(f"| {L or 'all'} | {sc} | {len(keys)} | {x.mean():.1f} | {diff.mean():+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
print("\n## Per task (mean score)\n")
tasks = list(dict.fromkeys(r["task"] for r in rs))
print("| length | task | " + " | ".join(schemes) + " |\n|" + "---|" * (len(schemes) + 2))
for L in sorted({k[0] for k in by}):
    for t in tasks:
        vals = [[v for (tt, _), v in by.get((L, sc), {}).items() if tt == t] for sc in schemes]
        if not any(vals):
            continue
        print(f"| {L or 'all'} | {t} | " + " | ".join(f"{np.mean(v):.1f}" if v else "" for v in vals) + " |")
