"""Standard long-context benchmarks through the compressed cache: LongBench (Bai et al., ACL 2024) and RULER (Hsieh et al.,
COLM 2024), same protocol as eval_e2e.retrieval: the context is prefilled exactly once and shared by all schemes; each
scheme then builds its cache from that snapshot and DECODES the rest of the prompt (question and answer cue) token by token
through the compressed cache, then generates greedily. So every scheme answers from its own compressed context.

Data (outside the repo, BENCH_DIR, default /mnt/ext/bench-data):
  longbench/data/<task>.jsonl, longbench/dataset2prompt.json, dataset2maxlen.json, metrics.py  (THUDM/LongBench)
  ruler/gen/<len>/<task>/validation.jsonl  (generated with NVIDIA/RULER scripts/data/prepare.py, tokenizer of the evaluated model)
Results: one JSON line per (sample, scheme) in results/<FOVEAKV_RESULTS_TAG>/bench_<suite>.jsonl; reruns skip done lines.

Usage: <python> gpu/eval_bench.py longbench --tasks hotpotqa,qasper --n 20 --schemes bf16,dyn5,kivi2
       <python> gpu/eval_bench.py ruler --lens 8192,16384 --tasks niah_single_1 --n 20
       <python> gpu/eval_bench.py score longbench|ruler"""
import argparse, json, os, random, re, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BENCH = os.environ.get("BENCH_DIR", "/mnt/ext/bench-data")
OUT = os.path.join(ROOT, "results", os.environ.get("FOVEAKV_RESULTS_TAG", "e2e"))
# LongBench: tasks scored without the chat template (few-shot / completion formats), as in LongBench's pred.py
NO_CHAT = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
STOP_NL = {"samsum"}                          # LongBench stops samsum at the first newline


def out_path(suite):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, f"bench_{suite}.jsonl")


def done_keys(path):
    if not os.path.exists(path):
        return set()
    return {(r["task"], r["L"], r["idx"], r["scheme"]) for r in map(json.loads, open(path))}


def chat(tok, text):
    return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def split_prompt(tok, full, cut, max_ctx):
    """Token ids of the prefix (exact prefill) and suffix (decoded through the cache). full[:cut] is the prefix; if the
    prompt is longer than max_ctx tokens, the middle of the prefix is cut (LongBench truncates the middle)."""
    pre = tok(full[:cut], add_special_tokens=False)["input_ids"]
    suf = tok(full[cut:], add_special_tokens=False)["input_ids"]
    room = max_ctx - len(suf)
    if len(pre) > room:
        h = room // 2
        pre = pre[:h] + pre[len(pre) - (room - h):]
    return pre, suf


@torch.no_grad()
def strip_think(t):
    """Drop a (possibly unclosed) <think> block the model emits even without the chat template."""
    return re.sub(r"<think>.*?(</think>|$)", "", t, flags=re.S)


def generate(R, scheme, snap, suf_ids, max_new, stop_nl=False):
    """Decode the prompt suffix through the scheme's cache, then greedy generation. Returns (text, bits, n_new)."""
    from gpu.eval_e2e import DEV
    caches = R.build(scheme, snap, extra=len(suf_ids) + max_new + 8)
    lin = snap["lin"].clone()
    pos = snap["T"]
    h = R.forward(torch.tensor(suf_ids, device=DEV), pos, caches, lin, prefill=False)
    pos += len(suf_ids)
    stops = {R.tok.eos_token_id, *[R.tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]}
    out = []
    nxt = int(R.logits(h[-1:])[0].argmax())
    while nxt not in stops and len(out) < max_new:
        out.append(nxt)
        if stop_nl and "\n" in R.tok.decode([nxt]) and strip_think(R.tok.decode(out)).strip():
            break                               # stop at the first newline after real text (not an empty think block)
        h = R.forward(torch.tensor([nxt], device=DEV), pos, caches, lin, prefill=False)
        pos += 1
        nxt = int(R.logits(h[-1:])[0].argmax())
    bits = float(np.mean([c.read_bits() for c in caches.values()]))
    del caches
    return R.tok.decode(out, skip_special_tokens=True), bits, len(out)


def run(R, suite, items, schemes, chunk):
    """items: list of dicts with task, L, idx, pre (ids), suf (ids), max_new, answers, extra."""
    from gpu.eval_e2e import DEV
    path = out_path(suite)
    done = done_keys(path)
    t0 = time.time()
    for n, it in enumerate(items):
        todo = [s for s in schemes if (it["task"], it["L"], it["idx"], s) not in done]
        if not todo:
            continue
        tp = time.time()
        snap = R.prefill(torch.tensor(it["pre"], device=DEV), chunk=chunk)
        tp = time.time() - tp
        for sc in todo:
            ts = time.time()
            txt, bits, nn = generate(R, sc, snap, it["suf"], it["max_new"], it.get("stop_nl", False))
            rec = dict(task=it["task"], L=it["L"], idx=it["idx"], scheme=sc, pred=txt, answers=it["answers"],
                       n_ctx=len(it["pre"]) + len(it["suf"]), bits=bits, n_new=nn, t_prefill=tp, t_scheme=time.time() - ts,
                       **it.get("extra", {}))
            with open(path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"[{time.time()-t0:6.0f}s] {n+1}/{len(items)} {it['task']} L={it['L']} #{it['idx']} {sc:20s} "
                  f"ctx {rec['n_ctx']} prefill {tp:.0f}s gen {rec['t_scheme']:.0f}s ({nn} tok) {txt.strip()[:50]!r}", flush=True)
        del snap
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------------------------------- LongBench
def longbench_items(tok, tasks, n, max_ctx, seed=0):
    d = os.path.join(BENCH, "longbench")
    prompts = json.load(open(os.path.join(d, "dataset2prompt.json")))
    maxlen = json.load(open(os.path.join(d, "dataset2maxlen.json")))
    items = []
    for task in tasks:
        rows = [json.loads(l) for l in open(os.path.join(d, "data", f"{task}.jsonl"))]
        idx = sorted(random.Random(seed).sample(range(len(rows)), min(n, len(rows))))
        for i in idx:
            r = rows[i]
            body = prompts[task].format(**r)
            marker = r["context"]
            full = body if task in NO_CHAT else chat(tok, body)
            cut = full.find(marker) + len(marker)       # prefix ends with the context
            pre, suf = split_prompt(tok, full, cut, max_ctx)
            items.append(dict(task=task, L=0, idx=i, pre=pre, suf=suf, max_new=maxlen[task], answers=r["answers"],
                              stop_nl=task in STOP_NL, extra=dict(all_classes=r.get("all_classes"), length=r["length"])))
    return items


def score_longbench(path):
    sys.path.insert(0, os.path.join(BENCH, "longbench"))
    import metrics as Mx
    m = {"narrativeqa": Mx.qa_f1_score, "qasper": Mx.qa_f1_score, "multifieldqa_en": Mx.qa_f1_score,
         "hotpotqa": Mx.qa_f1_score, "2wikimqa": Mx.qa_f1_score, "musique": Mx.qa_f1_score, "gov_report": Mx.rouge_score,
         "qmsum": Mx.rouge_score, "multi_news": Mx.rouge_score, "trec": Mx.classification_score,
         "triviaqa": Mx.qa_f1_score, "samsum": Mx.rouge_score, "passage_count": Mx.count_score,
         "passage_retrieval_en": Mx.retrieval_score, "lcc": Mx.code_sim_score, "repobench-p": Mx.code_sim_score}
    out = []
    for r in map(json.loads, open(path)):
        pred = strip_think(r["pred"])
        if r["task"] in ("trec", "triviaqa", "samsum", "lsht"):
            pred = pred.lstrip("\n").split("\n")[0]
        r["score"] = 100 * max(m[r["task"]](pred, g, all_classes=r.get("all_classes")) for g in r["answers"])
        out.append(r)
    return out


# ----------------------------------------------------------------------------------------------- RULER
# answer lengths: RULER's tokens_to_generate (generation stops at end of turn; 32 was too short for UUID answers)
RULER_GEN = {"vt": 30, "cwe": 120, "fwe": 50, "qa_1": 32, "qa_2": 32}
def ruler_items(tok, tasks, lens, n):
    items = []
    for L in lens:
        for task in tasks:
            p = os.path.join(BENCH, "ruler", "gen", str(L), task, "validation.jsonl")
            if not os.path.exists(p):
                print(f"missing {p}, skipped (rerun later; done lines are kept)", flush=True)
                continue
            rows = [json.loads(l) for l in open(p)][:n]
            for r in rows:
                body = r["input"]
                full = chat(tok, body) + r.get("answer_prefix", "")
                cut = full.rfind("\n", 0, full.rfind(body[-20:]))       # the question is the last line of the input
                pre, suf = split_prompt(tok, full, cut, 10 ** 9)
                items.append(dict(task=task, L=L, idx=r["index"], pre=pre, suf=suf, max_new=RULER_GEN.get(task, 128),
                                  answers=r["outputs"]))
    return items


def score_ruler(path):
    out = []
    for r in map(json.loads, open(path)):
        p = r["pred"].lower()
        hits = [a.lower() in p for a in r["answers"]]
        r["score"] = 100 * (max(hits) if r["task"].startswith("qa") else sum(hits) / len(hits))
        out.append(r)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("suite", choices=["longbench", "ruler", "plan", "score"])
    ap.add_argument("which", nargs="?", default="")
    ap.add_argument("--tasks", default="")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--lens", default="8192,16384,32768")
    ap.add_argument("--max_ctx", type=int, default=32768)
    ap.add_argument("--schemes", default="bf16,dyn5,kivi2,kivi4,tq4,specache5-eqtraffic,specache5")
    ap.add_argument("--chunk", type=int, default=2048)
    a = ap.parse_args()
    if a.suite == "score":
        rs = (score_longbench if a.which == "longbench" else score_ruler)(out_path(a.which))
        import collections
        tab = collections.defaultdict(list)
        for r in rs:
            tab[(r["task"], r["L"], r["scheme"])].append(r["score"])
        for k in sorted(tab):
            print(*k, f"{np.mean(tab[k]):6.2f}", len(tab[k]))
        sys.exit()
    from gpu.eval_e2e import Runner
    from gpu.foveakv_gpu import ext
    ext()
    R = Runner()
    # plan: several stages in one process (the model loads once), e.g.
    #   plan '[{"suite": "ruler", "tasks": "niah_single_1,vt", "n": 10, "lens": "8192"}, {"suite": "longbench", ...}]'
    stages = json.loads(a.which) if a.suite == "plan" else [dict(suite=a.suite, tasks=a.tasks, n=a.n, lens=a.lens)]
    for st in stages:
        sch = st.get("schemes", a.schemes).split(",")
        if st["suite"] == "longbench":
            items = longbench_items(R.tok, st["tasks"].split(","), st["n"], st.get("max_ctx", a.max_ctx))
        else:
            items = ruler_items(R.tok, st["tasks"].split(","), [int(x) for x in str(st["lens"]).split(",")], st["n"])
        print(f"{st['suite']}: {len(items)} items, schemes {sch}", flush=True)
        run(R, st["suite"], items, sch, a.chunk)
