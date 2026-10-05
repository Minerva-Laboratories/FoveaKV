"""Rebuild results/gpu_ppl.json and results/gpu_retrieval.json from the run logs (the JSONs of the full baseline run were
overwritten by a later tagged run due to a since-fixed bug in gpu/eval_e2e.py). PPL is printed with 4 decimals
(nll = log PPL, rel. precision ~1e-5); retrieval outputs are logged truncated to 60 characters."""
import ast, json, math, os, re
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
ppl = []
for line in open(os.path.join(R, "gpu_ppl.log")):
    m = re.match(r"\[\s*\d+s\] (\w+) L=(\d+) doc(\d+) (\S+)\s+PPL\s+([\d.]+) bits ([\d.]+)", line)
    if m:
        src, L, d, sc, p, b = m.groups()
        ppl.append(dict(src=src, L=int(L), doc=int(d), scheme=sc, nll=math.log(float(p)), bits=float(b), from_log=True))
ret = []
for line in open(os.path.join(R, "gpu_retrieval.log")):
    m = re.match(r"\[\s*\d+s\] L=(\d+) (\S+)\s+s(\d+) (\S+)\s+score ([\d.]+)\s+(.*)$", line.rstrip("\n"))
    if m:
        L, task, si, sc, score, out = m.groups()
        ret.append(dict(L=int(L), task=task, sample=int(si), scheme=sc, score=float(score), out=ast.literal_eval(out), from_log=True))
json.dump(ppl, open(os.path.join(R, "gpu_ppl.json"), "w"), indent=1)
json.dump(ret, open(os.path.join(R, "gpu_retrieval.json"), "w"), indent=1)
print(len(ppl), "ppl rows,", len(ret), "retrieval rows")
