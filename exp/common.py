import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from foveakv import kvdata

RES = os.path.join(ROOT, "results")
PLOTS = os.path.join(RES, "plots")
CORPORA = kvdata.corpora()
LAYERS = kvdata.meta(CORPORA[0])["attn_layers"]


def prefill_len(L):
    return L["T"] - L["E"]


def md_table(header, rows, fmt="{:.4f}"):
    s = "| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n"
    for r in rows:
        s += "| " + " | ".join(fmt.format(x) if isinstance(x, (float, np.floating)) else str(x) for x in r) + " |\n"
    return s


def write(name, text):
    p = os.path.join(RES, name)
    open(p, "w").write(text)
    print(text)
    return p

if os.environ.get("FOVEAKV_QUICK"):          # smoke-test mode: one corpus, one layer
    CORPORA, LAYERS = ["prose"], [7, 11]
