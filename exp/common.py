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
if os.environ.get("FOVEAKV_LAYERS"):          # subset of the captured attention layers
    LAYERS = [int(x) for x in os.environ["FOVEAKV_LAYERS"].split(",")]
SMOOTH = bool(int(os.environ.get("FOVEAKV_SMOOTH", "0")))   # per-channel key smoothing in the gain-shape caches (E13, E16-E18)
RES = os.environ.get("FOVEAKV_RESULTS", RES)  # one results folder per model
PLOTS = os.path.join(RES, "plots")
os.makedirs(PLOTS, exist_ok=True)


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


def head_chunks(L, n=2):
    """Split a layer dict into pieces of at most n KV heads (with their query heads), to bound memory on models
    with many heads. Each piece has the same keys; 'h0' is the index of its first KV head."""
    Hkv = L["k"].shape[1]
    G = L["q"].shape[1] // Hkv
    if Hkv <= n:
        yield dict(L, h0=0)
        return
    for h0 in range(0, Hkv, n):
        kv, qh = slice(h0, h0 + n), slice(h0 * G, (h0 + n) * G)
        part = dict(L, h0=h0)
        for key in ("k", "k_pre", "v"):
            part[key] = L[key][:, kv]
        for key in ("q", "q_pre", "q_spec", "q2_pre", "q2_post"):
            if key in L:
                part[key] = L[key][:, qh] if L[key].ndim == 3 else L[key][qh]
        yield part
