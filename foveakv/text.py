"""Evaluation corpora (token ids cached under data/corpus/)."""
import os, glob, numpy as np
from tokenizers import Tokenizer
DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
NEEDLE = " The secret passcode for the blue vault is 7-4-1-9-2. Remember it."
QUESTION = "\n\nQuestion: What is the secret passcode for the blue vault?\nAnswer: The secret passcode for the blue vault is"


def ensure_wikitext():
    """Export WikiText-2 (raw) test/train text from the local Hugging Face datasets cache to data/ (once)."""
    import glob
    os.makedirs(DATA, exist_ok=True)
    for split in ("test", "train"):
        out = os.path.join(DATA, f"wikitext2_{split}.txt")
        if os.path.exists(out):
            continue
        hub = os.path.expanduser("~/.cache/huggingface/hub")
        files = glob.glob(os.path.join(hub, "datasets--Salesforce--wikitext", "snapshots", "*", "wikitext-2-raw-v1", f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError("WikiText-2 not in the HF cache: huggingface-cli download --repo-type dataset Salesforce/wikitext")
        import pyarrow.parquet as pq
        open(out, "w").write("".join(pq.read_table(files[0]).column("text").to_pylist()))


def tokenizer(model_dir):
    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))


def wikitext(tok, n, offset_chars=0):
    t = open(os.path.join(DATA, "wikitext2_test.txt")).read()[offset_chars:]
    return tok.encode(t[: 6 * n + 20000]).ids[:n]


def wikitext_ids(tok, n, offset=0):
    return np.array(wikitext(tok, n + offset)[offset:])


def code(tok, n):
    import numpy
    root = os.path.dirname(numpy.__file__)
    files = sorted(glob.glob(os.path.join(root, "lib", "_*_impl.py")) + glob.glob(os.path.join(root, "linalg", "_linalg.py")))
    ids = []
    for fn in files:
        ids += tok.encode(f"# file: {os.path.relpath(fn, root)}\n" + open(fn).read() + "\n").ids
        if len(ids) >= n:
            break
    return ids[:n]


def needle(tok, n, depth):
    """Wikitext haystack (different offset than the prose corpus) with a planted fact at `depth`
    and the question at the end; the answer tokens are NOT included."""
    q = tok.encode(QUESTION).ids
    nd = tok.encode(NEEDLE).ids
    hay = wikitext(tok, n - len(q) - len(nd), offset_chars=400000)
    p = int(depth * len(hay))
    return hay[:p] + nd + hay[p:] + q, (p, p + len(nd))


def build_corpora(model_dir, n=8192):
    ensure_wikitext()
    tok = tokenizer(model_dir)
    out = os.path.join(DATA, "corpus")
    os.makedirs(out, exist_ok=True)
    meta = {}
    np.save(os.path.join(out, "prose.npy"), np.array(wikitext(tok, n), np.int32)); meta["prose"] = {}
    np.save(os.path.join(out, "code.npy"), np.array(code(tok, n), np.int32)); meta["code"] = {}
    for d in (0.1, 0.5, 0.9):
        ids, span = needle(tok, n, d)
        name = f"needle{int(d*100):02d}"
        np.save(os.path.join(out, name + ".npy"), np.array(ids, np.int32))
        meta[name] = {"needle_span": span, "answer": "7-4-1-9-2"}
    import json
    json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
    return meta


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(DATA))
    from foveakv.load import snapshot_dir
    print(build_corpora(snapshot_dir()))
