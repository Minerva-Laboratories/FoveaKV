"""Safetensors loading with NumPy only. bf16 -> fp32 via uint16 << 16. Tensors are memory-mapped
(no RAM held) and converted on demand, because the host has little free RAM."""
import glob, json, os, struct
import numpy as np

HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")


DEFAULT_MODEL = os.environ.get("FOVEAKV_MODEL", "Qwen/Qwen3.5-0.8B")   # any Qwen3.5 checkpoint in the local HF cache


def snapshot_dir(repo=None):
    repo = repo or DEFAULT_MODEL
    d = glob.glob(os.path.join(HF_HUB, "models--" + repo.replace("/", "--"), "snapshots", "*"))
    if not d:
        raise FileNotFoundError(repo)
    return d[0]


def model_weights():
    """Weights of FOVEAKV_MODEL: its safetensors, or the llama.cpp GGUF named by FOVEAKV_GGUF (load_gguf.py)."""
    if os.environ.get("FOVEAKV_GGUF"):
        from types import SimpleNamespace
        from .load_gguf import GGUFTensors
        c = json.load(open(os.path.join(snapshot_dir(), "config.json")))
        return GGUFTensors(os.environ["FOVEAKV_GGUF"], SimpleNamespace(**c.get("text_config", c)))
    return SafeTensors(snapshot_dir())


def bf16_to_f32(u16):
    """Raw bf16 bits (uint16 array) -> float32."""
    return (np.asarray(u16, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


class SafeTensors:
    """Lazy, mmap-backed reader for one or more .safetensors files."""
    _DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "I64": np.int64}

    def __init__(self, folder):
        self.index = {}
        for fn in sorted(glob.glob(os.path.join(folder, "*.safetensors"))):
            with open(fn, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                hdr = json.loads(f.read(n))
            mm = np.memmap(fn, dtype=np.uint8, mode="r")
            for k, v in hdr.items():
                if k != "__metadata__":
                    self.index[k] = (mm, 8 + n, v)

    def keys(self):
        return self.index.keys()

    def get(self, name, dtype=np.float32):
        mm, off, meta = self.index[name]
        a, b = meta["data_offsets"]
        raw = mm[off + a: off + b].view(self._DT[meta["dtype"]]).reshape(meta["shape"])
        if meta["dtype"] == "BF16":
            return bf16_to_f32(raw).astype(dtype, copy=False)
        return np.array(raw, dtype=dtype)

    def get_raw(self, name):
        """mmap view without conversion (bf16 stays uint16)."""
        mm, off, meta = self.index[name]
        a, b = meta["data_offsets"]
        return mm[off + a: off + b].view(self._DT[meta["dtype"]]).reshape(meta["shape"]), meta["dtype"]


def load_config(folder):
    with open(os.path.join(folder, "config.json")) as f:
        cfg = json.load(f)
    return cfg.get("text_config", cfg)


def attention_geometry(cfg):
    """Attention shape and RoPE of a Hugging Face config, for both layouts in use:
    Qwen3.5 (nested ``text_config``, ``rope_parameters`` with ``rope_theta`` / ``partial_rotary_factor``) and
    Qwen3 / Llama-style flat configs (top-level ``rope_theta``, full rotary). ``cfg`` is a config.json dict or a
    transformers config object. Returns head_dim, rot_dim (rotated channels per head), theta, Hq and Hkv."""
    get = (lambda k, d=None: cfg.get(k, d)) if isinstance(cfg, dict) else (lambda k, d=None: getattr(cfg, k, d))
    tc = get("text_config")
    if tc is not None and tc is not cfg:
        return attention_geometry(tc)
    rp = get("rope_parameters") or {}
    if not isinstance(rp, dict):
        rp = vars(rp)
    Hq = int(get("num_attention_heads"))
    hd = int(get("head_dim") or get("hidden_size") // Hq)
    theta = float(rp.get("rope_theta", get("rope_theta", 1e4)))
    prf = float(rp.get("partial_rotary_factor", get("partial_rotary_factor", 1.0)))
    return dict(head_dim=hd, rot_dim=int(hd * prf), theta=theta, Hq=Hq, Hkv=int(get("num_key_value_heads", Hq)))
