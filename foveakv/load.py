"""Safetensors loading with NumPy only. bf16 -> fp32 via uint16 << 16. Tensors are memory-mapped
(no RAM held) and converted on demand, because the host has little free RAM."""
import glob, json, os, struct
import numpy as np

HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")


def snapshot_dir(repo="Qwen/Qwen3.5-0.8B"):
    d = glob.glob(os.path.join(HF_HUB, "models--" + repo.replace("/", "--"), "snapshots", "*"))
    if not d:
        raise FileNotFoundError(repo)
    return d[0]


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
