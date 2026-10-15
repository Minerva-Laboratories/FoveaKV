"""Read a llama.cpp GGUF of a Qwen3.5 / Qwen3.5-MoE model as Hugging Face tensors (same interface as load.SafeTensors).

For models that are only available as GGUF. Tensors are dequantized on demand (gguf-py) and mapped back to the HF
names and layouts, undoing what llama.cpp's converter does (conversion/qwen.py, Qwen3NextModel and
_LinearAttentionVReorderBase):
  - RMSNorm weights are stored as 1 + w (HF Qwen3.5 norms compute x * (1 + w)), except the gated DeltaNet norm;
  - A_log is stored as -exp(A_log); conv1d is squeezed;
  - linear-attention V heads are stored tiled by K head ([K0, K1, ..., K0, K1, ...]) instead of grouped;
  - experts' gate and up projections are stored as two 3D tensors (HF fuses them into gate_up_proj).
The weights are the quantized ones (e.g. Q4_K experts), so the captured cache is that of the quantized model.
gguf-py is taken from FOVEAKV_GGUF_PY (default ~/llama.cpp/gguf-py).
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.expanduser(os.environ.get("FOVEAKV_GGUF_PY", "~/llama.cpp/gguf-py")))
from gguf import GGUFReader                    # noqa: E402
from gguf.quants import dequantize            # noqa: E402

LAYER = {
    "attn_norm.weight": "input_layernorm.weight", "post_attention_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight", "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight", "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight", "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight", "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight", "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight", "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight", "ssm_a": "linear_attn.A_log", "ssm_dt.bias": "linear_attn.dt_bias",
    "ffn_gate_inp.weight": "mlp.gate.weight", "ffn_down_exps.weight": "mlp.experts.down_proj",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight", "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight", "ffn_gate_inp_shexp.weight": "mlp.shared_expert_gate.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight", "ffn_down.weight": "mlp.down_proj.weight",
}
TOP = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight", "output.weight": "lm_head.weight"}
PLUS_ONE = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight", "model.norm.weight")


def _f32_to_bf16_bits(x):
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)          # round to nearest even


class GGUFTensors:
    def __init__(self, path, cfg):
        self.r = GGUFReader(path)
        self.t = {t.name: t for t in self.r.tensors}
        self.cfg = cfg
        self.hf = {}                                                          # HF name -> GGUF name(s)
        for g in self.t:
            if g in TOP:
                self.hf[TOP[g]] = g
            elif g.startswith("blk."):
                _, b, rest = g.split(".", 2)
                if rest in LAYER:
                    self.hf[f"model.layers.{b}.{LAYER[rest]}"] = g
                elif rest == "ffn_gate_exps.weight":
                    self.hf[f"model.layers.{b}.mlp.experts.gate_up_proj"] = (g, f"blk.{b}.ffn_up_exps.weight")
        self._raw = {}

    def keys(self):
        return list(self.hf)

    def _deq(self, g, chunk=16):
        """Dequantize; 3D expert tensors in slices of `chunk` experts (bounds the gguf-py temporaries)."""
        t = self.t[g]
        shape = [int(x) for x in reversed(t.shape)]
        d = t.data
        if d.ndim == 3:
            out = np.empty(shape, np.float32)
            for i in range(0, d.shape[0], chunk):
                out[i:i + chunk] = dequantize(np.ascontiguousarray(d[i:i + chunk]), t.tensor_type)
            return out
        return np.asarray(dequantize(d, t.tensor_type), np.float32).reshape(shape)

    def _v_untile(self, x, axis, d):
        """Inverse of the converter's grouped -> tiled V-head reorder along `axis` (blocks of d)."""
        c = self.cfg
        nk, nv = c.linear_num_key_heads, c.linear_num_value_heads
        if nk == nv:
            return x
        r = nv // nk
        x = np.moveaxis(x, axis, 0)
        sh = x.shape
        x = x.reshape(r, nk, d, *sh[1:]).swapaxes(0, 1).reshape(sh)
        return np.ascontiguousarray(np.moveaxis(x, 0, axis))

    def get(self, name):
        g = self.hf[name]
        if isinstance(g, tuple):                                              # (E, ff, H) gate and up -> (E, 2 ff, H)
            return np.concatenate([self._deq(g[0]), self._deq(g[1])], axis=1)
        x = self._deq(g)
        c = self.cfg
        dv, dk, nk = getattr(c, "linear_value_head_dim", 0), getattr(c, "linear_key_head_dim", 0), getattr(c, "linear_num_key_heads", 0)
        if name.endswith(PLUS_ONE):
            x = x - 1.0
        if name.endswith("linear_attn.A_log"):
            x = self._v_untile(np.log(-x), 0, 1)
        elif name.endswith("linear_attn.dt_bias"):
            x = self._v_untile(x, 0, 1)
        elif name.endswith("linear_attn.conv1d.weight"):                     # (C, K) -> (C, 1, K), V channels untiled
            qk = 2 * dk * nk
            x = np.concatenate([x[:qk], self._v_untile(x[qk:], 0, dv)])[:, None, :]
        elif name.endswith("linear_attn.in_proj_qkv.weight"):
            qk = 2 * dk * nk
            x = np.concatenate([x[:qk], self._v_untile(x[qk:], 0, dv)])
        elif name.endswith("linear_attn.in_proj_z.weight"):
            x = self._v_untile(x, 0, dv)
        elif name.endswith(("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight")):
            x = self._v_untile(x, 0, 1)
        elif name.endswith("linear_attn.out_proj.weight"):
            x = self._v_untile(x, 1, dv)
        elif name.endswith("mlp.shared_expert_gate.weight"):
            x = x.reshape(1, -1)
        return np.ascontiguousarray(x, np.float32)

    def get_raw(self, name):
        """(bf16 bits, None) like SafeTensors.get_raw; used for the embedding and LM head (kept in memory)."""
        if name not in self._raw:
            self._raw[name] = _f32_to_bf16_bits(self.get(name))
        return self._raw[name], None
