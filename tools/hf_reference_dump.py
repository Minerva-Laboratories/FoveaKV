"""ORACLE ONLY (not part of the NumPy pipeline): HF transformers on GPU with AMP (fp32 weights + autocast bf16)
on the same token ids; dumps per-position argmax + NLL to cross-check foveakv.model_hybrid.
Logits are computed in chunks of 256 positions (never the full T x 248K matrix).
Run capped:  systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 <venv>/bin/python tools/hf_reference_dump.py ids.npy out.npz"""
import sys, glob, os, numpy as np, torch
from transformers import Qwen3_5ForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foveakv.load import snapshot_dir
d = snapshot_dir()
ids = np.load(sys.argv[1]).astype(np.int64)
m = Qwen3_5ForCausalLM.from_pretrained(d, dtype=torch.float32, low_cpu_mem_usage=True, device_map="cuda").eval()
x = torch.tensor(ids[None], device="cuda")
arg, nll = [], []
FP32 = os.environ.get("ORACLE_FP32") == "1"   # pure fp32 (no autocast) for the top-1 gate
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=not FP32):
    h = m.model(input_ids=x, use_cache=False).last_hidden_state[0]
    for s in range(0, len(ids), 256):
        lg = m.lm_head(h[s:s + 256]).float()
        arg.append(lg.argmax(-1).cpu())
        lp = torch.log_softmax(lg, -1)
        tgt = x[0, s + 1:s + 257]
        nll.append(-lp[:len(tgt)].gather(1, tgt[:, None])[:, 0].cpu())
arg, nll = torch.cat(arg).numpy(), torch.cat(nll).double().numpy()
np.savez(sys.argv[2], argmax=arg, nll=nll)
print("hf", "fp32" if FP32 else "AMP bf16", "GPU ppl", float(np.exp(nll.mean())))
