"""ONE-TIME KV capture (torch, GPU, AMP bf16). Not part of the NumPy experiment pipeline.

For each corpus in data/corpus/*.npy and each full-attention layer, saves data/kv/<corpus>/L<i>.npz:
  k_q, k_s   : pre-RoPE keys (post k_norm), int8 per-token/per-head absmax + fp16 scale   (T,Hkv,d)/(T,Hkv)
  v_q, v_s   : values, same int8 format
  q          : pre-RoPE queries (post q_norm) for the last E positions, fp16             (E,Hq,d)
  q_spec     : speculative pre-RoPE queries for layer i, computed from the hidden state after
               layer i-4 (E7), fp16 (absent for the first attention layer)
  attn_in    : o_proj input (= softmax(qk)v * sigmoid(gate)) for the last E positions, fp16 (validation)
  gate       : gate logits for the last E positions, fp16 (validation)
For --fp16_ref corpora, also k16/v16 (fp16 copy of the bf16 cache) to validate int8 ~ lossless.
Also records the greedy fp continuation (needle sanity) in data/kv/<corpus>/meta.json.

Run (memory-capped):
  systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
     $FOVEAKV_TORCH_PYTHON tools/capture_kv.py
"""
import argparse, glob, json, os, time
import numpy as np, torch
from transformers import Qwen3_5ForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--eval", type=int, default=512, help="decode-window length E (queries kept)")
ap.add_argument("--fp16_ref", default="prose")
ap.add_argument("--corpora", default="")
args = ap.parse_args()

import sys
sys.path.insert(0, ROOT)
from foveakv.load import snapshot_dir
mdir = snapshot_dir()          # FOVEAKV_MODEL selects the checkpoint; FOVEAKV_KV_DIR the output folder
dev = "cuda"
model = Qwen3_5ForCausalLM.from_pretrained(mdir, dtype=torch.float32, low_cpu_mem_usage=True, device_map=dev).eval()
tok = AutoTokenizer.from_pretrained(mdir)
layers = model.model.layers
attn_ids = [i for i, t in enumerate(model.config.layer_types) if t == "full_attention"]
hd = model.config.head_dim
print("attention layers", attn_ids, "| GPU mem after load %.2f GB" % (torch.cuda.memory_allocated() / 1e9))


def int8(x):  # x: (T,H,d) float tensor on GPU -> int8 codes, fp16 scale
    x = x.float()
    s = x.abs().amax(-1).clamp_min(1e-8) / 127.0
    return torch.round(x / s[..., None]).clamp(-127, 127).to(torch.int8).cpu().numpy(), s.half().cpu().numpy()


names = args.corpora.split(",") if args.corpora else \
    sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(ROOT, "data/corpus/*.npy")))
for name in names:
    t0 = time.time()
    ids = np.load(os.path.join(ROOT, f"data/corpus/{name}.npy")).astype(np.int64)
    T, E = len(ids), args.eval
    cap = {i: {} for i in attn_ids}
    hidden = {}
    hooks = []
    for i in attn_ids:
        a = layers[i].self_attn
        def qhook(m, inp, out, i=i):
            cap[i]["q"] = out[0, -E:].detach()
            cap[i]["q_prefill"] = out[0, :-E].detach()
        hooks.append(a.q_norm.register_forward_hook(qhook))
        hooks.append(a.k_norm.register_forward_hook(lambda m, inp, out, i=i: cap[i].__setitem__("k", out[0].detach())))
        hooks.append(a.v_proj.register_forward_hook(lambda m, inp, out, i=i: cap[i].__setitem__("v", out[0].detach())))
        hooks.append(a.q_proj.register_forward_hook(lambda m, inp, out, i=i: cap[i].__setitem__(
            "gate", out[0, -E:].view(E, -1, 2 * hd)[..., hd:].reshape(E, -1).detach())))
        hooks.append(a.o_proj.register_forward_pre_hook(lambda m, inp, i=i: cap[i].__setitem__("attn_in", inp[0][0, -E:].detach())))
        hooks.append(layers[i].register_forward_hook(
            lambda m, inp, out, i=i: hidden.__setitem__(i, (out[0] if isinstance(out, tuple) else out)[0, -E:].detach())))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        model.model(input_ids=torch.tensor(ids[None], device=dev), use_cache=False)
        for h in hooks:
            h.remove()
        # speculative queries for layer j from the hidden state after layer j-4
        for j in attn_ids:
            if j - 4 in hidden:
                L = layers[j]
                x = L.input_layernorm(hidden[j - 4])
                qg = L.self_attn.q_proj(x).view(E, -1, 2 * hd)[..., :hd]
                cap[j]["q_spec"] = L.self_attn.q_norm(qg)

    od = os.path.join(os.environ.get("FOVEAKV_KV_DIR", os.path.join(ROOT, "data", "kv")), name)
    os.makedirs(od, exist_ok=True)
    Hkv = model.config.num_key_value_heads
    for i in attn_ids:
        c = cap[i]
        k = c["k"].view(T, Hkv, hd)
        v = c["v"].view(T, Hkv, hd)
        kq, ks = int8(k)
        vq, vs = int8(v)
        # per-channel second moments of prefill queries, pre- and post-RoPE (closed-form calibration stats)
        qp = c["q_prefill"].float()                                            # (P, Hq, d) pre-RoPE
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        Pn = qp.shape[0]
        cos, sin = model.model.rotary_emb(qp, torch.arange(Pn, device=dev)[None])
        qpost, _ = apply_rotary_pos_emb(qp.transpose(0, 1)[None], qp.transpose(0, 1)[None], cos.float(), sin.float())
        q2_pre = (qp ** 2).mean(0).cpu().numpy()
        q2_post = (qpost[0].transpose(0, 1) ** 2).mean(0).cpu().numpy()
        out = dict(k_q=kq, k_s=ks, v_q=vq, v_s=vs, q=c["q"].half().cpu().numpy(), q2_pre=q2_pre, q2_post=q2_post,
                   attn_in=c["attn_in"].half().cpu().numpy(), gate=c["gate"].half().cpu().numpy())
        if "q_spec" in c:
            out["q_spec"] = c["q_spec"].half().cpu().numpy()
        if name == args.fp16_ref:
            out["k16"] = k.half().cpu().numpy()
            out["v16"] = v.half().cpu().numpy()
            assert np.isfinite(out["k16"]).all() and np.isfinite(out["v16"]).all()
        np.savez(os.path.join(od, f"L{i}.npz"), **out)
    del cap, hidden

    # fp greedy continuation (needle sanity: does the unquantized model retrieve at this length?)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        g = model.generate(torch.tensor(ids[None], device=dev), max_new_tokens=12, do_sample=False)
    cont = tok.decode(g[0, T:].tolist())
    json.dump({"T": T, "E": E, "attn_layers": attn_ids, "fp_greedy": cont,
               "amp": "fp32 weights + autocast bf16"}, open(os.path.join(od, "meta.json"), "w"), indent=1)
    torch.cuda.empty_cache()
    print(f"{name}: T={T} {time.time()-t0:.1f}s peak GPU {torch.cuda.max_memory_allocated()/1e9:.2f} GB | fp greedy: {cont!r}")
