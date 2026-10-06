"""ONE-TIME KV capture with layer streaming (torch, CPU or GPU, AMP bf16). Same output as tools/capture_kv.py.

Use it when the model does not fit in memory, or when the GPU is busy. Weights stay in the safetensors file
(memory-mapped); one decoder layer is materialized at a time and all corpora go through it before the next one,
so each weight is read once. Attention runs as causal SDPA without a mask (no T x T matrix).
Peak memory = one layer in fp32 + hidden states of all corpora + the captured K/V of the current layer.

Differences from capture_kv.py: no generated continuation; instead meta.json records the model's next-token
prediction at the last position and the perplexity of each corpus (a forward-pass sanity check).

Run (memory-capped):
  systemd-run --user --scope -p MemoryMax=4G -p MemorySwapMax=0 \
     $FOVEAKV_TORCH_PYTHON tools/capture_kv_stream.py --device cpu --layers 0,4,9
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from foveakv.load import snapshot_dir, SafeTensors, bf16_to_f32
from foveakv.text import CORPUS

ap = argparse.ArgumentParser()
ap.add_argument("--eval", type=int, default=512, help="decode-window length E (queries kept)")
ap.add_argument("--fp16_ref", default="prose", help="corpora that also get an fp16 K/V copy (comma list, or 'all')")
ap.add_argument("--corpora", default="")
ap.add_argument("--layers", default="", help="comma list of attention layers to keep (default: all)")
ap.add_argument("--device", default="cpu")
args = ap.parse_args()

torch.manual_seed(0)
mdir = snapshot_dir()
dev = args.device
full = AutoConfig.from_pretrained(mdir)
hybrid = full.model_type.startswith("qwen3_5")
cfg = full.text_config if hybrid else full
cfg._attn_implementation = "sdpa"
if hybrid and dev == "cpu":     # flash-linear-attention / causal-conv1d kernels are Triton/CUDA only: use torch fallbacks
    import transformers.models.qwen3_5.modeling_qwen3_5 as mq
    mq.FusedRMSNormGated = mq.chunk_gated_delta_rule = mq.fused_recurrent_gated_delta_rule = None
    mq.causal_conv1d_fn = mq.causal_conv1d_update = None
with torch.device("meta"):
    if hybrid:
        from transformers import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM(cfg)
    else:
        model = AutoModelForCausalLM.from_config(cfg)
model.eval()
layers = model.model.layers
rotary = type(model.model.rotary_emb)(config=cfg, device=dev)
ST = SafeTensors(mdir)
keys = set(ST.keys())
PRE = next(p for p in ("model.language_model.layers", "model.layers") if f"{p}.0.input_layernorm.weight" in keys)
ROOTKEY = PRE[: -len(".layers")]
EMB = ST.get_raw(f"{ROOTKEY}.embed_tokens.weight")[0]                     # (V, H) bf16 bits, memory-mapped
HEAD = ST.get_raw("lm_head.weight")[0] if "lm_head.weight" in keys else EMB
tok = AutoTokenizer.from_pretrained(mdir)

types = getattr(cfg, "layer_types", None) or ["full_attention"] * cfg.num_hidden_layers
attn_ids = [i for i, t in enumerate(types) if t == "full_attention"]
if args.layers:
    attn_ids = [i for i in map(int, args.layers.split(",")) if i in attn_ids]
hd, Hq, Hkv = cfg.head_dim, cfg.num_attention_heads, cfg.num_key_value_heads
rp = getattr(cfg, "rope_parameters", None) or {}
theta = float(rp.get("rope_theta", getattr(cfg, "rope_theta", 1e4)))
rot_dim = int(hd * float(rp.get("partial_rotary_factor", getattr(cfg, "partial_rotary_factor", 1.0))))
amp = torch.autocast(dev, dtype=torch.bfloat16)


def materialize(i):
    L = layers[i]
    sd = {n: torch.from_numpy(ST.get(f"{PRE}.{i}.{n}")) for n, _ in L.named_parameters()}
    L.to_empty(device=dev)
    missing, unexpected = L.load_state_dict(sd, strict=False)
    bad = [m for m in missing if m in dict(L.named_parameters())]
    assert not bad and not unexpected, (bad, unexpected)
    return L


def release(i):
    layers[i].to("meta")


L0 = materialize(attn_ids[0])
gated = L0.self_attn.q_proj.out_features == 2 * Hq * hd                   # Qwen3.5 output gate
apply_rotary_pos_emb = sys.modules[type(L0.self_attn).__module__].apply_rotary_pos_emb
bufs = [n for n, _ in L0.named_buffers()]
release(attn_ids[0])
assert not bufs, f"layer buffers would be uninitialized: {bufs}"
print(f"{os.path.basename(os.path.dirname(os.path.dirname(mdir)))}: attention layers {attn_ids} | gated {gated} "
      f"| head_dim {hd} Hq {Hq} Hkv {Hkv} | rot_dim {rot_dim} theta {theta}", flush=True)


def int8(x):  # (T,H,d) -> int8 codes, fp16 scale
    x = x.float()
    s = x.abs().amax(-1).clamp_min(1e-8) / 127.0
    return torch.round(x / s[..., None]).clamp(-127, 127).to(torch.int8).cpu().numpy(), s.half().cpu().numpy()


names = args.corpora.split(",") if args.corpora else \
    sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(CORPUS, "*.npy")))
IDS = {n: np.load(os.path.join(CORPUS, f"{n}.npy")).astype(np.int64) for n in names}
E = args.eval
X = {n: torch.from_numpy(bf16_to_f32(EMB[IDS[n]])).to(dev)[None] for n in names}      # (1,T,H) fp32
PE = {}
for n in names:
    T = len(IDS[n])
    pos = torch.arange(T, device=dev)[None]
    PE[n] = rotary(X[n], pos)
need_hidden = {j - 4 for j in attn_ids if j >= 4}
HID = {n: {} for n in names}
KVDIR = os.environ.get("FOVEAKV_KV_DIR", os.path.join(ROOT, "data", "kv"))
t0 = time.time()

for li in range(cfg.num_hidden_layers):
    L = materialize(li)
    keep = li in attn_ids
    for n in names:
        T = len(IDS[n])
        cap, hooks = {}, []
        if keep:
            a = L.self_attn
            def qhook(m, inp, out):
                cap["q"] = out[0, -E:].detach().float()
                cap["q_prefill"] = out[0, :-E].detach().float()
            hooks.append(a.q_norm.register_forward_hook(qhook))
            hooks.append(a.k_norm.register_forward_hook(lambda m, inp, out: cap.__setitem__("k", out[0].detach().float())))
            hooks.append(a.v_proj.register_forward_hook(lambda m, inp, out: cap.__setitem__("v", out[0].detach().float())))
            if gated:
                hooks.append(a.q_proj.register_forward_hook(lambda m, inp, out: cap.__setitem__(
                    "gate", out[0, -E:].view(E, -1, 2 * hd)[..., hd:].reshape(E, -1).detach().float())))
            hooks.append(a.o_proj.register_forward_pre_hook(lambda m, inp: cap.__setitem__("attn_in", inp[0][0, -E:].detach().float())))
        with torch.no_grad(), amp:
            pos = torch.arange(T, device=dev)[None]
            y = L(X[n], position_embeddings=PE[n], attention_mask=None, position_ids=pos)
            X[n] = (y[0] if isinstance(y, tuple) else y).float()
            for h in hooks:
                h.remove()
            if keep and li - 4 in HID[n]:          # speculative query for this layer from the hidden state after layer li-4
                xs = L.input_layernorm(HID[n][li - 4])
                qg = L.self_attn.q_proj(xs).view(E, -1, (2 if gated else 1) * hd)[..., :hd]
                cap["q_spec"] = L.self_attn.q_norm(qg).float()
        if li in need_hidden:
            HID[n][li] = X[n][0, -E:].clone()
        if not keep:
            continue
        od = os.path.join(KVDIR, n)
        os.makedirs(od, exist_ok=True)
        k, v = cap["k"].view(T, Hkv, hd), cap["v"].view(T, Hkv, hd)
        kq, ks = int8(k)
        vq, vs = int8(v)
        qp = cap["q_prefill"]                                              # (P,Hq,d) pre-RoPE
        cos, sin = rotary(qp, torch.arange(qp.shape[0], device=dev)[None])
        qpost, _ = apply_rotary_pos_emb(qp.transpose(0, 1)[None], qp.transpose(0, 1)[None], cos.float(), sin.float())
        out = dict(k_q=kq, k_s=ks, v_q=vq, v_s=vs, q=cap["q"].half().cpu().numpy(),
                   q2_pre=(qp ** 2).mean(0).cpu().numpy(), q2_post=(qpost[0].transpose(0, 1) ** 2).mean(0).cpu().numpy(),
                   attn_in=cap["attn_in"].half().cpu().numpy())
        if gated:
            out["gate"] = cap["gate"].half().cpu().numpy()
        if "q_spec" in cap:
            out["q_spec"] = cap["q_spec"].half().cpu().numpy()
        if args.fp16_ref == "all" or n in args.fp16_ref.split(","):
            out["k16"], out["v16"] = k.half().cpu().numpy(), v.half().cpu().numpy()
            assert np.isfinite(out["k16"]).all() and np.isfinite(out["v16"]).all()
        np.savez(os.path.join(od, f"L{li}.npz"), **out)
        del cap
    release(li)
    print(f"layer {li:2d} done ({'captured' if keep else 'pass'}) {time.time()-t0:.0f}s", flush=True)

# final norm + LM head: perplexity per corpus and next-token prediction at the last position (sanity checks)
norm = model.model.norm
norm.to_empty(device=dev)
norm.load_state_dict({"weight": torch.from_numpy(ST.get(f"{ROOTKEY}.norm.weight"))})
CH = 4096
for n in names:
    ids = IDS[n]
    T = len(ids)
    with torch.no_grad():
        h = norm(X[n][0]).float()                                         # (T,H)
        lse = torch.full((T,), -float("inf"), device=dev)
        best = torch.full((T,), -float("inf"), device=dev)
        arg = torch.zeros(T, dtype=torch.long, device=dev)
        tgt = torch.zeros(T, device=dev)
        nxt = torch.from_numpy(np.append(ids[1:], 0)).to(dev)
        for v0 in range(0, HEAD.shape[0], CH):
            W = torch.from_numpy(bf16_to_f32(HEAD[v0:v0 + CH])).to(dev)
            lg = h @ W.T                                                   # (T, CH)
            lse = torch.logaddexp(lse, torch.logsumexp(lg, -1))
            m, a = lg.max(-1)
            upd = m > best
            best, arg = torch.where(upd, m, best), torch.where(upd, a + v0, arg)
            inr = (nxt >= v0) & (nxt < v0 + lg.shape[1])
            tgt[inr] = lg[inr, (nxt[inr] - v0)]
        nll = (lse - tgt)[: T - 1]
        ppl = float(torch.exp(nll.mean()))
        last = tok.decode([int(arg[-1])])
    json.dump({"T": T, "E": E, "attn_layers": attn_ids, "model": os.path.basename(os.path.dirname(os.path.dirname(mdir))),
               "head_dim": hd, "Hq": Hq, "Hkv": Hkv, "rot_dim": rot_dim, "theta": theta, "gated": gated,
               "ppl_fp": ppl, "fp_next_token": last, "amp": "fp32 weights + autocast bf16, layer streaming"},
              open(os.path.join(KVDIR, n, "meta.json"), "w"), indent=1)
    print(f"{n}: T={T} ppl {ppl:.2f} | next token at end: {last!r}", flush=True)
print(f"total {time.time()-t0:.0f}s")
