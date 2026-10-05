import numpy as np, time
rng=np.random.default_rng(0); t0=time.time()
N,d,G=8192,128,128   # tokens, head dim, quant group (tokens per scale for keys)

def quant(x,B,axis):  # midpoint uniform quant, returns codes + dequant fn
    lo=x.min(axis,keepdims=True); hi=x.max(axis,keepdims=True); s=(hi-lo)+1e-8
    c=np.clip(np.floor((x-lo)/s*2**B),0,2**B-1).astype(np.int32)
    return c,lo,s
def deq(c,lo,s,B): return lo+(c+0.5)/2**B*s
def kquant(K,B):  # per-channel within token groups (KIVI-style)
    Kg=K.reshape(-1,G,d); c,lo,s=quant(Kg,B,1); return c,lo,s
def make_kv(outlier=True):
    sc=np.ones(d); 
    if outlier: sc[rng.choice(d,6,replace=False)]=12
    K=(rng.standard_normal((N,d))+rng.standard_normal(d)*2)*sc
    V=rng.standard_normal((N,d))
    return K,V
def make_q(K,peak):  # query aligned with a few "relevant" keys -> peaked attention
    idx=rng.choice(N,8,replace=False)
    q=K[idx].mean(0)-K.mean(0); q=q/np.linalg.norm(q)*peak + rng.standard_normal(d)*0.3
    return q
def softmax(s): e=np.exp(s-s.max()); return e/e.sum()
def H(n):  # normalized Hadamard
    h=np.array([[1.]])
    while h.shape[0]<n: h=np.block([[h,h],[h,-h]])
    return h/np.sqrt(n)

b,b2=2,2; B=b+b2
K,V=make_kv()
# --- Test 1: nested grid exactness
c,lo,s=kquant(K,B); hi=c>>b2; 
full=deq(c,lo,s,B); rebuilt=deq((hi<<b2)|(c&(2**b2-1)),lo,s,B)
print("T1 nested bit-planes exact:",np.allclose(full,rebuilt),
      "| coarse==direct 2-bit:",np.allclose(deq(hi,lo,s,b),deq(*kquant(K,b),b)))

def run(K,V,q,k_frac,rot=False,sel_bits=b,page=1):
    if rot: R=H(d); K=K@R; q=q@R; V=V@R   # orthogonal: q.k preserved; rotate V back at end
    c,lo,s=kquant(K,B); Kf=deq(c,lo,s,B).reshape(N,d); Kc=deq(c>>b2,lo,s,b).reshape(N,d)
    Ks=deq(c>>(B-sel_bits),lo,s,sel_bits).reshape(N,d)
    cv,lv,sv=quant(V,B,1); Vf=deq(cv,lv,sv,B); Vc=deq(cv>>b2,lv,sv,b)
    sq=np.sqrt(d); a=softmax(K@q/sq); o=a@V
    ts=Ks@q/sq; k=int(k_frac*N)
    if page>1:  # page selection by max score in page
        ps=ts.reshape(-1,page).max(1); hp=np.argsort(-ps)[:max(1,k//page)]
        hot=(hp[:,None]*page+np.arange(page)).ravel()
    else: hot=np.argpartition(-ts,k)[:k]
    Kh=Kc.copy(); Vh=Vc.copy(); Kh[hot]=Kf[hot]; Vh[hot]=Vf[hot]
    ah=softmax(Kh@q/sq); oh=ah@Vh
    o2=softmax(Kc@q/sq)@Vc; o4=softmax(Kf@q/sq)@Vf
    true_top=np.argpartition(-a,k)[:k]
    mH=a[hot].sum(); mopt=a[true_top].sum()
    rel=lambda x: np.linalg.norm(x-o)/np.linalg.norm(o)
    return dict(err_dyn=rel(oh),err2=rel(o2),err4=rel(o4),mH=mH,mopt=mopt,
                overlap=len(np.intersect1d(hot,true_top))/len(hot),top1pct=np.sort(a)[-k:].sum())

print("\nT2/T3 error vs hot fraction (rel L2 err of attention output; avg 20 queries, peak=6)")
print(" k%   dyn(2+2)  uni2    uni4    m_H   m_opt  sel-overlap")
for kf in [0.01,0.02,0.05,0.1]:
    R=[run(K,V,make_q(K,6),kf) for _ in range(20)]
    m=lambda key: np.mean([r[key] for r in R])
    print(f"{kf*100:4.0f}  {m('err_dyn'):.4f}   {m('err2'):.4f}  {m('err4'):.4f}  {m('mH'):.3f} {m('mopt'):.3f}  {m('overlap'):.2f}")

print("\nT4 effect of attention sharpness (k=5%)")
for pk in [2,4,6,10]:
    R=[run(K,V,make_q(K,pk),0.05) for _ in range(20)]
    m=lambda key: np.mean([r[key] for r in R])
    print(f" peak={pk:2d}: dyn {m('err_dyn'):.4f} uni2 {m('err2'):.4f} uni4 {m('err4'):.4f} m_opt {m('mopt'):.2f}")

print("\nT5 selection with fewer bits / Hadamard / pages (k=5%, peak=6)")
for name,kw in [("2-bit sel",{}),("1-bit sel (MSB plane)",dict(sel_bits=1)),
                ("2-bit sel + Hadamard",dict(rot=True)),("1-bit sel + Hadamard",dict(rot=True,sel_bits=1)),
                ("pages of 32",dict(page=32)),("pages of 32 + Hadamard",dict(page=32,rot=True))]:
    R=[run(K,V,make_q(K,6),0.05,**kw) for _ in range(20)]
    m=lambda key: np.mean([r[key] for r in R])
    print(f" {name:24s} err {m('err_dyn'):.4f}  m_H {m('mH'):.3f}/{m('mopt'):.3f}  overlap {m('overlap'):.2f}")

print("\nT6 online-softmax delta correction exactness")
c,lo,s=kquant(K,B); Kf=deq(c,lo,s,B).reshape(N,d); Kc=deq(c>>b2,lo,s,b).reshape(N,d)
q=make_q(K,6); sc=Kc@q/np.sqrt(d); sf=Kf@q/np.sqrt(d); hot=np.argpartition(-sc,400)[:400]
mx=sc.max(); l=np.exp(sc-mx).sum(); acc=np.exp(sc-mx)@V
l2=l+(np.exp(sf[hot]-mx)-np.exp(sc[hot]-mx)).sum(); acc2=acc+(np.exp(sf[hot]-mx)-np.exp(sc[hot]-mx))@V[hot]
s_ref=sc.copy(); s_ref[hot]=sf[hot]; ref=softmax(s_ref)@V
print(" max abs diff vs recompute:",np.abs(acc2/l2-ref).max())

print("\nT7 GQA union of hot sets (k=5% per head)")
for g in [1,4,8,16]:
    hots=[set(np.argpartition(-(K@make_q(K,6)),int(.05*N))[:int(.05*N)]) for _ in range(g)]
    print(f" group {g:2d}: union = {len(set().union(*hots))/N*100:.1f}% of tokens")

print("\nT8 temporal reuse: query drifts as AR(1) over decode steps")
for rho in [0.99,0.95,0.9]:
    q=make_q(K,6); hits=[]; prev=None; k=int(.05*N)
    for t in range(50):
        q=rho*q+np.sqrt(1-rho**2)*make_q(K,6)
        hot=set(np.argpartition(-(K@q),k)[:k])
        if prev: hits.append(len(hot&prev)/k)
        prev=hot
    print(f" rho={rho}: step-to-step hot-set hit rate {np.mean(hits):.2f}")
print(f"\nruntime {time.time()-t0:.1f}s")
