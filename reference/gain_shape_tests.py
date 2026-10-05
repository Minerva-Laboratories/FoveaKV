import numpy as np; from math import erf,sqrt,pi,exp
rng=np.random.default_rng(1)
Phi=np.vectorize(lambda x:0.5*(1+erf(x/sqrt(2)))); phi=lambda x:np.exp(-x*x/2)/sqrt(2*pi)
def cmean(a,b): return (phi(a)-phi(b))/np.maximum(Phi(b)-Phi(a),1e-300)   # E[X|a<X<b], X~N(0,1)
def mse(bnd,lev):  # exact MSE for N(0,1)
    e=np.concatenate([[-np.inf],bnd,[np.inf]]); t=0
    for i,c in enumerate(lev):
        a,b=e[i],e[i+1]; p=Phi(b)-Phi(a)
        m2=p - (b*phi(b) if np.isfinite(b) else 0) + (a*phi(a) if np.isfinite(a) else 0)  # E[X^2 1{a<X<b}]
        t+=m2-2*c*(phi(a)-phi(b))+c*c*p
    return t
def lloyd(n,it=300):
    lev=np.linspace(-2,2,n)
    for _ in range(it):
        bnd=(lev[1:]+lev[:-1])/2; e=np.concatenate([[-np.inf],bnd,[np.inf]])
        lev=np.array([cmean(e[i],e[i+1]) for i in range(n)])
    return bnd,lev
def coarse_from(bnd,group):  # merge consecutive fine cells, centroid recon
    e=np.concatenate([[-np.inf],bnd,[np.inf]]); cb=e[group::group][:-1]
    ce=np.concatenate([[-np.inf],cb,[np.inf]])
    return cb,np.array([cmean(ce[i],ce[i+1]) for i in range(len(cb)+1)])
def refine(cb,k,it=200):  # coarse-first: conditional Lloyd inside each coarse cell
    ce=np.concatenate([[-np.inf],cb,[np.inf]]); fb=[]
    for i in range(len(ce)-1):
        a,b=ce[i],ce[i+1]; lo=max(a,-4); hi=min(b,4); lev=np.linspace(lo,hi,k+2)[1:-1]
        for _ in range(it):
            bb=(lev[1:]+lev[:-1])/2; e=np.concatenate([[a],bb,[b]]); lev=np.array([cmean(e[j],e[j+1]) for j in range(k)])
        fb+=list(bb)+([b] if np.isfinite(b) else [])
    fb=np.array(fb); e=np.concatenate([[-np.inf],fb,[np.inf]])
    return fb,np.array([cmean(e[j],e[j+1]) for j in range(len(fb)+1)])
def unif(n,c):
    s=2*c/n; bnd=-c+s*np.arange(1,n); e=np.concatenate([[-np.inf],bnd,[np.inf]])
    return bnd,np.array([-c+s*(i+.5) for i in range(n)])
db=lambda m:f"{m:.4f} ({-10*np.log10(m):.2f} dB)"
b2,l2=lloyd(4); b4,l4=lloyd(16)
print("Gaussian MSE per coordinate (sigma=1), coarse 2-bit | fine 4-bit")
print(" independent Lloyd-Max (not nested): ",db(mse(b2,l2)),"|",db(mse(b4,l4)))
cb,cl=coarse_from(b4,4); print(" fine-first nested (merge 4b cells): ",db(mse(cb,cl)),"|",db(mse(b4,l4)))
fb,fl=refine(b2,4); print(" coarse-first nested (split 2b cells):",db(mse(b2,l2)),"|",db(mse(fb,fl)))
best=min((mse(*unif(16,c)),c) for c in np.linspace(2,4,41)); bu,lu=unif(16,best[1]); cbu,clu=coarse_from(bu,4)
print(f" uniform nested (clip {best[1]:.2f}σ, centroid coarse):",db(mse(cbu,clu)),"|",db(best[0]))
np.save("lut.npy",np.stack([fb,]) if False else fl)

# --- gaussianization check + end-to-end attention with gain-shape
N,d=8192,128
sc=np.ones(d); sc[rng.choice(d,6,replace=False)]=12
mu=rng.standard_normal(d)*2
K=(rng.standard_normal((N,d))+mu)*sc; V=rng.standard_normal((N,d))
Hm=np.array([[1.]])
while Hm.shape[0]<d: Hm=np.block([[Hm,Hm],[Hm,-Hm]])
R=(Hm/np.sqrt(d))*rng.choice([-1,1],d)  # randomized Hadamard
kurt=lambda x:np.mean(((x-x.mean())/x.std())**4)
Kc=K-K.mean(0)   # mean removal: q·mu is constant over tokens -> softmax-invariant
U=(Kc@R); g=np.linalg.norm(U,axis=1,keepdims=True); U/=g
print(f"\nkurtosis raw keys {kurt(K):.1f} -> rotated unit shape {kurt(U*np.sqrt(d)):.2f} (Gaussian=3)")
def q_shape(U,bnd,lev): z=U*np.sqrt(d); idx=np.searchsorted(bnd,z); return idx,lev[idx]/np.sqrt(d)
def gs_quant(X,R,fbnd,flev,gbits,center):
    m=X.mean(0) if center else 0; Y=(X-m)@R; g=np.linalg.norm(Y,axis=1,keepdims=True); Uy=Y/g
    lg=np.log(g); lo,hi=lg.min(),lg.max(); n=2**gbits
    gq=np.exp(lo+(np.clip(np.floor((lg-lo)/(hi-lo+1e-9)*n),0,n-1)+.5)/n*(hi-lo))
    idx,_=q_shape(Uy,fbnd,flev); return idx,gq,m
e=np.concatenate([[-np.inf],fb,[np.inf]]); cbnd=fb[3::4]
ce=np.concatenate([[-np.inf],cbnd,[np.inf]]); clev=np.array([cmean(ce[i],ce[i+1]) for i in range(4)])
def recon(idx,gq,m,R,coarse): lev=clev[idx>>2] if coarse else fl[idx]; return (gq*lev/np.sqrt(d))@R.T+m
def softmax(s): x=np.exp(s-s.max()); return x/x.sum()
def mkq(peak):
    i=rng.choice(N,8,replace=False); q=K[i].mean(0)-K.mean(0); return q/np.linalg.norm(q)*peak+rng.standard_normal(d)*.3
print("\nattention output rel. error (20 queries, peak=6), gain = 8-bit log")
ik,gk,mk=gs_quant(K,R,fb,fl,8,True); iv,gv,mv=gs_quant(V,R,fb,fl,8,True)
Kf=recon(ik,gk,mk,R,False); Kc_=recon(ik,gk,mk,R,True); Vf=recon(iv,gv,mv,R,False); Vc=recon(iv,gv,mv,R,True)
for kf in [0.02,0.05]:
    res=[]
    for _ in range(20):
        q=mkq(6); a=softmax(K@q/np.sqrt(d)); o=a@V; k=int(kf*N)
        for sel in ["2bit","sign"]:
            if sel=="2bit": ts=Kc_@q
            else: ts=(gk[:,0]*np.sign(((K-mk)@R)) .dot(R.T@q))  # 1-bit sign plane x exact gain
            hot=np.argpartition(-ts,k)[:k]; Kh=Kc_.copy(); Vh=Vc.copy(); Kh[hot]=Kf[hot]; Vh[hot]=Vf[hot]
            res.append((sel,np.linalg.norm(softmax(Kh@q/np.sqrt(d))@Vh-o)/np.linalg.norm(o),a[hot].sum()))
        r=lambda X,Y:np.linalg.norm(softmax(X@q/np.sqrt(d))@Y-o)/np.linalg.norm(o)
        res.append(("uni2",r(Kc_,Vc),0)); res.append(("uni4",r(Kf,Vf),0))
    for s in ["2bit","sign","uni2","uni4"]:
        rr=[x for x in res if x[0]==s]; print(f" k={kf*100:.0f}% {s:5s} err {np.mean([x[1] for x in rr]):.4f}"+(f"  m_H {np.mean([x[2] for x in rr]):.3f}" if s in("2bit","sign") else ""))
print("\ngain bits: logit error from gain alone (max |Δs| over tokens, avg over queries)")
for gb in [4,6,8]:
    _,gq,_=gs_quant(K,R,fb,fl,gb,True); gt=np.linalg.norm((K-K.mean(0))@R,axis=1)
    errs=[np.max(np.abs((gq[:,0]/gt-1)*((K-K.mean(0))@mkq(6))/np.sqrt(d))) for _ in range(10)]
    print(f" {gb}-bit log gain: max rel gain err {np.max(np.abs(gq[:,0]/gt-1))*100:.2f}%  max logit err {np.mean(errs):.3f}")
