import gzip, numpy as np
from scipy.stats import spearmanr
NB_=193; BIN=1_000_000; OFF=3_000_000; CHROM="chr1"
d={}
with open("scratch/spec3/joint.3dg") as f:
    for line in f:
        if line[0]=="#": continue
        a=line.split()
        if len(a)<5: continue
        d.setdefault(a[0],{})[int(a[1])]=np.array([float(x) for x in a[2:5]])
def grid(key):
    st=np.array(sorted(d[key])); pos=np.array([d[key][p] for p in st])
    out=np.full((NB_,3),np.nan)
    for i in range(NB_):
        j=np.searchsorted(st,OFF+i*BIN,side="right")-1
        if j>=0: out[i]=pos[j]
    return out
Gz,Ga,Gb=grid("c1c"),grid("c1a"),grid("c1b")
Gmid=(Ga+Gb)/2.0
iu=np.triu_indices(NB_,1)
sp=lambda a,b: spearmanr(a,b)[0]
dm=lambda G: np.linalg.norm(G[:,None,:]-G[None,:,:],axis=-1)
DA,DB=dm(Ga),dm(Gb)
print("truth contrast 1-rho(DA,DB) = %+.4f" % (1-sp(DA[iu],DB[iu])))

def construct(G0,u,t):
    """dA = |v + t*du|，dB = |v - t*du|，其中 v 来自 G0，du = u_i - u_j。"""
    Vt=G0[:,None,:]-G0[None,:,:]
    Ut=u[:,None,:]-u[None,:,:]
    dA=np.linalg.norm(Vt+t*Ut,axis=-1); dB=np.linalg.norm(Vt-t*Ut,axis=-1)
    return 1-sp(dA[iu],dB[iu])

umid=(Ga-Gb)/2.0
print()
print("A) Z = (Ga+Gb)/2 (the exact midpoint), u = (Ga-Gb)/2:")
for t in [1.0]:
    print("   t=%.1f contrast = %+.4f   (must equal the truth value above)" % (t, construct(Gmid,umid,t)))
print()
print("B) Z = the FDG consensus c1c, u = (Ga-Gb)/2:")
for t in [1.0]:
    print("   t=%.1f contrast = %+.4f" % (t, construct(Gz,umid,t)))
print()
print("C) Z = (Ga+Gb)/2, random u of the same magnitude:")
rng=np.random.default_rng(0)
ur=rng.normal(size=(NB_,3)); ur*=np.sqrt((umid**2).mean())/np.sqrt((ur**2).mean())
for t in [0.5,1.0]:
    print("   t=%.1f contrast = %+.4f" % (t, construct(Gmid,ur,t)))
print()
print("D) how far is the FDG consensus from the true midpoint? (frame-free)")
print("   rho(DZ, Dmid) = %+.4f ; Rg(Z)=%.3f Rg(mid)=%.3f" % (
    sp(dm(Gz)[iu], dm(Gmid)[iu]),
    np.sqrt(((Gz-Gz.mean(0))**2).sum(1).mean()), np.sqrt(((Gmid-Gmid.mean(0))**2).sum(1).mean())))
