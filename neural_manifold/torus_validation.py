import numpy as np, time
from ripser import ripser
from numpy.linalg import svd
rng = np.random.default_rng(0)

def pca(X, dim=6, maxpts=500):
    if len(X) > maxpts: X = X[rng.choice(len(X), maxpts, replace=False)]
    Xc = X - X.mean(0); U, S, _ = svd(Xc, full_matrices=False)
    return U[:, :dim] * S[:dim]
def bars(Z, thresh=np.inf):
    d = ripser(Z, maxdim=2, thresh=thresh)['dgms']
    out = []
    for k in (1, 2):
        L = np.sort((d[k][:, 1] - d[k][:, 0]))[::-1]
        out.append(np.round(L[:4], 3))
    return out

# (A) direct torus
n=800; a=rng.random(n)*2*np.pi; b=rng.random(n)*2*np.pi
T3=np.stack([(2+0.8*np.cos(b))*np.cos(a),(2+0.8*np.cos(b))*np.sin(a),0.8*np.sin(b)],1)+rng.normal(0,0.015,(n,3))
t0=time.time(); h1,h2=bars(T3); print(f"(A) direct torus: H1={h1} H2={h2} ({time.time()-t0:.1f}s)")

# (B) grid rate manifold
N=200; ks=np.array([[np.cos(a2),np.sin(a2)] for a2 in [0,np.pi/3,2*np.pi/3]])
lam=0.30; kf=2*np.pi/lam; phases=rng.random((N,2)); pos=rng.random((3000,2))*1.5
def gr(pos,p): g=sum(np.cos(kf*((pos-p*lam)@k)) for k in ks); return np.clip((g+1.5)/4.5,0,None)
R=np.stack([gr(pos,phases[i]) for i in range(N)],1)
Z=pca(R,6,500)
t0=time.time(); h1,h2=bars(Z); print(f"(B) grid rates:   H1={h1} H2={h2} ({time.time()-t0:.1f}s)")
