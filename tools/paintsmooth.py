"""paintsmooth.py — ровная покраска сетки по меткам граней.
Идея: метка по граням даёт «пилу» по рёбрам треугольников. Вместо этого:
 1) из меток граней делаем гладкое поле «вероятностей» цвета в вершинах (диффузия по поверхности,
    через резкие складки диффузия не идёт, поэтому граница прилипает к складкам);
 2) убираем мелкие островки и тонкие волоски (порог в мм² ПЕЧАТНОГО размера);
 3) РАЗРЕЗАЕМ треугольники по изолиниям поля: граница цвета = гладкая ломаная, каждый треугольник одного цвета.
Все размеры в мм печатного масштаба. Нужны numpy, scipy."""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components

def face_geom(V,F):
    p0,p1,p2=V[F[:,0]],V[F[:,1]],V[F[:,2]]
    n=np.cross(p1-p0,p2-p0); a=np.linalg.norm(n,axis=1)/2; n=n/(2*a[:,None]+1e-30)
    return n,a

def edge_faces(F):
    """уникальные рёбра (i<j) и две смежные грани для каждого (манифолд)."""
    nf=len(F); e=np.concatenate([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); fid=np.tile(np.arange(nf),3)
    a=np.minimum(e[:,0],e[:,1]); b=np.maximum(e[:,0],e[:,1]); key=a.astype(np.int64)*(F.max()+1)+b
    o=np.argsort(key,kind='stable'); key,fid,a,b=key[o],fid[o],a[o],b[o]
    s=np.r_[True,key[1:]!=key[:-1]]; idx=np.cumsum(s)-1
    E=np.c_[a[s],b[s]]; ef=-np.ones((len(E),2),int)
    first=np.where(s)[0]; ef[:,0]=fid[first]
    second=np.where(~s)[0]; ef[idx[second],1]=fid[second]
    return E,ef

def laplacian(V,F,crease_deg=25.0):
    """Котангенсный лапласиан (отрицательные веса обрезаны) с ослаблением через складки:
    вес *= exp(-(угол/crease_deg)^2). Возвращает (L, M) — жёсткость и массы вершин."""
    n,a=face_geom(V,F); N=len(V)
    I=[];J=[];W=[]
    for k in range(3):
        i,j,o=F[:,k],F[:,(k+1)%3],F[:,(k+2)%3]
        u=V[i]-V[o]; v=V[j]-V[o]
        cot=(u*v).sum(1)/(np.linalg.norm(np.cross(u,v),axis=1)+1e-30)
        I.append(i);J.append(j);W.append(0.5*np.clip(cot,0,10))
    I=np.concatenate(I);J=np.concatenate(J);W=np.concatenate(W)
    E,ef=edge_faces(F)
    ok=ef[:,1]>=0
    ang=np.zeros(len(E)); ang[ok]=np.degrees(np.arccos(np.clip((n[ef[ok,0]]*n[ef[ok,1]]).sum(1),-1,1)))
    a_=np.minimum(I,J);b_=np.maximum(I,J);key=a_.astype(np.int64)*(N)+b_
    ekey=E[:,0].astype(np.int64)*N+E[:,1]; pos=np.searchsorted(ekey,key)
    W=W*np.exp(-(ang[pos]/crease_deg)**2)
    Wm=sparse.coo_matrix((W,(a_,b_)),shape=(N,N)).tocsr(); Wm=Wm+Wm.T
    L=sparse.diags(np.asarray(Wm.sum(1)).ravel())-Wm
    m=np.zeros(N); 
    for k in range(3): np.add.at(m,F[:,k],a/3)
    m=np.maximum(m,1e-9*max(m.max(),1e-12))   # вершины без граней (крошки) не делают систему вырожденной
    return L.tocsc(),sparse.diags(m).tocsc(),ang,E,ef

def vertex_scores(V,F,L,K):
    _,a=face_geom(V,F); N=len(V); S=np.zeros((N,K))
    for k in range(3): np.add.at(S,(F[:,k],L),a)
    return S/(S.sum(1,keepdims=True)+1e-30)

def smooth_scores(S,Lap,M,sigma):
    """Неявная диффузия (M + t L) s = M s0, t = sigma^2/2 (sigma в мм печатного размера).
    До ~150 тыс. вершин — прямой решатель, больше — сопряжённые градиенты (память)."""
    t=sigma**2/2; A=(M+t*Lap).tocsc(); B=M@S
    if A.shape[0]<=150000:
        lu=splu(A); out=np.column_stack([lu.solve(B[:,k]) for k in range(S.shape[1])])
    else:
        from scipy.sparse.linalg import cg
        d=A.diagonal(); P=sparse.diags(1/d)
        out=np.column_stack([cg(A,B[:,k],x0=S[:,k],M=P,rtol=1e-6,maxiter=500)[0] for k in range(S.shape[1])])
    return np.clip(out,0,None)

def face_labels(F,S):
    return S[F].mean(1).argmax(1)

def remove_small(V,F,Lf,min_area,protect=None,iters=4):
    """Островки (компоненты одной метки по граням) меньше min_area мм² → в метку соседей.
    protect: bool по граням, такие островки не трогать (глаза и т.п.)."""
    _,a=face_geom(V,F); nf=len(F); E,ef=edge_faces(F); ok=ef[:,1]>=0; fa,fb=ef[ok,0],ef[ok,1]
    Lf=Lf.copy(); log=[]
    for _ in range(iters):
        s=Lf[fa]==Lf[fb]
        g=sparse.coo_matrix((np.ones(s.sum()),(fa[s],fb[s])),shape=(nf,nf))
        nc,cc=connected_components(g,directed=False); area=np.bincount(cc,weights=a,minlength=nc)
        small=area[cc]<min_area
        if protect is not None: small&=~protect
        if not small.any(): break
        d=Lf[fa]!=Lf[fb]; K=Lf.max()+1
        # для каждой малой компоненты: метка соседа с наибольшей общей длиной границы
        el=np.linalg.norm(V[E[ok][:,0]]-V[E[ok][:,1]],axis=1)
        h=np.zeros((nc,K)); 
        m=d&small[fa]; np.add.at(h,(cc[fa[m]],Lf[fb[m]]),el[m])
        m=d&small[fb]; np.add.at(h,(cc[fb[m]],Lf[fa[m]]),el[m])
        comps=np.unique(cc[small]); changed=0
        for c in comps:
            if h[c].sum()>0:
                Lf[cc==c]=h[c].argmax(); changed+=1
        log.append('merged %d islands'%changed)
        if changed==0: break
    return Lf,log

def iso_cut(V,F,S,snap=0.12):
    """Разрезает треугольники по изолиниям s_k = max_{j!=k} s_j. Возвращает V2,F2,метки граней,S2.
    snap: если точка пересечения ближе snap*длина_ребра к вершине — вершина «садится» на изолинию (без щепок)."""
    V=V.copy(); S=S.copy(); F=F.copy(); K=S.shape[1]
    for k in range(K):
        others=np.delete(S,k,axis=1).max(1); g=S[:,k]-others
        # снап вершин к нулю, чтобы не плодить щепки
        E,ef=edge_faces(F); ga,gb=g[E[:,0]],g[E[:,1]]
        cr=(ga*gb<0); t=ga[cr]/(ga[cr]-gb[cr])
        near0=np.zeros(len(V),bool); ea=E[cr]
        np.logical_or.at(near0,ea[t<snap,0],True); np.logical_or.at(near0,ea[t>1-snap,1],True)
        g[near0]=0.0
        # для вершин на нуле выравниваем значения: s_k = max others (чтобы следующие разрезы их понимали)
        if near0.any():
            idx=np.where(near0)[0]; mo=np.delete(S[idx],k,axis=1).max(1)
            avg=(S[idx,k]+mo)/2; S[idx,k]=avg
            o=np.delete(np.arange(K),k)[np.delete(S[idx],k,axis=1).argmax(1)]
            S[idx,o]=avg
        sgn=np.sign(np.where(np.abs(g)<1e-12,0,g)).astype(int)
        newV=[];newS=[];ek={}
        def cross(i,j):
            key=(min(i,j),max(i,j))
            if key not in ek:
                t=g[i]/(g[i]-g[j]); p=V[i]+t*(V[j]-V[i]); s=S[i]+t*(S[j]-S[i])
                # на изолинии: s_k = max others
                ek[key]=len(V)+len(newV); newV.append(p); newS.append(s)
            return ek[key]
        sF=sgn[F]; pos=(sF>0).sum(1); neg=(sF<0).sum(1)
        need=(pos>0)&(neg>0)
        out=[F[~need]]
        rows=[]
        for f in F[need]:
            s3=sgn[f]
            z=np.where(s3==0)[0]
            if len(z)==1:
                r=z[0]; a,b,c=f[r],f[(r+1)%3],f[(r+2)%3]; p=cross(b,c)
                rows+= [(a,b,p),(a,p,c)]
            else:
                # одна вершина на одной стороне, две на другой
                lone=[i for i in range(3) if (s3==s3[i]).sum()==1][0]
                a,b,c=f[lone],f[(lone+1)%3],f[(lone+2)%3]; p=cross(a,b); q=cross(a,c)
                rows.append((a,p,q))
                # четырёхугольник p,b,c,q: делим по короткой диагонали
                Vb=lambda i: V[i] if i<len(V) else newV[i-len(V)]
                if np.linalg.norm(Vb(p)-Vb(c))<np.linalg.norm(Vb(b)-Vb(q)): rows+=[(p,b,c),(p,c,q)]
                else: rows+=[(p,b,q),(b,c,q)]
        if newV:
            V=np.vstack([V,np.array(newV)]); S=np.vstack([S,np.array(newS)])
        F=np.vstack(out+[np.array(rows,int).reshape(-1,3)])
    # метка грани = argmax в центре
    Lf=S[F].mean(1).argmax(1)
    return V,F,Lf,S

def boundary_stats(V,F,Lf):
    """Длина границ цвета и «пилообразность»: отношение длины границы к длине сглаженной границы."""
    E,ef=edge_faces(F); ok=ef[:,1]>=0
    b=ok.copy(); b[ok]=Lf[ef[ok,0]]!=Lf[ef[ok,1]]
    BE=E[b]; L=np.linalg.norm(V[BE[:,0]]-V[BE[:,1]],axis=1).sum()
    # сглаженная версия: 10 итераций лапласа вдоль ломаной (вершины степени 2)
    N=len(V); deg=np.bincount(BE.ravel(),minlength=N)
    P=V.copy()
    nb=[[] for _ in range(N)]
    for x,y in BE: nb[x].append(y); nb[y].append(x)
    idx=np.where(deg==2)[0]; A=np.array([nb[i] for i in idx])
    for _ in range(10): P[idx]=0.5*P[idx]+0.25*(P[A[:,0]]+P[A[:,1]])
    Ls=np.linalg.norm(P[BE[:,0]]-P[BE[:,1]],axis=1).sum()
    return dict(length=L,smooth_length=Ls,jag=L/max(Ls,1e-9),edges=int(b.sum()))

def surface_samples(V,F,S,n=1500000,seed=0):
    """Равномерные точки на поверхности + интерполированные очки цвета + нормали граней."""
    rng=np.random.default_rng(seed); nrm,a=face_geom(V,F); p=a/a.sum()
    idx=rng.choice(len(F),n,p=p); w=rng.dirichlet([1,1,1],n)
    P=np.einsum('nk,nkd->nd',w,V[F[idx]]); Ss=np.einsum('nk,nkd->nd',w,S[F[idx]])
    return np.vstack([P,V]),np.vstack([Ss,S]),np.vstack([nrm[idx],np.zeros_like(V)])

def transfer_scores(Vsrc,Fsrc,Ssrc,Vdst,dist_out=None,eps=0.03,below=True):
    """Очки цвета на новую сетку. Точки на исходной поверхности (|d|<eps) берут цвет ближайшей точки.
    Точки подсыпки (вне исходника) берут цвет поверхности ПОД собой (опора: «тело поднимается навстречу»)."""
    from scipy.spatial import cKDTree
    P,Ss,Nn=surface_samples(Vsrc,Fsrc,Ssrc); t=cKDTree(P)
    d,nn=t.query(Vdst); out=Ss[nn].copy()
    if below and dist_out is not None:
        far=np.where(dist_out>eps)[0]
        if len(far):
            dd,ii=t.query(Vdst[far],k=64)
            for r,(row,v) in enumerate(zip(ii,Vdst[far])):
                cand=row[(P[row,2]<v[2]-0.02)&(Nn[row,2]>0.2)]
                if len(cand): out[far[r]]=Ss[cand[0]]
    return out

def repaint(V,F,S,sigma=0.1,min_area=0.3,crease_deg=25.0,protect=None):
    """Полный цикл ровной покраски на готовой сетке: сглаживание -> островки -> разрез по изолиниям."""
    Lap,M,_,_,_=laplacian(V,F,crease_deg); K=S.shape[1]
    S1=smooth_scores(S,Lap,M,sigma); Lf=face_labels(F,S1)
    Lf,log=remove_small(V,F,Lf,min_area,protect=protect)
    S2=smooth_scores(vertex_scores(V,F,Lf,K),Lap,M,sigma*0.7)
    V2,F2,L2,S3=iso_cut(V,F,S2)
    return V2,F2,L2,log
