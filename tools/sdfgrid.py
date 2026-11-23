"""sdfgrid.py — точное знаковое расстояние до сетки на регулярной решётке (узкая полоса точно, дальше EDT),
перестройка сетки marching cubes. Размеры в мм. Нужны numpy/scipy/skimage и бинарники rayz, sdfband рядом."""
import numpy as np,subprocess,os,tempfile
from scipy import ndimage as ndi
HERE=os.path.dirname(os.path.abspath(__file__))
class Grid:
    def __init__(s,lo,hi,h):
        s.h=h; s.lo=np.asarray(lo,float); s.n=(np.ceil((np.asarray(hi)-s.lo)/h).astype(int)+1)
    @property
    def shape(s): return tuple(int(x) for x in s.n)
    def coords(s,axis): return s.lo[axis]+np.arange(s.n[axis])*s.h
    def idx(s,P): return np.round((np.asarray(P)-s.lo)/s.h).astype(int)
def occupancy(V,F,G):
    """bool[NX,NY,NZ] по числу обмотки вертикальных лучей (перекрытия оболочек = объединение)."""
    td=tempfile.mkdtemp(); NX,NY,NZ=G.shape
    V.astype(np.float64).tofile(td+'/v.bin'); F.astype(np.int32).tofile(td+'/f.bin')
    subprocess.run([HERE+'/rayz',td+'/v.bin',td+'/f.bin',str(len(V)),str(len(F)),str(G.lo[0]),str(G.lo[1]),str(G.h),str(NX),str(NY),td+'/h.bin'],check=True)
    r=np.fromfile(td+'/h.bin',dtype=np.int32).reshape(-1,4)
    col=r[:,0]; z=r[:,2].view(np.float32).astype(float); w=-r[:,3].view(np.float32).astype(int)
    o=np.argsort(z,kind='stable'); col,z,w=col[o],z[o],w[o]
    zs=G.coords(2); occ=np.zeros((NX*NY,NZ),bool); acc=np.zeros(NX*NY,int); j=0
    for k,zk in enumerate(zs):
        j2=np.searchsorted(z,zk)
        if j2>j: np.add.at(acc,col[j:j2],w[j:j2]); j=j2
        occ[:,k]=acc>0
    return occ.reshape(NX,NY,NZ)
def band(V,F,G,B):
    td=tempfile.mkdtemp(); NX,NY,NZ=G.shape
    V.astype(np.float64).tofile(td+'/v.bin'); F.astype(np.int32).tofile(td+'/f.bin')
    subprocess.run([HERE+'/sdfband',td+'/v.bin',td+'/f.bin',str(len(V)),str(len(F)),*map(str,G.lo),str(G.h),str(NX),str(NY),str(NZ),str(B),td+'/d.bin'],check=True)
    return np.fromfile(td+'/d.bin',dtype=np.float32).reshape(NX,NY,NZ)
def sdf(V,F,G,band_vox=3,far=True):
    """знаковое расстояние (мм), внутри < 0. float32."""
    occ=occupancy(V,F,G); B=band_vox*G.h; d=band(V,F,G,B)
    if far:
        out=ndi.distance_transform_edt(~occ).astype(np.float32)*G.h-0.5*G.h
        ins=ndi.distance_transform_edt(occ).astype(np.float32)*G.h-0.5*G.h
        far_d=np.where(occ,-ins,out); del out,ins
        inb=d<B; d=np.where(inb,np.where(occ,-d,d),far_d); del far_d
    else:
        d=np.where(occ,-d,d)
    return d.astype(np.float32),occ
def sdf_from_occ(occ,h):
    out=ndi.distance_transform_edt(~occ).astype(np.float32)*h-0.5*h
    ins=ndi.distance_transform_edt(occ).astype(np.float32)*h-0.5*h
    return np.where(occ,-ins,out).astype(np.float32)
def smin(a,b,k):
    """гладкий минимум (полиномиальный, k в мм): объединение с плавным переходом радиуса ~k."""
    hh=np.clip(0.5+0.5*(b-a)/k,0,1); return (b*(1-hh)+a*hh-k*hh*(1-hh)).astype(np.float32)
def mesh(d,G,level=0.0):
    from skimage.measure import marching_cubes
    v,f,_,_=marching_cubes(d,level,spacing=(G.h,)*3,allow_degenerate=False)
    v=v+G.lo; f=f[:,[0,2,1]]
    T=v[f]; vol=np.einsum('ij,ij->i',T[:,0],np.cross(T[:,1],T[:,2])).sum()/6
    if vol<0: f=f[:,[0,2,1]]
    return v,f
def decimate(V,F,target):
    td=tempfile.mkdtemp()
    V.astype(np.float64).tofile(td+'/v.bin'); F.astype(np.int32).tofile(td+'/f.bin')
    r=subprocess.run([HERE+'/qem',td+'/v.bin',td+'/f.bin',str(len(V)),str(len(F)),str(target),td+'/vo.bin',td+'/fo.bin'],check=True,capture_output=True,text=True)
    return np.fromfile(td+'/vo.bin').reshape(-1,3),np.fromfile(td+'/fo.bin',dtype=np.int32).reshape(-1,3).astype(np.int64)
def check_closed(F):
    from collections import Counter
    E=np.vstack([F[:,[0,1]],F[:,[1,2]],F[:,[2,0]]]); c=Counter(map(tuple,E.tolist()))
    return sum(1 for e in c if c[e]!=1 or (e[1],e[0]) not in c)

def taubin(V,F,mask,iters=10,lam=0.5,mu=-0.53):
    """Сглаживание Таубина (без усадки) только для вершин mask: убирает «ступеньки» на подсыпке."""
    from scipy import sparse
    n=len(V); I=np.r_[F[:,0],F[:,1],F[:,2],F[:,1],F[:,2],F[:,0]]; J=np.r_[F[:,1],F[:,2],F[:,0],F[:,0],F[:,1],F[:,2]]
    A=sparse.coo_matrix((np.ones(len(I)),(I,J)),shape=(n,n)).tocsr(); A.data[:]=1
    deg=np.asarray(A.sum(1)).ravel(); W=sparse.diags(1/np.maximum(deg,1))@A
    V=V.copy(); m=mask[:,None]
    for _ in range(iters):
        for f in (lam,mu):
            V=np.where(m,V+f*(W@V-V),V)
    return V

def largest_shell(V,F):
    """Оставить самую большую оболочку (крошки после marching cubes/QEM — прочь)."""
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    n=len(V); A=sparse.coo_matrix((np.ones(len(F)*3),(np.r_[F[:,0],F[:,1],F[:,2]],np.r_[F[:,1],F[:,2],F[:,0]])),shape=(n,n))
    fc=connected_components(A,directed=False)[1][F[:,0]]; keep=F[fc==np.bincount(fc).argmax()]
    u=np.unique(keep); mp=-np.ones(n,int); mp[u]=np.arange(len(u)); return V[u],mp[keep]
