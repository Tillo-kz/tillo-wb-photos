"""supportfree.py — «без поддержек»: карта нависаний и мягкая самонесущая подсыпка.
Все размеры в мм печатного масштаба. Использует sdfgrid.py."""
import numpy as np
from scipy import ndimage as ndi
from sdfgrid import Grid,sdf_from_occ,smin,HERE

def slice_sdf2d(m,h):
    """2D знаковое расстояние в плоскости слоя (мм), внутри > 0."""
    if not m.any(): return np.full(m.shape,-1e3,np.float32)
    if m.all(): return np.full(m.shape,1e3,np.float32)
    return (ndi.distance_transform_edt(m)*h-ndi.distance_transform_edt(~m)*h).astype(np.float32)

def _bridges(N,cov,base_dist,h,bridge_max,min_width=0.6,min_sep=0.8):
    """Компоненты N, которые слайсер перекинет МОСТОМ: широкий «потолок» (>=min_width мм), держится за опору
    этого же слоя минимум с двух разных сторон, до опоры снизу не дальше bridge_max мм. Тонкая кромка (подол,
    край капюшона) мостом НЕ считается: её печатает висящий в воздухе периметр."""
    lab,n=ndi.label(N,structure=np.ones((3,3))); out=np.zeros_like(N)
    if n==0: return out,N
    inner=ndi.distance_transform_edt(N)
    for i,sl in enumerate(ndi.find_objects(lab)):
        c=lab[sl]==i+1
        if base_dist[sl][c].max()*h>bridge_max: continue
        if 2*inner[sl][c].max()*h<min_width: continue
        sl2=tuple(slice(max(0,q.start-3),q.stop+3) for q in sl)
        cc=lab[sl2]==i+1
        anc=ndi.binary_dilation(cc,structure=np.ones((3,3)),iterations=2)&cov[sl2]
        al,an=ndi.label(anc,structure=np.ones((3,3)))
        if an<2: continue
        cs=np.array(ndi.center_of_mass(anc,al,range(1,an+1)))
        if (np.ptp(cs[:,0])+np.ptp(cs[:,1]))*h>=min_sep: out[sl2]|=cc
    return out,N&~out

def cone_fill(occ,G,theta_deg=55.0,floor_z=None,floor_mask=None,zone=None,gap_close=0.6,tol=0.15,bridge_max=0.6,max_strut=2.5,log=None):
    """Самонесущая подсыпка сверху вниз, угол theta от вертикали. Для каждого слоя k (снизу опора O_k):
    N = части слоя выше, которые дальше h*tan(theta) от опоры. Из N выкидываем «мостики» (широкий потолок между
    двумя опорами, пролёт до 2*bridge_max: щель между ногами и т.п.). Под остальным:
      1) толстое: конус Phi_k = max(Phi_O, Phi_{k+1} - h*tan(theta)) только под непокрытыми местами;
      2) тонкое (кромка подола, край капюшона): зазор между кромкой и телом закрывается «замыканием» радиуса
         gap_close, дальше вниз его подрезает тот же конус -> клин под углом theta (подол прижат к телу снизу).
         tol: выступы меньше tol мм не трогаем (их печатает слайсер).
    floor_z/floor_mask: уровень и 2D-маска опоры (верх площадки крепления). zone: где подсыпать можно.
      3) кончик конуса, повисший в воздухе (чашка в лапах, подбородок): «стекает» к ближайшему телу под углом theta,
         промежуток заливается. Стекание длиннее max_strut мм пишется в log: это проблема позы (лучше чинить вид)."""
    NX,NY,NZ=occ.shape; h=G.h; rpx=np.tan(np.radians(theta_deg)); r=h*rpx
    zs=G.coords(2); kfloor=0 if floor_z is None else int(np.searchsorted(zs,floor_z))
    S=occ.copy(); phi=slice_sdf2d(occ[:,:,NZ-1],h); prev=occ[:,:,NZ-1].copy(); dangling=0.0; nbr=0
    for k in range(NZ-2,kfloor-1,-1):
        base=occ[:,:,k].copy()
        if k==kfloor and floor_mask is not None: base|=floor_mask
        if not prev.any():
            phi=slice_sdf2d(base,h); S[:,:,k]=base; prev=base; continue
        bd=ndi.distance_transform_edt(~base) if base.any() else np.full(base.shape,1e6)
        N=prev&(bd>rpx)
        if N.any():
            br,N=_bridges(N,prev&~N,bd,h,bridge_max); nbr+=int(br.any())
        po=slice_sdf2d(base,h)
        if N.any():
            allow=ndi.binary_dilation(N,structure=np.ones((3,3)),iterations=int(np.ceil(rpx))+1)
            if zone is not None: allow&=zone[:,:,k]
            phi=np.maximum(po,np.where(allow,phi-r,-1e3))
        else: phi=po
        cur=(phi>0)|base
        if N.any():
            dist=ndi.distance_transform_edt(~cur)
            Uall=N&(dist>rpx)&(bd*h>tol)
            U=Uall&occ[:,:,k+1]              # тонкая кромка исходной модели
            Uf=Uall&~occ[:,:,k+1]            # кончик конуса подсыпки, не дотянувшийся до тела
            if U.any():
                # кромка (подол, край капюшона): закрываем зазор до опоры «замыканием» радиуса gap_close ->
                # на этом слое появляется пол под кромкой, дальше его подрезает конус -> клин под углом theta
                Rp=max(2,int(round(gap_close/h)))
                yy,xx=np.mgrid[-Rp:Rp+1,-Rp:Rp+1]; disk=(xx*xx+yy*yy)<=Rp*Rp
                C=ndi.binary_closing(U|cur,structure=disk)&ndi.binary_dilation(U,structure=disk)&~cur
                if zone is not None: C&=zone[:,:,k]
                cur|=C
                left=U&(ndi.distance_transform_edt(~cur)>rpx)
                Uf|=left
            if Uf.any():
                # «стекание» к ближайшему телу под углом theta: точка сдвигается к опоре на h*tan(theta),
                # промежуток до опоры заливается (клин, а не столб)
                dist,ind=ndi.distance_transform_edt(~cur,return_indices=True)
                Uf&=dist>rpx
                if Uf.any():
                    pi,pj=np.where(Uf); bi,bj=ind[0][pi,pj],ind[1][pi,pj]; d=dist[pi,pj]
                    step=np.minimum(rpx*0.9,d); qi=pi+(bi-pi)*step/d; qj=pj+(bj-pj)*step/d
                    for t in np.linspace(0,1,int(np.ceil(d.max()))+2):
                        ii=np.round(qi+(bi-qi)*t).astype(int); jj=np.round(qj+(bj-qj)*t).astype(int)
                        if zone is not None: ok=zone[ii,jj,k]; ii,jj=ii[ok],jj[ok]
                        cur[ii,jj]=True
                    dangling=max(dangling,float(d.max()*h))
            if U.any() or Uf.any(): phi=np.maximum(phi,slice_sdf2d(cur,h))
        S[:,:,k]=cur|occ[:,:,k]; prev=S[:,:,k]
    if log is not None:
        log.append('slices with bridges left to slicer: %d'%nbr)
        log.append('самое длинное «стекание» к телу: %.2f мм%s'%(dangling,'  <-- далеко от тела: лучше исправить позу/вид' if dangling>max_strut else ''))
    A=S&~occ
    if floor_z is not None: A[:,:,:kfloor]=False
    return S,A

def clean_added(A,min_vox=300,open_iter=0):
    """Убираем мелочь (шкурку в 1 воксель по вогнутостям, крошки): оставляем настоящие подсыпки.
    Возвращает (маска, список компонент, метки)."""
    B=ndi.binary_opening(A,iterations=open_iter) if open_iter>0 else A.copy()
    lab,n=ndi.label(B,structure=np.ones((3,3,3)))
    if n==0: return B,[],lab
    sz=np.asarray(ndi.sum(B,lab,range(1,n+1))); keep=np.zeros(n+1,bool); keep[1:]=sz>=min_vox
    B=keep[lab]; lab,n=ndi.label(B,structure=np.ones((3,3,3)))
    comps=[dict(id=i+1,vox=int((lab[sl]==i+1).sum()),slices=sl) for i,sl in enumerate(ndi.find_objects(lab))]
    return B,comps,lab

def blend(dO,A,G,k=0.35,sigma_vox=2.0,close_mm=0.4):
    """Гладкое объединение исходника и подсыпки (радиус перехода k мм).
    Каждый кусок подсыпки сначала «замыкаем» шаром close_mm: соседние столбики/капли сливаются в одну
    перемычку (как шея или ткань), потом сглаживаем (sigma_vox) — без ступенек и «сосулек».
    Вдали от подсыпки поле = исходное (детали не трогаем)."""
    h=G.h; r=max(1,int(round(close_mm/h)))
    if r>0 and A.any():
        zz,yy,xx=np.mgrid[-r:r+1,-r:r+1,-r:r+1]; ball=(xx*xx+yy*yy+zz*zz)<=r*r
        lab,n=ndi.label(ndi.binary_dilation(A,iterations=r),structure=np.ones((3,3,3)))   # близкие куски — вместе
        out=A.copy()
        for sl in ndi.find_objects(lab):
            sl=tuple(slice(max(0,q.start-r-1),min(A.shape[i],q.stop+r+1)) for i,q in enumerate(sl))
            out[sl]|=ndi.binary_closing(np.pad(A[sl],r+1),structure=ball)[r+1:-r-1,r+1:-r-1,r+1:-r-1]
        A=out
    dA=sdf_from_occ(A,h)
    if sigma_vox>0: dA=ndi.gaussian_filter(dA,sigma_vox).astype(np.float32)
    return smin(dO,dA,k)


def check(V,F,layer=0.2,h=0.04,ok=0.25,slow=0.42,floor_z=None,floor_r=None,bridge_max=0.6):
    """Проверка «как слайсер»: слои layer мм, шаг контура слоя за предыдущий слой.
    air  = выступ > slow мм (0.42 = ширина линии): висит в воздухе, без поддержки будет брак;
    slow = выступ ok..slow: печатается медленно, может слегка провиснуть (допустимо на мелких местах);
    bridge = широкий «потолок» между двумя опорами с пролётом до 2*bridge_max: слайсер кладёт мост (норма);
    island = новый кусок слоя, вообще не касающийся слоя ниже (кроме стола/площадки) — недопустимо.
    floor_z/floor_r: площадка крепления (z верха и радиус) — опора. Возвращает dict со строками по слоям и
    3D-точками проблем (для подсветки граней)."""
    lo=V.min(0)-0.5; hi=V.max(0)+0.5
    G2=Grid(lo,hi,h); zc=np.arange(V[:,2].min()+layer/2,V[:,2].max(),layer)   # срезы в серединах слоёв
    occ=_occ_at(V,F,G2,zc)
    x=G2.coords(0); y=G2.coords(1); rows=[]; pts=[]
    disk=None if floor_r is None else (np.hypot(x[:,None],y[None,:])<=floor_r)
    for k in range(1,len(zc)):
        cur=occ[:,:,k]; below=occ[:,:,k-1].copy()
        if floor_z is not None and zc[k-1]<=floor_z+layer and disk is not None: below|=disk
        if not cur.any(): continue
        d=ndi.distance_transform_edt(~below)*h if below.any() else np.full(cur.shape,1e6)
        N=cur&(d>slow)
        br=np.zeros_like(N)
        if N.any(): br,N=_bridges(N,cur&~N,d/h,h,bridge_max)
        lab,n=ndi.label(cur,structure=np.ones((3,3))); isl=0; isl_a=0.0
        if n:
            t=np.asarray(ndi.maximum(below,lab,range(1,n+1)))
            for i in np.where(t==0)[0]:
                isl+=1; isl_a+=(lab==i+1).sum()*h*h; N|=lab==i+1
        rows.append(dict(z=zc[k],air=N.sum()*h*h,slow=(cur&(d>ok)&(d<=slow)).sum()*h*h,bridge=br.sum()*h*h,islands=isl,island_area=isl_a))
        ii,jj=np.where(N)
        if len(ii): pts.append(np.c_[x[ii],y[jj],np.full(len(ii),zc[k])])
    P=np.vstack(pts) if pts else np.zeros((0,3))
    return dict(rows=rows,points=P,
                air=sum(r['air'] for r in rows),slow=sum(r['slow'] for r in rows),bridge=sum(r['bridge'] for r in rows),
                islands=sum(r['islands'] for r in rows))

def _occ_at(V,F,G,zs):
    import subprocess,tempfile
    td=tempfile.mkdtemp(); NX,NY=int(G.n[0]),int(G.n[1])
    V.astype(np.float64).tofile(td+'/v.bin'); F.astype(np.int32).tofile(td+'/f.bin')
    subprocess.run([HERE+'/rayz',td+'/v.bin',td+'/f.bin',str(len(V)),str(len(F)),str(G.lo[0]),str(G.lo[1]),str(G.h),str(NX),str(NY),td+'/h.bin'],check=True)
    r=np.fromfile(td+'/h.bin',dtype=np.int32).reshape(-1,4)
    col=r[:,0]; z=r[:,2].view(np.float32).astype(float); w=-r[:,3].view(np.float32).astype(int)
    o=np.argsort(z,kind='stable'); col,z,w=col[o],z[o],w[o]
    occ=np.zeros((NX*NY,len(zs)),bool); acc=np.zeros(NX*NY,int); j=0
    for k,zk in enumerate(zs):
        j2=np.searchsorted(z,zk)
        if j2>j: np.add.at(acc,col[j:j2],w[j:j2]); j=j2
        occ[:,k]=acc>0
    return occ.reshape(NX,NY,len(zs))

def report(R,name=''):
    s='%s: в воздухе %.2f мм² | медленно %.2f мм² | мосты %.2f мм² | островки %d'%(name,R['air'],R['slow'],R['bridge'],R['islands'])
    top=sorted(R['rows'],key=lambda r:-(r['air']+r['island_area']))[:6]
    s+=''.join('\n   z=%.1f: воздух %.2f, островков %d'%(r['z'],r['air'],r['islands']) for r in top if r['air']>0.05)
    return s

def mark_faces(V,F,P,radius=0.35):
    """Грани рядом с проблемными точками (для подсветки на рендере)."""
    from scipy.spatial import cKDTree
    m=np.zeros(len(F),bool)
    if len(P)==0: return m
    C=V[F].mean(1); t=cKDTree(P); d,_=t.query(C,distance_upper_bound=radius); m[np.isfinite(d)]=True
    return m
