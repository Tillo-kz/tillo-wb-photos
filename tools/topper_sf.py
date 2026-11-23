"""topper_sf.py — фигурка без поддержек за один вызов: подсыпка + перестройка сетки + ровная покраска."""
import numpy as np
from scipy.ndimage import map_coordinates
from sdfgrid import Grid,sdf,mesh,decimate,taubin,largest_shell,check_closed
from supportfree import cone_fill,clean_added,blend,check,report
from paintsmooth import laplacian,vertex_scores,smooth_scores,transfer_scores,repaint,boundary_stats

def support_free(Vfig,Ffig,Lfig,floor_z=3.4617,floor_r=7.05,theta=55.0,k=0.35,target=130000,h=None,log=None):
    """Vfig,Ffig,Lfig: фигурка БЕЗ крепления, мм печати, низ утоплен в площадку (z от 2.96), метки граней 0..K-1
    (лучше плотная сетка Hunyuan после autofix). floor_z/floor_r: верх и радиус площадки крепления.
    target: граней до разреза цветов (многоцветный топпер ~130 тыс.; 85 тыс. проверено — рвутся глаза).
    Возвращает (V,F,L) готовой фигурки: самонесущая, ≤ target граней до разреза цветов, ровная покраска."""
    log=[] if log is None else log
    H=np.ptp(Vfig[:,2]); h=h or max(0.05,round(H/650,3))          # шаг решётки: 0.05 мм для топперов до ~32 мм
    G=Grid(Vfig.min(0)-0.8,Vfig.max(0)+0.8,h)
    dO,occ=sdf(Vfig,Ffig,G,far=True)
    x,y=G.coords(0),G.coords(1); disk=np.hypot(x[:,None],y[None,:])<=floor_r-0.05
    _,A=cone_fill(occ,G,theta,floor_z=floor_z,floor_mask=disk,log=log)
    A,comps,_=clean_added(A,min_vox=300)
    log.append('подсыпка %.2f мм³ (%.2f%% объёма), кусков %d'%(A.sum()*h**3,100*A.sum()/max(occ.sum(),1),len(comps)))
    fld=blend(dO,A,G,k=k); v,f=mesh(fld,G); del fld
    v=taubin(v,f,map_coordinates(dO,((v-G.lo)/h).T,order=1)>0.004,iters=8)    # гладим только подсыпку
    v,f=decimate(v,f,target); v,f=largest_shell(v,f)
    dv=map_coordinates(dO,((v-G.lo)/h).T,order=1); del dO,occ
    K=int(Lfig.max())+1; Lap,M,_,_,_=laplacian(Vfig,Ffig)
    So=smooth_scores(vertex_scores(Vfig,Ffig,Lfig,K),Lap,M,0.12)
    S=transfer_scores(Vfig,Ffig,So,v,dist_out=dv)                          # подсыпка = цвет поверхности под ней
    V2,F2,L2,plog=repaint(v,f,S,sigma=0.1,min_area=0.3); log+=plog
    b=boundary_stats(V2,F2,L2)
    log.append('граней %d, открытых рёбер %d, границы цвета %.0f мм, пила %.3f'%(len(F2),check_closed(F2),b['length'],b['jag']))
    return V2,F2,L2
