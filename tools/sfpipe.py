import pickle,numpy as np,time,sys
sys.path.insert(0,'/home/claude/tillo-wb-photos/tools')
from scipy.ndimage import map_coordinates
from sdfgrid import Grid,sdf,mesh,decimate,taubin,largest_shell,check_closed
from supportfree import cone_fill,clean_added,blend,check,report
from paintsmooth import laplacian,vertex_scores,smooth_scores,transfer_scores,repaint,boundary_stats
S=sys.argv[5]
theta=float(sys.argv[1]); k=float(sys.argv[2]); target=int(sys.argv[3]); out=sys.argv[4]
Vfig,Ffig=pickle.load(open(S+'/mm.pkl','rb')); Lfig=np.load(S+'/L2.npy').copy()
# enlarge nostrils on dense mesh
import os; comps=pickle.load(open(S+'/blackcomps.pkl','rb')) if os.path.exists(S+'/blackcomps.pkl') else []
cen=Vfig[Ffig].mean(1)
for ar,c,ext in comps:
    if ar<1: Lfig[np.linalg.norm(cen-c,axis=1)<0.47]=3
log=[]; t=time.time()
h=max(0.05,round(np.ptp(Vfig[:,2])/650,3))
G=Grid(Vfig.min(0)-0.8,Vfig.max(0)+0.8,h)
dO,occ=sdf(Vfig,Ffig,G,far=True)
x,y=G.coords(0),G.coords(1); disk=np.hypot(x[:,None],y[None,:])<=50
_,A=cone_fill(occ,G,theta,floor_z=0.2,floor_mask=disk,log=log)
A,comps2,_=clean_added(A,min_vox=300)
log.append('подсыпка %.2f мм³ (%.2f%%)'%(A.sum()*h**3,100*A.sum()/max(occ.sum(),1)))
fld=blend(dO,A,G,k=k); v,f=mesh(fld,G); del fld
v=taubin(v,f,map_coordinates(dO,((v-G.lo)/h).T,order=1)>0.004,iters=8)
v,f=decimate(v,f,target); v,f=largest_shell(v,f)
dv=map_coordinates(dO,((v-G.lo)/h).T,order=1)
K=int(Lfig.max())+1; Lap,M,_,_,_=laplacian(Vfig,Ffig)
So=smooth_scores(vertex_scores(Vfig,Ffig,Lfig,K),Lap,M,0.12)
Sc=transfer_scores(Vfig,Ffig,So,v,dist_out=dv)
fillv=(dv>0.06)&(v[:,2]>v[:,2].min()+1.0); Sc[fillv]=0; Sc[fillv,0]=1      # подсыпка = цвет тела
V2,F2,L2,plog=repaint(v,f,Sc,sigma=0.1,min_area=0.3); log+=plog
V2[:,2]=np.maximum(V2[:,2],0.2); V2[:,2]-=0.2
b=boundary_stats(V2,F2,L2)
log.append('граней %d, открытых %d, пила %.3f, t %.0f'%(len(F2),check_closed(F2),b['jag'],time.time()-t))
R=check(V2,F2,floor_z=V2[:,2].min(),floor_r=50); log.append(report(R,'после'))
print('\n'.join(log)); pickle.dump((V2,F2,L2),open(S+'/'+out,'wb'))
