import pickle,numpy as np,sys,cv2
sys.path.insert(0,'/home/claude/tillo-wb-photos/tools'); sys.path.insert(0,sys.argv[1]+'/../horse')
from paintsmooth import vertex_scores,iso_cut
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from render import render
S=sys.argv[1]; grow=float(sys.argv[2])
V,F,L=pickle.load(open(S+'/sfB.pkl','rb'))
A=np.linalg.norm(np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]]),axis=1)/2
cen=V[F].mean(1)
# components of black faces via shared vertices
bf=np.where(L==2)[0]
E=np.vstack([F[bf][:,[0,1]],F[bf][:,[1,2]]])
G=sp.coo_matrix((np.ones(len(E)),(E[:,0],E[:,1])),shape=(len(V),len(V)))
n,lab=connected_components(G,directed=False)
fl=lab[F[bf,0]]
ells=[]
for c in np.unique(fl):
    m=bf[fl==c]; a=A[m].sum()
    if a<0.3: continue
    P=cen[m][:,[0,2]]; w=A[m]/a; mu=(P*w[:,None]).sum(0); D=P-mu; C=(D*w[:,None]).T@D
    C2=C*4*grow**2 if a>1.5 else np.diag([0.5**2,0.33**2]); mu2=mu if a>1.5 else np.array([mu[0],cen[m][:,2][np.argmin(cen[m][:,1])]]); ells.append((mu2,C2,cen[m][:,1].min() if a<=1.5 else cen[m][:,1].mean(),a))
    print('comp area %.2f center %s size %s'%(a,mu.round(2),(2*np.sqrt(np.linalg.eigvalsh(C))*2*grow).round(2)))
# vertex normals
fn=np.cross(V[F[:,1]]-V[F[:,0]],V[F[:,2]]-V[F[:,0]])
vn=np.zeros_like(V); 
for k in range(3): np.add.at(vn,F[:,k],fn)
vn/=np.linalg.norm(vn,axis=1,keepdims=True)+1e-12
Sc=vertex_scores(V,F,L,3)
blk=Sc[:,2]>0; Sc[:,2]=0
row=Sc.sum(1); z=row<1e-6
nose=[e for e in ells if e[3]<=1.5]
for e in nose:
    near=z&(np.hypot(V[:,0]-e[0][0],V[:,2]-e[0][1])<1.3)&(vn[:,1]<0)
    Sc[near,1]=1
z=Sc.sum(1)<1e-6; Sc[z,0]=1
Sc/=Sc.sum(1,keepdims=True)
b=np.zeros(len(V))
for mu,C,yc,a in ells:
    Ci=np.linalg.inv(C); D=V[:,[0,2]]-mu; r=np.sqrt(np.einsum('ij,jk,ik->i',D,Ci,D))
    front=(vn[:,1]<-0.1)&(np.abs(V[:,1]-yc)<1.0)
    bb=np.clip(0.5+(1-r)*4,0,1)*front
    b=np.maximum(b,bb)
Sc=Sc*(1-b)[:,None]; Sc[:,2]=b
V2,F2,L2,_=iso_cut(V,F,Sc)
print(np.bincount(L2))
pickle.dump((V2,F2,L2),open(S+'/sfC.pkl','wb'))
pal=np.array([[0,150,255],[250,250,250],[20,20,20]])
img=render(V2,F2,pal[L2],'front',900); cv2.imwrite(S+'/eyes.jpg',img[150:600,150:750])
