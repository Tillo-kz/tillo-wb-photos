#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <queue>
#include <algorithm>
using namespace std;
struct Q{double a[10];Q(){for(int i=0;i<10;i++)a[i]=0;}
 void add(const Q&o){for(int i=0;i<10;i++)a[i]+=o.a[i];}
 double ev(double x,double y,double z)const{return a[0]*x*x+2*a[1]*x*y+2*a[2]*x*z+2*a[3]*x+a[4]*y*y+2*a[5]*y*z+2*a[6]*y+a[7]*z*z+2*a[8]*z+a[9];}};
int nV,nF;vector<double>V;vector<int>F;vector<char>fa;vector<vector<int>>vf;vector<Q>q;vector<int>ver;vector<char>va;
struct E{double c;int u,v,vu,vv;double p[3];bool operator<(const E&o)const{return c>o.c;}};
priority_queue<E>pq;
void fnorm(int a,int b,int c,const double*pa,const double*pb,const double*pc,double*n){double u[3],w[3];for(int i=0;i<3;i++){u[i]=pb[i]-pa[i];w[i]=pc[i]-pa[i];}n[0]=u[1]*w[2]-u[2]*w[1];n[1]=u[2]*w[0]-u[0]*w[2];n[2]=u[0]*w[1]-u[1]*w[0];}
bool solve(const Q&Qm,double*p){double A[3][3]={{Qm.a[0],Qm.a[1],Qm.a[2]},{Qm.a[1],Qm.a[4],Qm.a[5]},{Qm.a[2],Qm.a[5],Qm.a[7]}};double b[3]={-Qm.a[3],-Qm.a[6],-Qm.a[8]};
 double det=A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])-A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])+A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
 double sc=fabs(A[0][0])+fabs(A[1][1])+fabs(A[2][2]); if(fabs(det)<1e-9*sc*sc*sc||sc==0)return false;
 for(int i=0;i<3;i++){double M[3][3];for(int r=0;r<3;r++)for(int c=0;c<3;c++)M[r][c]=(c==i)?b[r]:A[r][c];
  p[i]=(M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1])-M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0])+M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]))/det;}return true;}
void push(int u,int v){Q Qm=q[u];Qm.add(q[v]);E e;e.u=u;e.v=v;e.vu=ver[u];e.vv=ver[v];double*pu=&V[3*u],*pv=&V[3*v];
 double mid[3]={(pu[0]+pv[0])/2,(pu[1]+pv[1])/2,(pu[2]+pv[2])/2};double L=sqrt((pu[0]-pv[0])*(pu[0]-pv[0])+(pu[1]-pv[1])*(pu[1]-pv[1])+(pu[2]-pv[2])*(pu[2]-pv[2]));
 double p[3];bool ok=solve(Qm,p);if(ok){double d=sqrt((p[0]-mid[0])*(p[0]-mid[0])+(p[1]-mid[1])*(p[1]-mid[1])+(p[2]-mid[2])*(p[2]-mid[2]));if(d>L)ok=false;}
 if(ok){e.c=Qm.ev(p[0],p[1],p[2]);for(int i=0;i<3;i++)e.p[i]=p[i];}
 else{double best=1e300;const double*cs[3]={pu,pv,mid};for(int k=0;k<3;k++){double c=Qm.ev(cs[k][0],cs[k][1],cs[k][2]);if(c<best){best=c;for(int i=0;i<3;i++)e.p[i]=cs[k][i];}}e.c=best;}
 pq.push(e);}
void nbrs(int u,vector<int>&out){out.clear();for(int f:vf[u])if(fa[f])for(int k=0;k<3;k++){int w=F[3*f+k];if(w!=u)out.push_back(w);}sort(out.begin(),out.end());out.erase(unique(out.begin(),out.end()),out.end());}
int main(int argc,char**argv){nV=atoi(argv[3]);nF=atoi(argv[4]);int target=atoi(argv[5]);
 V.resize(3*nV);F.resize(3*nF);FILE*f=fopen(argv[1],"rb");if(!fread(V.data(),8,3*nV,f))return 1;fclose(f);f=fopen(argv[2],"rb");if(!fread(F.data(),4,3*nF,f))return 1;fclose(f);
 fa.assign(nF,1);vf.resize(nV);q.resize(nV);ver.assign(nV,0);va.assign(nV,1);
 for(int t=0;t<nF;t++){int*g=&F[3*t];double n[3];fnorm(g[0],g[1],g[2],&V[3*g[0]],&V[3*g[1]],&V[3*g[2]],n);double l=sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);if(l<1e-20)l=1e-20;double A=l/2;for(int i=0;i<3;i++)n[i]/=l;
  double d=-(n[0]*V[3*g[0]]+n[1]*V[3*g[0]+1]+n[2]*V[3*g[0]+2]);Q Qt;double pl[4]={n[0],n[1],n[2],d};int idx=0;double*a=Qt.a;
  a[0]=pl[0]*pl[0];a[1]=pl[0]*pl[1];a[2]=pl[0]*pl[2];a[3]=pl[0]*pl[3];a[4]=pl[1]*pl[1];a[5]=pl[1]*pl[2];a[6]=pl[1]*pl[3];a[7]=pl[2]*pl[2];a[8]=pl[2]*pl[3];a[9]=pl[3]*pl[3];(void)idx;
  for(int i=0;i<10;i++)a[i]*=A;for(int k=0;k<3;k++){q[g[k]].add(Qt);vf[g[k]].push_back(t);}}
 for(int t=0;t<nF;t++)for(int k=0;k<3;k++){int a=F[3*t+k],b=F[3*t+(k+1)%3];if(a<b)push(a,b);}
 int alive=nF;vector<int>nu,nv,common;
 while(alive>target&&!pq.empty()){E e=pq.top();pq.pop();int u=e.u,v=e.v;if(!va[u]||!va[v]||ver[u]!=e.vu||ver[v]!=e.vv)continue;
  nbrs(u,nu);nbrs(v,nv);common.clear();set_intersection(nu.begin(),nu.end(),nv.begin(),nv.end(),back_inserter(common));
  int shared=0;for(int t:vf[u])if(fa[t]){int*g=&F[3*t];if(g[0]==v||g[1]==v||g[2]==v)shared++;}
  if(shared!=2||(int)common.size()!=2)continue;
  // flip / quality check
  bool bad=false;for(int pass=0;pass<2&&!bad;pass++){int x=pass?v:u;for(int t:vf[x]){if(!fa[t])continue;int*g=&F[3*t];if((g[0]==u||g[1]==u||g[2]==u)&&(g[0]==v||g[1]==v||g[2]==v))continue;
   double P[3][3],Pn[3][3];for(int k=0;k<3;k++)for(int i=0;i<3;i++){P[k][i]=V[3*g[k]+i];Pn[k][i]=(g[k]==x)?e.p[i]:P[k][i];}
   double n0[3],n1[3];fnorm(0,0,0,P[0],P[1],P[2],n0);fnorm(0,0,0,Pn[0],Pn[1],Pn[2],n1);double l0=sqrt(n0[0]*n0[0]+n0[1]*n0[1]+n0[2]*n0[2]),l1=sqrt(n1[0]*n1[0]+n1[1]*n1[1]+n1[2]*n1[2]);
   if(l1<1e-20||(n0[0]*n1[0]+n0[1]*n1[1]+n0[2]*n1[2])<0.2*l0*l1){bad=true;break;}
   double me=0;for(int k=0;k<3;k++){double s=0;for(int i=0;i<3;i++){double dd=Pn[k][i]-Pn[(k+1)%3][i];s+=dd*dd;}me=max(me,s);}
   if(l1/2/me<0.02){bad=true;break;}}}
  if(bad)continue;
  for(int i=0;i<3;i++)V[3*u+i]=e.p[i];
  for(int t:vf[v]){if(!fa[t])continue;int*g=&F[3*t];bool hasu=(g[0]==u||g[1]==u||g[2]==u);if(hasu){fa[t]=0;alive--;continue;}for(int k=0;k<3;k++)if(g[k]==v)g[k]=u;vf[u].push_back(t);}
  va[v]=0;q[u].add(q[v]);ver[u]++;
  vector<int>tmp;for(int t:vf[u])if(fa[t])tmp.push_back(t);vf[u]=tmp;
  nbrs(u,nu);for(int w:nu){ver[w]++;}
  // re-push edges around u and its neighbors' edges touching u
  for(int w:nu)push(min(u,w)==u?u:w,min(u,w)==u?w:u);
  for(int w:nu){vector<int>nw;nbrs(w,nw);for(int x:nw)if(x!=u)push(w,x);}
 }
 vector<int>map(nV,-1);int c=0;FILE*o=fopen(argv[6],"wb");for(int i=0;i<nV;i++)if(va[i]){map[i]=c++;fwrite(&V[3*i],8,3,o);}fclose(o);
 o=fopen(argv[7],"wb");int cf=0;for(int t=0;t<nF;t++)if(fa[t]){int g[3]={map[F[3*t]],map[F[3*t+1]],map[F[3*t+2]]};fwrite(g,4,3,o);cf++;}fclose(o);
 printf("%d %d\n",c,cf);}
