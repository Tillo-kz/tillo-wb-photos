// Unsigned distance to a triangle mesh on a regular grid, only inside a band (|d|<=B).
// usage: sdfband V.bin F.bin nV nF x0 y0 z0 h NX NY NZ B out.bin   (out: float32 NX*NY*NZ, index (i*NY+j)*NZ+k, >B => B)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
struct P{double x,y,z;};
static inline P sub(P a,P b){return {a.x-b.x,a.y-b.y,a.z-b.z};}
static inline double dot(P a,P b){return a.x*b.x+a.y*b.y+a.z*b.z;}
static inline P add(P a,P b){return {a.x+b.x,a.y+b.y,a.z+b.z};}
static inline P mul(P a,double s){return {a.x*s,a.y*s,a.z*s};}
// Ericson, Real-Time Collision Detection, ClosestPtPointTriangle
static P closest(P p,P a,P b,P c){
  P ab=sub(b,a),ac=sub(c,a),ap=sub(p,a);double d1=dot(ab,ap),d2=dot(ac,ap);if(d1<=0&&d2<=0)return a;
  P bp=sub(p,b);double d3=dot(ab,bp),d4=dot(ac,bp);if(d3>=0&&d4<=d3)return b;
  double vc=d1*d4-d3*d2;if(vc<=0&&d1>=0&&d3<=0){double v=d1/(d1-d3);return add(a,mul(ab,v));}
  P cp=sub(p,c);double d5=dot(ab,cp),d6=dot(ac,cp);if(d6>=0&&d5<=d6)return c;
  double vb=d5*d2-d1*d6;if(vb<=0&&d2>=0&&d6<=0){double w=d2/(d2-d6);return add(a,mul(ac,w));}
  double va=d3*d6-d5*d4;if(va<=0&&(d4-d3)>=0&&(d5-d6)>=0){double w=(d4-d3)/((d4-d3)+(d5-d6));return add(b,mul(sub(c,b),w));}
  double den=1.0/(va+vb+vc);double v=vb*den,w=vc*den;return add(a,add(mul(ab,v),mul(ac,w)));}
int main(int argc,char**argv){
  int nV=atoi(argv[3]),nF=atoi(argv[4]);double x0=atof(argv[5]),y0=atof(argv[6]),z0=atof(argv[7]),h=atof(argv[8]);
  long NX=atol(argv[9]),NY=atol(argv[10]),NZ=atol(argv[11]);double B=atof(argv[12]);
  std::vector<double>V(3*(size_t)nV);std::vector<int>F(3*(size_t)nF);
  FILE*f=fopen(argv[1],"rb");if(fread(V.data(),8,3*(size_t)nV,f)!=3*(size_t)nV)return 1;fclose(f);
  f=fopen(argv[2],"rb");if(fread(F.data(),4,3*(size_t)nF,f)!=3*(size_t)nF)return 1;fclose(f);
  std::vector<float>D((size_t)NX*NY*NZ,(float)B);
  for(int t=0;t<nF;t++){
    P a={V[3*F[3*t]],V[3*F[3*t]+1],V[3*F[3*t]+2]},b={V[3*F[3*t+1]],V[3*F[3*t+1]+1],V[3*F[3*t+1]+2]},c={V[3*F[3*t+2]],V[3*F[3*t+2]+1],V[3*F[3*t+2]+2]};
    double mn[3]={std::min({a.x,b.x,c.x})-B,std::min({a.y,b.y,c.y})-B,std::min({a.z,b.z,c.z})-B};
    double mx[3]={std::max({a.x,b.x,c.x})+B,std::max({a.y,b.y,c.y})+B,std::max({a.z,b.z,c.z})+B};
    long i0=std::max(0L,(long)ceil((mn[0]-x0)/h)),i1=std::min(NX-1,(long)floor((mx[0]-x0)/h));
    long j0=std::max(0L,(long)ceil((mn[1]-y0)/h)),j1=std::min(NY-1,(long)floor((mx[1]-y0)/h));
    long k0=std::max(0L,(long)ceil((mn[2]-z0)/h)),k1=std::min(NZ-1,(long)floor((mx[2]-z0)/h));
    for(long i=i0;i<=i1;i++)for(long j=j0;j<=j1;j++){size_t base=((size_t)i*NY+j)*NZ;
      for(long k=k0;k<=k1;k++){P p={x0+i*h,y0+j*h,z0+k*h};P q=closest(p,a,b,c);P d=sub(p,q);double dd=sqrt(dot(d,d));
        if(dd<D[base+k])D[base+k]=(float)dd;}}
  }
  f=fopen(argv[13],"wb");fwrite(D.data(),4,D.size(),f);fclose(f);return 0;}
