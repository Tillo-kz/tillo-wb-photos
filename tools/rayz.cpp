// vertical ray hits: for each column (i,j) center x0+i*h, y0+j*h, output hits (col, z, fid, sign)
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
int main(int argc,char**argv){
  int nV=atoi(argv[3]),nF=atoi(argv[4]); double x0=atof(argv[5]),y0=atof(argv[6]),h=atof(argv[7]); int NX=atoi(argv[8]),NY=atoi(argv[9]);
  std::vector<double>V(3*nV);std::vector<int>F(3*nF);
  FILE*f=fopen(argv[1],"rb");fread(V.data(),8,3*nV,f);fclose(f);
  f=fopen(argv[2],"rb");fread(F.data(),4,3*nF,f);fclose(f);
  FILE*o=fopen(argv[10],"wb");
  for(int t=0;t<nF;t++){
    double*a=&V[3*F[3*t]],*b=&V[3*F[3*t+1]],*c=&V[3*F[3*t+2]];
    double mnx=std::min({a[0],b[0],c[0]}),mxx=std::max({a[0],b[0],c[0]}),mny=std::min({a[1],b[1],c[1]}),mxy=std::max({a[1],b[1],c[1]});
    int i0=(int)ceil((mnx-x0)/h),i1=(int)floor((mxx-x0)/h),j0=(int)ceil((mny-y0)/h),j1=(int)floor((mxy-y0)/h);
    double d=(b[0]-a[0])*(c[1]-a[1])-(c[0]-a[0])*(b[1]-a[1]); if(fabs(d)<1e-14)continue;
    for(int i=std::max(i0,0);i<=std::min(i1,NX-1);i++)for(int j=std::max(j0,0);j<=std::min(j1,NY-1);j++){
      double px=x0+i*h+1e-7,py=y0+j*h+1.3e-7;
      double w1=((px-a[0])*(c[1]-a[1])-(c[0]-a[0])*(py-a[1]))/d;
      double w2=((b[0]-a[0])*(py-a[1])-(px-a[0])*(b[1]-a[1]))/d;
      double w0=1-w1-w2; if(w0<0||w1<0||w2<0)continue;
      double z=w0*a[2]+w1*b[2]+w2*c[2]; float rec[2]={(float)z,(float)(d>0?1:-1)}; int ri[2]={i*NY+j,t};
      fwrite(ri,4,2,o);fwrite(rec,4,2,o);
    }
  }
  fclose(o);
}
