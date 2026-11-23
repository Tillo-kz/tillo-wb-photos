"""viewcheck.py — проверка картинок-видов ChatGPT на нависания ДО генерации 3D (дёшево, без кредитов).
Силуэт фигурки на светлом фоне -> нижние (смотрящие вниз) участки контура круче theta от вертикали -> красным.
Ловит: подбородок/морду над грудью, чашку/лапы в воздухе, подол, поля шляпы, поднятую руку."""
import numpy as np,cv2
def silhouette(img,thr=28):
    b=np.concatenate([img[:8].reshape(-1,3),img[-8:].reshape(-1,3),img[:,:8].reshape(-1,3),img[:,-8:].reshape(-1,3)])
    bg=np.median(b,0); d=np.abs(img.astype(int)-bg).sum(2); m=(d>thr).astype(np.uint8)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((3,3),np.uint8)); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    n,lab,st,_=cv2.connectedComponentsWithStats(m); k=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); m=(lab==k).astype(np.uint8)
    ff=m.copy(); h,w=m.shape; mk=np.zeros((h+2,w+2),np.uint8); cv2.floodFill(ff,mk,(0,0),1); m=m|(1-ff)   # дырки внутри силуэта закрыть
    return m
def overhangs(img,theta=55,base_frac=0.04,sigma=2.0):
    """Возвращает (маска силуэта, bool-маска плохих пикселей контура, доля по высоте, картинка-отчёт)."""
    m=silhouette(img); H=m.shape[0]; ys=np.where(m.any(1))[0]; y0,y1=ys.min(),ys.max(); hgt=y1-y0
    s=cv2.GaussianBlur(m.astype(np.float32),(0,0),sigma); gx=cv2.Sobel(s,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(s,cv2.CV_32F,0,1,ksize=3)
    g=np.hypot(gx,gy)+1e-9; ny=-gy/g                      # наружная нормаль, y вниз
    edge=(m>0)&(cv2.erode(m,np.ones((3,3),np.uint8))==0)
    alpha=np.degrees(np.arcsin(np.clip(ny,0,1)))          # 90 = плоский «потолок», 0 = вертикаль
    bad=edge&(alpha>theta)&(np.arange(H)[:,None]<y1-base_frac*hgt)
    # маленькие отрезки (шум) не считаем
    n,lab,st,_=cv2.connectedComponentsWithStats(cv2.dilate(bad.astype(np.uint8),np.ones((3,3),np.uint8)))
    keep=np.zeros(n,bool); keep[1:]=st[1:,cv2.CC_STAT_AREA]>=max(6,0.01*hgt)
    bad=bad&keep[lab]
    rep=img.copy(); rep[cv2.dilate(bad.astype(np.uint8),np.ones((5,5),np.uint8))>0]=(0,0,255)
    zones=[]
    n2,lab2,st2,cen=cv2.connectedComponentsWithStats(cv2.dilate(bad.astype(np.uint8),np.ones((9,9),np.uint8)))
    for i in range(1,n2):
        zones.append(dict(height_pct=round(100*(y1-cen[i][1])/hgt),x_pct=round(100*(cen[i][0]-m.shape[1]/2)/hgt),len_px=int(st2[i,cv2.CC_STAT_AREA])))
    return m,bad,bad.sum()/max(hgt,1),rep,sorted(zones,key=lambda z:-z['len_px'])
