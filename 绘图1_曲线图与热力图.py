import numpy as np
import openpyxl
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path

#第一问传热与水扩散不耦合，可以分别求解。第二三四问需要替换对应部分。

相对路径=Path(__file__).resolve().parent #必须改成相对路径!
sr=str(相对路径/"附件"/"附件1.xlsx")
sc=str(相对路径/"result1.xlsx")

R=0.02  #药材半径/m  #第四问中成为变量
L=0.25  #药材长度/m

#隐式求解，后续Von-Neumann稳定性分析沿用此数据。 ！！调试完需复位：N=1600，dt=0.05
dt=0.05              #时间步长/s
N=1600               #径向网格区间数
dr=R/N               #径向步长/m
ts=1800.0            #总时长/s
nt=round(ts/dt)      #总时间步数
ot=round(1.0/dt)     #相邻输出时刻间隔的内部步数
nr=20                #输出径向区间数（0.001m/格）

rho=820.0    #药材密度/(kg/m³)
h=25.0       #表面热对流系数/[W/(m²·K)]
hm=8e-7      #表面水扩散系数/(m/s)
cp=2600.0    #比热容/[J/(kg·K)]
k=0.36       #导热系数/[W/(m·K)]

sj=pd.read_excel(sr).to_numpy(dtype=float)
sj=sj[np.isfinite(sj).all(axis=1)] 
sj=sj[np.argsort(sj[:,0])]
t=sj[:,0]   #时间
Ta=sj[:,1]  #空气温度
Ca=sj[:,2]  #空气水分浓度

ta=sj[:,0]     #取半径
Ta=sj[:,1]
tg=np.arange(nt+1)*dt
Tg=np.interp(tg,ta,Ta)

r=np.arange(N+1,dtype=float)*dr  #温度节点
alpha=k/(rho*cp)                 #热扩散率/(m²/s)

ii=np.arange(1,N,dtype=float)
gl=(1-1/(2*ii))/dr**2  #径向向左差分
gr=(1+1/(2*ii))/dr**2  #径向向右差分

#-----------------------------------------------到此为止均可复用

def tdma(dl,d,du,b):  #！！顺序是否有误？
    n=len(d)
    dl=dl.copy()  #下对角
    d=d.copy()    #主对角
    du=du.copy()  #上对角
    b=b.copy()

    for i in range(1,n):
        m=dl[i-1]/d[i-1]
        d[i]-=m*du[i-1]
        b[i]-=m*b[i-1]
    x=np.empty(n,dtype=float)
    x[-1]=b[-1]/d[-1]
    for i in range(n-2,-1,-1):
        x[i]=(b[i]-du[i]*x[i+1])/d[i]
    return x

#有限差分（暂时不考虑有限体积法）
def fd(u,a,be,ua):       #边界参数：a：扩散率，be：边界参数，ua：空气边界值
    af=(a[:-1]+a[1:])/2  #水分扩散率
    al=dt*gl*af[:-1]     #向左差分系数
    ar=dt*gr*af[1:]      #向右差分系数

    dl=np.zeros(N)
    d=np.zeros(N)
    du=np.zeros(N)
    b=u[:-1].copy()

    a0=dt*4*af[0]/dr**2  #圆心差分的系数
    d[0]=1+a0
    du[0]=-a0

    d[1:N-1]=1+al[:-1]+ar[:-1] #内部节点的平铺
    dl[:N-1]=-al
    du[1:N-1]=-ar[:-1]
    d[-1]=1+al[-1]+ar[-1]
    dl[N-2]=-al[-1]

    v=3+2*dr*be  #v、p、q、s：表面恢复系数
    p=4/v
    q=-1/v
    s=2*dr*be*ua/v 

    d[-1]+=-ar[-1]*p
    dl[N-2]+=-ar[-1]*q
    b[-1]+=ar[-1]*s

    x=tdma(dl,d,du,b)
    return np.append(x,p*x[-1]+q*x[-2]+s)



#保存全部节点（新！）
T=np.full(N+1,28.0)  
aa=np.full(N+1,alpha)
C=np.full(N+1,2.55) 
Da=np.full(N+1,7e-9*np.exp(-0.89/2.55))   # 水分扩散系数数组/(m²/s)

th=np.arange(nt+1)*dt     #温度
Th=np.empty((nt+1,N+1),dtype=np.float32)
Th[0]=T

tg=np.arange(nt+1)*dt
Tg=np.interp(tg,t,Ta)
Cg=np.interp(tg,t,Ca)

Ch=np.empty((nt+1,N+1),dtype=np.float32)
Cg=np.interp(tg,t,Ca)

for it in range(1,nt+1):
    T=fd(T,aa,h/k,Tg[it])
    Th[it,:]=T
    D=7e-9*np.exp(-0.89/C)
    C=fd(C,D,hm/D[-1],Cg[it])
    Ch[it,:]=C

#-----------------------------------绘图部分

plt.rc('font',size=16)
plt.rc('font',family='SimHei')
plt.rc('axes',unicode_minus=False)

#不同半径对应的时间-温度曲线图
plt.figure(figsize=(9,5))
ds=np.array([0,0.5,1,1.5,2])     #暂取result半径
ind=np.rint(ds/100/dr).astype(int)
cs=plt.cm.YlOrRd(np.linspace(.35,.78,len(ds)))  #可调同色渐变
for d,c,i in zip(ds,cs,ind):
    plt.plot(th,Th[:,i],color=c,linewidth=1.5,alpha=.4,label=f"{d:g} cm")
plt.xlabel("时间/s")
plt.ylabel("温度/℃")
plt.title("问题1 时间—温度曲线")
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("C:\\Users\\Administrator\\Desktop\\问题1 时间—温度曲线图.png",dpi=500)




#不同半径对应的时间-含水率曲线图
plt.figure(figsize=(9,5))
ds=np.array([0,0.5,1,1.5,2])     #暂取result半径/cm
ind=np.rint(ds/100/dr).astype(int)
cs=plt.cm.YlOrRd(np.linspace(.35,.78,len(ds)))
for d,c,i in zip(ds,cs,ind):
    plt.plot(th,Ch[:,i],color=c,linewidth=1.5,alpha=.4,label=f"{d:g} cm")
plt.xlabel("时间/s")
plt.ylabel("含水率/kg·kg$^{-1}$")
plt.title("问题1 时间—含水率曲线")
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("C:\\Users\\Administrator\\Desktop\\问题1 时间—含水率曲线图.png",dpi=500)
plt.show()

#温度热图
plt.figure(figsize=(9,5))
sets=plt.imshow(Th.T,extent=[th[0],th[-1],0,R*100],origin="lower",aspect="auto",cmap="YlOrRd")
r_cm=np.linspace(0,R*100,Th.shape[1])
cs=plt.contour(th,r_cm,Th.T,levels=8,colors="black",linewidths=0.8)    #8条等温线
plt.clabel(cs,inline=True,fontsize=8,fmt="%.1f")  #数值
plt.xlabel("时间/s")
plt.ylabel("半径/cm")
plt.title("问题1 温度热图")
plt.colorbar(sets,label="温度/℃")
plt.tight_layout()
plt.savefig("C:\\Users\\Administrator\\Desktop\\问题1 温度热力图.png",dpi=500)

#含水率热图
plt.figure(figsize=(9,5))
sets=plt.imshow(Ch.T,extent=[th[0],th[-1],0,R*100],origin="lower",aspect="auto",cmap="Blues")  #Blues_r
r_cm=np.linspace(0,R*100,Ch.shape[1])
cs=plt.contour(th,r_cm,Ch.T,levels=8,colors="black",linewidths=0.8)    #8条等含水率线
plt.clabel(cs,inline=True,fontsize=8,fmt="%.2f")  #数值
plt.xlabel("时间/s")
plt.ylabel("半径/cm")
plt.title("问题1 含水率热图")
plt.colorbar(sets,label="含水率/(kg/kg)")
plt.tight_layout()
plt.savefig("C:\\Users\\Administrator\\Desktop\\问题1 含水率热力图.png",dpi=500)
plt.show()
