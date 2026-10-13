import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path

#第二三四问，传热与水扩散相耦合。

相对路径=Path(__file__).resolve().parent #必须改成相对路径!
sr=str(相对路径/"附件"/"附件1.xlsx")
sc=str(相对路径/"result2.xlsx")


R=0.02  #药材半径/m  #第四问中成为变量
L=0.25  #药材长度/m

dt=0.125             #时间步长/s    #第一问：0.05
N=800                #径向网格区间数   #第一问：1600
dr=R/N               #径向步长/m
ts=10800.0           #总时长/s
nt=round(ts/dt)      #总时间步数
ot=round(1.0/dt)     #相邻输出时刻间隔的内部步数
nr=20                #输出径向区间数（0.001m/格）

h=25.0     #表面热对流系数/[W/(m²·K)]
hm=8e-7    #表面水扩散系数/(m/s)
pc=1e-10   #Picard迭代中的收敛值ε
pm=30      #最大Picard迭代数

sj=pd.read_excel(sr).iloc[:,:3].to_numpy(dtype=float)
sj=sj[np.isfinite(sj).all(axis=1)]
sj=sj[np.argsort(sj[:,0])]
t=sj[:,0]    #时间
Ta=sj[:,1]   #空气温度
Ca=sj[:,2]   #空气水分浓度

if t[0]>0 or t[-1]<ts:
    raise ValueError("附件1的时间范围必须覆盖0~10800s。")

r=np.arange(N+1,dtype=float)*dr   #径向节点/m
ii=np.arange(1,N,dtype=float)     #内部节点编号
gl=(1-1/(2*ii))/dr**2             #径向向左差分
gr=(1+1/(2*ii))/dr**2             #径向向右差分


def mat(T,C):
    Cs=np.maximum(C,1e-12)
    Tk=T+273.15
    rho=650+128*C
    cp=1450+2736*C/(C+1)
    k=0.21+0.38*C/(C+1)
    D=2.4e-3*np.exp(-0.45/Cs-3850/Tk)
    return rho,cp,k,D


def tdma(dl,d,du,b):
    n=len(d)
    dl=dl.copy()
    d=d.copy()
    du=du.copy()
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


def fd(u,a,S,be,ua):
    af=(a[:-1]+a[1:])/2
    al=dt*gl*af[:-1]/S[1:N]
    ar=dt*gr*af[1:]/S[1:N]

    dl=np.zeros(N)
    d=np.zeros(N)
    du=np.zeros(N)
    b=u[:-1].copy()

    a0=dt*4*af[0]/(S[0]*dr**2)
    d[0]=1+a0
    du[0]=-a0

    d[1:N-1]=1+al[:-1]+ar[:-1]
    dl[:N-1]=-al
    du[1:N-1]=-ar[:-1]
    d[-1]=1+al[-1]+ar[-1]
    dl[N-2]=-al[-1]

    v=3+2*dr*be
    p=4/v
    q=-1/v
    s=2*dr*be*ua/v

    d[-1]+=-ar[-1]*p
    dl[N-2]+=-ar[-1]*q
    b[-1]+=ar[-1]*s

    x=tdma(dl,d,du,b)
    return np.append(x,p*x[-1]+q*x[-2]+s)


time_start=time.time()
T0=28.0
C0=2.55
T=np.full(N+1,T0)
C=np.full(N+1,C0)
S1=np.ones(N+1)

# 保持原程序的输出精度：每1 s、每0.1 cm保存一次
To=np.empty((nt//ot+1,nr+1))
Co=np.empty_like(To)
To[0]=T[::N//nr]
Co[0]=C[::N//nr]

tg=np.arange(nt+1,dtype=float)*dt
Tg=np.interp(tg,t,Ta)
Cg=np.interp(tg,t,Ca)

pis=0
pim=0

for it in range(1,nt+1):
    Ti=T.copy()
    Ci=C.copy()

    for ip in range(1,pm+1):
        rhoi,cpi,ki,Di=mat(Ti,Ci)
        Si=rhoi*cpi
        Tn=fd(T,ki,Si,h/ki[-1],Tg[it])

        rhoi,cpi,ki,Di=mat(Tn,Ci)
        Cn=fd(C,Di,S1,hm/Di[-1],Cg[it])

        er=max(np.max(np.abs(Tn-Ti))/50,
               np.max(np.abs(Cn-Ci))/C0)

        Ti=Tn
        Ci=Cn

        if er<pc:
            break
    else:
        raise RuntimeError(
            "t={:.6f}s时Picard迭代在{}次内未收敛。".format(tg[it],pm)
        )

    T=Ti
    C=Ci
    pis+=ip
    pim=max(pim,ip)

    if it%ot==0:
        iz=it//ot
        To[iz]=T[::N//nr]
        Co[iz]=C[::N//nr]

    if it%(600*ot)==0:
        print("已计算前{}s".format(round(tg[it])))


#-----------------------------------绘图部分
plt.rc("font",size=16)
plt.rc("font",family="SimHei")
plt.rc("axes",unicode_minus=False)

times=np.arange(nt//ot+1,dtype=float)*dt*ot
r_cm=np.arange(nr+1,dtype=float)/10
ds=np.array([0,0.5,1,1.5,2],dtype=float)
ind=np.rint(ds/100/dr).astype(int)//(N//nr)

#不同半径处的时间—温度曲线
plt.figure(figsize=(9,5))
colors_T=plt.cm.YlOrRd(np.linspace(.35,.78,len(ds)))
for d,c,i in zip(ds,colors_T,ind):
    plt.plot(times,To[:,i],color=c,linewidth=1.5,
             alpha=.8,label=f"{d:g} cm")
plt.xlabel("时间/s")
plt.ylabel("温度/℃")
plt.title("问题2 时间—温度曲线")
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("C:\\Users\\Administrator\\Desktop\\问题2 时间—温度曲线图.png",dpi=500)

#不同半径处的时间—含水率曲线
plt.figure(figsize=(9,5))
colors_C=plt.cm.Blues(np.linspace(.35,.82,len(ds)))
for d,c,i in zip(ds,colors_C,ind):
    plt.plot(times,Co[:,i],color=c,linewidth=1.5,
             alpha=.8,label=f"{d:g} cm")
plt.xlabel("时间/s")
plt.ylabel("含水率/(kg·kg$^{-1}$)")
plt.title("问题2 时间—含水率曲线")
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig("C:\\Users\\Administrator\\Desktop\\问题2 时间—含水率曲线图.png",dpi=500)

#温度热力图
plt.figure(figsize=(9,5))
sets=plt.imshow(To.T,extent=[times[0],times[-1],r_cm[0],r_cm[-1]],origin="lower",aspect="auto",cmap="YlOrRd")
r_cm=np.linspace(0,R*100,To.shape[1])
cs=plt.contour(times,r_cm,To.T,levels=8,colors="black",linewidths=0.8)    #8条等温线
plt.xlabel("时间/s")
plt.ylabel("半径/cm")
plt.title("问题2 温度热力图")
plt.colorbar(sets,label="温度/℃")
plt.tight_layout()
plt.savefig("C:\\Users\\Administrator\\Desktop\\问题2 温度热力图.png",dpi=500)

#含水率热力图
plt.figure(figsize=(9,5))
sets=plt.imshow(Co.T,extent=[times[0],times[-1],r_cm[0],r_cm[-1]],origin="lower",aspect="auto",cmap="Blues")
cs=plt.contour(times,r_cm,Co.T,levels=8,colors="black",linewidths=0.8)    #8条等含水率线
plt.clabel(cs,inline=True,fontsize=8,fmt="%.2f")  #数值
plt.xlabel("时间/s")
plt.ylabel("半径/cm")
plt.title("问题2 含水率热力图")
plt.colorbar(sets,label="含水率/(kg/kg)")
plt.tight_layout()
plt.savefig("C:\\Users\\Administrator\\Desktop\\问题2 含水率热力图.png",dpi=500)

plt.show()

time_end=time.time()
time0=time_end-time_start
print("计算完成，耗时{:.4f}秒".format(time0))
print("Picard平均迭代次数：{:.4f}".format(pis/nt))
print("Picard单格最大迭代次数：{}".format(pim))
print("四张图已保存至桌面。")
