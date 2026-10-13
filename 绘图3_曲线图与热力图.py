import numpy as np
import pandas as pd
import os
import time
import matplotlib.pyplot as plt
from pathlib import Path

"""依旧温度与水分耦合，药材中心即为含水率最高处，取其含水率不高于0.15kg/kg的时刻即可。
P.S.但是否必须严格小于？"""

相对路径=Path(__file__).resolve().parent #必须改成相对路径！
sr=str(相对路径/"附件"/"附件1.xlsx")
sc=str(相对路径/"result3.xlsx")

R=0.02  #药材半径/m
L=0.25  #药材长度/m

#dt、N、pc关键 POINT  稳定性与误差分析需要调整这三个参数！！
dt=5.0                #时间步长/s  #问题1：dt=0.05 问题2：0.125s  #必须取60的约数，否则报错！
N=800                 #径向网格区间数（不能取400以下）
dr=R/N                #径向步长/m
tm=7*24*3600.0        #最长计算时间（实际达不到）
nt=round(tm/dt)       #最大时间步数
ot=round(60.0/dt)
oh=round(21600.0/dt)
nr=20                 #结果径向区间数（0.001m/格）

h=25.0      #表面热对流系数/[W/(m²·K)]
hm=8e-7     #表面水扩散系数/(m/s)
pc=1e-8     #Picard收敛值ε
pm=100      #最大Picard迭代数
Cl=0.15     #含水率阈值/(kg/kg)，到达后结束

sj=pd.read_excel(sr).iloc[:,:3].to_numpy(dtype=float)
sj=sj[np.isfinite(sj).all(axis=1)]
sj=sj[np.argsort(sj[:,0])]
t=sj[:,0]    #时间/s
Ta=sj[:,1]   #空气温度/℃
Ca=sj[:,2]   #空气水分浓度/(kg/kg)

r=np.arange(N+1,dtype=float)*dr
ii=np.arange(1,N,dtype=float)
gl=(1-1/(2*ii))/dr**2
gr=(1+1/(2*ii))/dr**2


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


#-----------------------------------------------到此为止均沿用问题2的微分方程

time_start=time.time()
T0=28.0
C0=2.55
T=np.full(N+1,T0)
C=np.full(N+1,C0)
S1=np.ones(N+1)

# 保存每60 s、每0.1 cm的温度和含水率结果
to=[0.0]
To=[T[::N//nr].copy()]
Co=[C[::N//nr].copy()]
pis=0
pim=0

for it in range(1,nt+1):
    tn=it*dt
    Tp=T.copy()
    Cp=C.copy()
    Ti=T.copy()
    Ci=C.copy()

    Tai=float(np.interp(tn,t,Ta))
    Cai=float(np.interp(tn,t,Ca))

    for ip in range(1,pm+1):
        rhoi,cpi,ki,Di=mat(Ti,Ci)
        Si=rhoi*cpi
        Tn=fd(T,ki,Si,h/ki[-1],Tai)

        rhoi,cpi,ki,Di=mat(Tn,Ci)
        Cn=fd(C,Di,S1,hm/Di[-1],Cai)

        er=max(np.max(np.abs(Tn-Ti))/50,
               np.max(np.abs(Cn-Ci))/C0)

        Ti=Tn
        Ci=Cn

        if er<pc:
            break
    else:
        raise RuntimeError(
            "t={:.6f}s时Picard迭代在{}次内未收敛。".format(tn,pm)
        )

    T=Ti
    C=Ci
    pis+=ip
    pim=max(pim,ip)

    if np.max(C)<=Cl:
        mk=Cp>Cl
        if np.any(mk):
            th=np.max((Cp[mk]-Cl)/(Cp[mk]-C[mk]))
            th=float(np.clip(th,0,1))
        else:
            th=0.0

        te=(it-1+th)*dt
        Ce=Cp+th*(C-Cp)
        Te=Tp+th*(T-Tp)

        if abs(te-to[-1])<1e-10:
            To[-1]=Te[::N//nr].copy()
            Co[-1]=Ce[::N//nr].copy()
        else:
            to.append(te)
            To.append(Te[::N//nr].copy())
            Co.append(Ce[::N//nr].copy())
        break

    if it%ot==0:
        to.append(tn)
        To.append(T[::N//nr].copy())
        Co.append(C[::N//nr].copy())

    if it%oh==0:
        print("已计算{:.0f}h，中心含水率={:.6f}".format(tn/3600,C[0]))
else:
    raise RuntimeError(
        "计算7天后药材各处含水率仍未降至0.15kg/kg以下。"
    )

to=np.asarray(to,dtype=float)
To=np.asarray(To,dtype=float)
Co=np.asarray(Co,dtype=float)

#-----------------------------------绘图部分

plt.rc("font",size=16)
plt.rc("font",family="SimHei")
plt.rc("axes",unicode_minus=False)

times=to
r_cm=np.arange(nr+1,dtype=float)/10
ri=[0,5,10,15,20]
ds=[0,0.5,1,1.5,2]

# 1. 不同半径处的时间—温度曲线
plt.figure(figsize=(9,5))
colors_T=plt.cm.YlOrRd(np.linspace(.35,.78,len(ri)))
for d,c,i in zip(ds,colors_T,ri):
    plt.plot(times,To[:,i],color=c,linewidth=1.5,
             alpha=.8,label=f"{d:g} cm")
plt.xlabel("时间/s")
plt.ylabel("温度/℃")
plt.title("问题3 时间—温度曲线")
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(
    "C:\\Users\\Administrator\\Desktop\\问题3 时间—温度曲线图.png",
    dpi=500
)

# 2. 不同半径处的时间—含水率曲线
plt.figure(figsize=(9,5))
colors_C=plt.cm.Blues(np.linspace(.35,.82,len(ri)))
for d,c,i in zip(ds,colors_C,ri):
    plt.plot(times,Co[:,i],color=c,linewidth=1.5,
             alpha=.8,label=f"{d:g} cm")
plt.xlabel("时间/s")
plt.ylabel("含水率/(kg·kg$^{-1}$)")
plt.title("问题3 时间—含水率曲线")
plt.legend()
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(
    "C:\\Users\\Administrator\\Desktop\\问题3 时间—含水率曲线图.png",
    dpi=500
)

# 3. 温度热力图
plt.figure(figsize=(9,5))
im=plt.imshow(
    To.T,
    extent=[times[0],times[-1],r_cm[0],r_cm[-1]],
    origin="lower",
    aspect="auto",
    cmap="YlOrRd"
)
plt.xlabel("时间/s")
plt.ylabel("半径/cm")
plt.title("问题3 温度热力图")
plt.colorbar(im,label="温度/℃")
plt.tight_layout()
plt.savefig(
    "C:\\Users\\Administrator\\Desktop\\问题3 温度热力图.png",
    dpi=500
)

# 4. 含水率热力图
plt.figure(figsize=(9,5))
im=plt.imshow(
    Co.T,
    extent=[times[0],times[-1],r_cm[0],r_cm[-1]],
    origin="lower",
    aspect="auto",
    cmap="Blues"
)
plt.xlabel("时间/s")
plt.ylabel("半径/cm")
plt.title("问题3 含水率热力图")
plt.colorbar(im,label="含水率/(kg·kg$^{-1}$)")
plt.tight_layout()
plt.savefig(
    "C:\\Users\\Administrator\\Desktop\\问题3 含水率热力图.png",
    dpi=500
)

# 只在四张图全部生成后统一显示
plt.show()

ri=[0,5,10,15,20]
print("\n表5：药材烘干过程的水分浓度/(kg/kg)")
print("时间/h       0cm       0.5cm       1cm       1.5cm       2cm")
for hs in range(6,int(te//21600)*6+1,6):
    ir=int(hs*3600/60)
    print("{:6.1f}  ".format(float(hs))+"  ".join(
        "{:.4f}".format(x) for x in Co[ir,ri]
    ))
print("结束{:6.4f}  ".format(te/3600)+"  ".join(
    "{:.4f}".format(x) for x in Co[-1,ri]
))

time_end=time.time()
time0=time_end-time_start
print("\n计算完成，总耗时{:.4f}秒".format(time0))
print("烘干结束时间：{:.4f}h（{:.4f}s）".format(te/3600,te))
print("烘干结束时中心含水率：{:.10f}kg/kg".format(np.max(Ce)))
print("Picard平均迭代次数：{:.4f}".format(pis/it))
print("Picard最大迭代次数：{}".format(pim))
print("四张图已保存至桌面。")
