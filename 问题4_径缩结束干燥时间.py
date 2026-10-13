import numpy as np
import openpyxl
import pandas as pd
import time
from scipy.interpolate import PchipInterpolator
from pathlib import Path

'''思路上，继续沿用问题3的温度—水分耦合模型，径向收缩由归一化消除；
同时用PCHIP保形三次样条插值对时间-半径插值^^'''

相对路径=Path(__file__).resolve().parent  #必须改成相对路径!
sr1=str(相对路径/"附件"/"附件1.xlsx")
sr2=str(相对路径/"附件"/"附件2.xlsx")
sc=str(相对路径/"result4.xlsx")

L=0.25                 #药材长度不变，简化只考虑径向收缩
dt=5.0                 #时间步长/s；#问题1：dt=0.05 问题2：0.125s  #必须取60的约数，否则报错！
N=800                  #归一化径向网格区间数  #跟问题三保持一致
dx=1.0/N               #归一化径向步长
tm=7*24*3600.0         #最长计算时间/s （必然小于第三问）
nt=round(tm/dt)        #最大时间步数
ot=round(60.0/dt)      #60s内步数
oh=round(21600.0/dt)   #6h内步数
nr=20

h=25.0                 #表面热对流系数/[W/(m²·K)]
hm=8e-7                #表面水扩散系数/(m/s)
pc=1e-8                #Picard收敛值
pm=100                 #最大Picard迭代数（不用担心不收敛）
Cl=0.15                #烘干结束含水率阈值/(kg/kg)，到达后结束

"""附件1"""
sj1=pd.read_excel(sr1).iloc[:,:3].to_numpy(dtype=float)
sj1=sj1[np.isfinite(sj1).all(axis=1)]
sj1=sj1[np.argsort(sj1[:,0])]
t=sj1[:,0]             #时间/s
Ta=sj1[:,1]            #空气温度/℃
Ca=sj1[:,2]            #空气水分浓度/(kg/kg)
"""附件2"""
sj2=pd.read_excel(sr2).iloc[:,:2].to_numpy(dtype=float)
sj2=sj2[np.isfinite(sj2).all(axis=1)]
sj2=sj2[np.argsort(sj2[:,0])]
tr=sj2[:,0]            #时间/s
Ra=sj2[:,1]/100.0      #半径/m

rp=PchipInterpolator(tr,Ra,extrapolate=False)  #半径插值/m
rd=rp.derivative()                             #半径变化率/(m/s)

def radius(tn):
    if tn<=tr[-1]: Rn=float(rp(tn)) 
    else: Rn=float(Ra[-1])
    return Rn

def radius_rate(tn):    #半径变化率
    if tn<tr[-1]: Rv=float(rd(tn))
    else: Rv=0.0
    if abs(Rv)<1e-12: Rv=0.0
    return Rv

x=np.arange(N+1,dtype=float)*dx  #固定ξ
ii=np.arange(1,N,dtype=float) 
gl=(1-1/(2*ii))/dx**2            #向左
gr=(1+1/(2*ii))/dx**2            #向右

rq=np.arange(nr+1,dtype=float)*0.001  #0.001m输出，SI标准!

def mat(T,C):
    Cs=np.maximum(C,1e-12)  #取微小值，防止扩散系数公式除以0
    Tk=T+273.15             ##开氏度。
    rho=760+90*C                       #药材密度/(kg/m³)
    cp=1850+2150*C/(C+1)               #比热容/[J/(kg·K)]
    k=0.12+0.20*C/(C+1)                #导热系数/[W/(m·K)]
    D=4.2e-4*np.exp(-0.30/Cs-3850/Tk)  #水分扩散系数/(m²/s)
    return rho,cp,k,D

def rad(tn,tp):  #r(t)  r'(t)
    Rn=radius(tn)
    Rv=radius_rate(tn)
    return Rn,Rv

def sanduijiao(dl,d,du,b): 
    n=len(d)
    dl=dl.copy();d=d.copy();du=du.copy() #三对角
    b=b.copy()

    for i in range(1,n):
        m=dl[i-1]/d[i-1]
        d[i]-=m*du[i-1]
        b[i]-=m*b[i-1]
    z=np.empty(n,dtype=float)  #解向量 POINT
    z[-1]=b[-1]/d[-1]
    for i in range(n-2,-1,-1):
        z[i]=(b[i]-du[i]*z[i+1])/d[i]
    return z

#复用有限差分方程（需要增加1/R²和由r'产生的迎风迁移系数）!
def fd(u,a,S,be,ua,Rn,Rv):
    af=(a[:-1]+a[1:])/2                   #半节点
    al=dt*gl*af[:-1]/(S[1:N]*Rn**2)       #向左
    ar=dt*gr*af[1:]/(S[1:N]*Rn**2)        #向右
    et=-dt*(Rv/Rn)*ii                     #收缩迁移系数η_i

    dl=np.zeros(N);d=np.zeros(N);du=np.zeros(N)   #下对角&&主对角&&上对角
    b=u[:-1].copy()  #右端向量

    a0=dt*4*af[0]/(S[0]*Rn**2*dx**2)  #圆心扩散
    d[0]=1+a0
    du[0]=-a0

    d[1:N-1]=1+al[:-1]+ar[:-1]+et[:-1]   #内部-主对角
    dl[:N-1]=-(al+et)                    #下对角
    du[1:N-1]=-ar[:-1]                   #上对角
    d[-1]=1+al[-1]+ar[-1]+et[-1]         #倒二-主对角
    dl[N-2]=-(al[-1]+et[-1])             #下对角

    v=3+2*dx*be
    p=4/v
    q=-1/v 
    s=2*dx*be*ua/v

    d[-1]+=-ar[-1]*p
    dl[N-2]+=-ar[-1]*q
    b[-1]+=ar[-1]*s

    z=sanduijiao(dl,d,du,b)  #内部解
    return np.append(z,p*z[-1]+q*z[-2]+s)

#-----------------------------------------------到此为止均可复用(改)

def sample(C,Rn):
    z=np.full(nr+1,np.nan,dtype=float)  #固定位置的含水率；域外设为inf
    mk=rq<=Rn+1e-12
    xq=np.clip(rq[mk]/Rn,0,1)          
    z[mk]=np.interp(xq,x,C)             #固定物理位置
    return z

time_start=time.time()
T0=28.0                 #初始温度/℃
C0=2.55                 #初始含水率/(kg/kg)
T=np.full(N+1,T0)
C=np.full(N+1,C0) 
S1=np.ones(N+1)         #水分方程储能系数
R0=float(Ra[0])         #初始半径/m

to=[0.0]                #Excel输出时刻/s
Co=[sample(C,R0)]       #含水率
Cs=[float(C[-1])]       #表面含水率
Ro=[R0*100]             #当前半径/cm
pis=0                   #全部时间步的Picard迭代次数之和
pim=0                   #单个时间步的最大Picard迭代次数


Ta0=50.00036     #延拓温度--NEW
Ca0=0.0499913    #延拓湿度--NEW

for it in range(1,nt+1):
    tn=it*dt                           #当前新时刻/s
    tp=(it-1)*dt                       #上一个时刻/s
    Rn,Rv=rad(tn,tp)                   #当前半径及PCHIP瞬时变化率
    Cp=C.copy()                        #上一个时刻含水率，用于达标时间插值
    Ti=T.copy()
    Ci=C.copy()
    '''Tai=float(np.interp(tn,t,Ta))     #当前空气温度/℃   #弃用
    Cai=float(np.interp(tn,t,Ca))     #当前空气水分浓度/(kg/kg)'''

    Tai=float(np.interp(tn,t,Ta,right=Ta0))   #取均值延拓，具体数值由excel给出
    Cai=float(np.interp(tn,t,Ca,right=Ca0))

    for ip in range(1,pm+1):
        rhoi,cpi,ki,Di=mat(Ti,Ci)
        Si=rhoi*cpi                           #体积热容/[J/(m³·K)]
        Tn=fd(T,ki,Si,Rn*h/ki[-1],Tai,Rn,Rv)  #更新温度场

        rhoi,cpi,ki,Di=mat(Tn,Ci)             #新扩散系数
        Cn=fd(C,Di,S1,Rn*hm/Di[-1],Cai,Rn,Rv) #新含水率

        er=max(np.max(np.abs(Tn-Ti))/50,np.max(np.abs(Cn-Ci))/C0)
        Ti=Tn
        Ci=Cn

        if er<pc:  #Picard收敛条件
            break
    else:
        raise RuntimeError("t={:.6f}s时Picard迭代在{}次内未收敛。".format(tn,pm))

    T=Ti              #温度已收敛
    C=Ci              #含水率已收敛
    pis+=ip
    pim=max(pim,ip)   #最大Picard迭代次数

    if np.max(C)<=Cl:
        mk=Cp>Cl
        if np.any(mk):
            th=np.max((Cp[mk]-Cl)/(Cp[mk]-C[mk]))
            th=float(np.clip(th,0,1))
        else:
            th=0.0
        te=(it-1+th)*dt                #烘干结束时间/s
        Ce=Cp+th*(C-Cp)
        Re=radius(te)                  #PCHIP得结束时刻半径/m
        if abs(te-to[-1])<1e-10:
            Co[-1]=sample(Ce,Re)
            Cs[-1]=float(Ce[-1])
            Ro[-1]=Re*100
        else:
            to.append(te)
            Co.append(sample(Ce,Re))
            Cs.append(float(Ce[-1]))
            Ro.append(Re*100)
        break

    if it%ot==0:
        to.append(tn)
        Co.append(sample(C,Rn))
        Cs.append(float(C[-1]))
        Ro.append(Rn*100)

    if it%oh==0:
        print("已计算{:.0f}h，当前半径={:.4f}cm，中心含水率={:.6f}".format(tn/3600,Rn*100,C[0]))
else:
    raise RuntimeError("计算7天后药材各处含水率仍未降至0.15kg/kg以下。")

to=np.asarray(to,dtype=float)  #全xcel输出时刻/s
Co=np.asarray(Co,dtype=float)  #固定位置含水率
Cs=np.asarray(Cs,dtype=float)  #表面含水率
Ro=np.asarray(Ro,dtype=float)  #当前半径/cm


def xlsx(nm,z,zs,rr,tt):
    sh=ans.create_sheet(nm)
    nf="0.0000"
    sh.cell(1,1,"t(s)/r(cm)")
    for j in range(nr+1):
        c=sh.cell(1,j+2,float(j/10))
        c.number_format="General"   #!!头一行半径不取浮点格式
    sh.cell(1,nr+3,"药材表面")

    for i in range(len(tt)):
        c=sh.cell(i+2,1,round(float(tt[i]),4))
        c.number_format="General"   #时间列不取浮点!!

        for j in range(nr+1):
            if np.isfinite(z[i,j]):
                c=sh.cell(i+2,j+2,round(float(z[i,j]),4))
                c.number_format=nf     
            else:
                sh.cell(i+2,j+2,None)    #收缩后该位置已成滚木

        c=sh.cell(i+2,nr+3,round(float(zs[i]),4))
        c.number_format=nf

ans=openpyxl.Workbook()
w0=ans.active
ans.remove(w0)
xlsx("水分浓度",Co,Cs,Ro,to)
ans.save(sc)

ri=[0,5,10]
print("\n表6：考虑收缩时药材烘干过程的水分浓度/(kg/kg)")
print("时间/h    半径/cm       0cm       0.5cm       1cm       药材表面")
for hs in range(6,int(te//21600)*6+1,6):
    ir=int(hs*3600/60)
    zv=["{:.4f}".format(Co[ir,j]) if np.isfinite(Co[ir,j]) else "    --" for j in ri]
    print("{:6.1f}  {:9.4f}  {}  {:10.4f}".format(float(hs),Ro[ir],"  ".join(zv),Cs[ir]))

zv=["{:.4f}".format(Co[-1,j]) if np.isfinite(Co[-1,j]) else "    --" for j in ri]   #必须输出占位符，否则NA/inf报错!
print("结束{:6.4f}  {:9.4f}  {}  {:10.4f}".format(te/3600,Ro[-1],"  ".join(zv),Cs[-1]))

time_end=time.time()
time0=time_end-time_start
print("\n计算完成，总耗时{:.4f}秒".format(time0))
print("考虑收缩后的烘干结束时间：{:.4f}h（{:.4f}s）".format(te/3600,te))
print("烘干结束时药材半径：{:.4f}cm".format(Re*100))
print("烘干结束时最大含水率：{:.10f}kg/kg".format(np.max(Ce)))
print("Picard平均迭代次数：{:.4f}".format(pis/it))
print("Picard最大迭代次数：{}".format(pim))
print("结果已保存至：{}".format(sc))
