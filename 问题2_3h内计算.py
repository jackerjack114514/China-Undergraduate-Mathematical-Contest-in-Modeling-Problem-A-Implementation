import numpy as np
import openpyxl
import pandas as pd
import os
import time
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
pc=1e-8   #Picard迭代中的收敛值ε  非常关键！(1234统一为1e-8（原为1e-10）)
pm=30      #最大Picard迭代数（或许可以改为10？）

sj=pd.read_excel(sr).iloc[:,:3].to_numpy(dtype=float) 
sj=sj[np.isfinite(sj).all(axis=1)]
sj=sj[np.argsort(sj[:,0])] 
t=sj[:,0]    #时间
Ta=sj[:,1]   #空气温度
Ca=sj[:,2]   #空气水分浓度

if t[0]>0 or t[-1]<ts:
    raise ValueError("附件1的时间范围必须覆盖0~10800s。")

r=np.arange(N+1,dtype=float)*dr   #径向节点/m
ii=np.arange(1,N,dtype=float) 
gl=(1-1/(2*ii))/dr**2             #左
gr=(1+1/(2*ii))/dr**2             #右


def mat(T,C):
    Cs=np.maximum(C,1e-12)  #必须取非0的小值，不然除以0报错!!
    Tk=T+273.15   #只能取开氏度，不能取摄氏度...
    rho=650+128*C                      #药材密度/(kg/m³)
    cp=1450+2736*C/(C+1)               #比热容/[J/(kg·K)]
    k=0.21+0.38*C/(C+1)                #导热系数/[W/(m·K)]
    D=2.4e-3*np.exp(-0.45/Cs-3850/Tk)  #水分扩散系数/(m²/s)
    return rho,cp,k,D

def sanduijiao(dl,d,du,b):
    n=len(d)
    dl=dl.copy();d=d.copy();du=du.copy() #三对角方程
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
def fd(u,a,S,be,ua):  
    af=(a[:-1]+a[1:])/2       #半节点
    al=dt*gl*af[:-1]/S[1:N]   #左
    ar=dt*gr*af[1:]/S[1:N]    #右

    dl=np.zeros(N);d=np.zeros(N);du=np.zeros(N)   #三对角
    b=u[:-1].copy()  #右端向量

    a0=dt*4*af[0]/(S[0]*dr**2)  #圆心差分的系数
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

    x=sanduijiao(dl,d,du,b)  #内部节点解
    return np.append(x,p*x[-1]+q*x[-2]+s)

time_start=time.time()  #显示计算耗时用，可删去
T0=28.0                 #初始温度/℃
C0=2.55                 #初始干基含水率/(kg/kg)
T=np.full(N+1,T0)       #温度初始化
C=np.full(N+1,C0)       #含水率初始化
S1=np.ones(N+1)

To=np.empty((nt//ot+1,nr+1)) 
Co=np.empty_like(To)
To[0]=T[::N//nr]
Co[0]=C[::N//nr]

"""分段线性插值——【或许可以改为三次样条插值，以便后续问题复用？】
暂无必要，烘房内气温与湿度稳定缓慢上升，线性插值足矣"""
tg=np.arange(nt+1,dtype=float)*dt  
Tg=np.interp(tg,t,Ta)  #空气温度插值,60秒一跳
Cg=np.interp(tg,t,Ca)  #空气湿度插值,60秒一跳

pis=0  #Picard迭代次数总和
pim=0  #最大Picard迭代次数

for it in range(1,nt+1):  #时间推进
    Ti=T.copy()
    Ci=C.copy()

    for ip in range(1,pm+1): 
        rhoi,cpi,ki,Di=mat(Ti,Ci)        #计算下一步状态时冻结当前物理状态，规避非线性项【合理性证明？论文正文附出】
        Si=rhoi*cpi                      #体积热容
        Tn=fd(T,ki,Si,h/ki[-1],Tg[it])   #新温度

        rhoi,cpi,ki,Di=mat(Tn,Ci)
        Cn=fd(C,Di,S1,hm/Di[-1],Cg[it])  #新含水率

        er=max(np.max(np.abs(Tn-Ti))/50,np.max(np.abs(Cn-Ci))/C0)
        Ti=Tn  #更新温度迭代值
        Ci=Cn  #更新含水率迭代值

        if er<pc:  #满足Picard迭代条件，跳出循环？
            break
    else:
        raise RuntimeError("t={:.6f}s时Picard迭代在{}次内未收敛。".format(tg[it],pm))

    """中途是否会出现Picard迭代30次仍不收敛的情况？
    暂时不用考虑，跑一遍能跑通就行，且可在正文中用残差或Von-Neumann稳定性分析验证"""

    T=Ti              #温度已收敛
    C=Ci              #含水率已收敛
    pis+=ip
    pim=max(pim,ip)   #更新最大Picard迭代次数

    if it%ot==0:
        iz=it//ot
        To[iz]=T[::N//nr]
        Co[iz]=C[::N//nr]

    if it%(600*ot)==0:
        print("已计算前{}s".format(round(tg[it])))

def xlsx(nm,z):
    sh=ans.create_sheet(nm) 
    nf="0.0000"  #题目要求必须四位浮点数......
    sh.cell(1,1,"t(s)/r(cm)")
    for j in range(nr+1):      #径向
        sh.cell(1,j+2,float(j/10))      #不加这项就会把28.0000输出成28
    for i in range(nt//ot+1):  #时间
        c=sh.cell(i+2,1,float(i))
        for j in range(nr+1):
            c=sh.cell(i+2,j+2,round(float(z[i,j]),4))  #隐式Euler向后差分方法是二阶精度，对应应有二位浮点精度——前面取了N=400，足以保证四位精度^^
            c.number_format=nf

ans=openpyxl.Workbook()
w0=ans.active
ans.remove(w0)
xlsx("温度",To)
xlsx("水分浓度",Co)
ans.save(sc)
time_end=time.time()
time0=time_end-time_start
print("计算完成，耗时{:.4f}秒".format(time0))
print("Picard平均迭代次数：{:.4f}".format(pis/nt))
print("Picard单格最大迭代次数：{}".format(pim))
print("结果已保存至：{}！".format(sc))
