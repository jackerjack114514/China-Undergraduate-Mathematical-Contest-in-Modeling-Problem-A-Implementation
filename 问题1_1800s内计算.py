import numpy as np
import openpyxl
import pandas as pd
import os
import time
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

r=np.arange(N+1,dtype=float)*dr  #温度节点
alpha=k/(rho*cp)                 #热扩散率/(m²/s)

ii=np.arange(1,N,dtype=float)
gl=(1-1/(2*ii))/dr**2  #左
gr=(1+1/(2*ii))/dr**2  #右

#-----------------------------------------------到此为止均可复用

def sanduijiao(dl,d,du,b):  #！！顺序是否有误？
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
def fd(u,a,be,ua):       #边界参数：a：扩散率，be：边界参数，ua：空气边界值
    af=(a[:-1]+a[1:])/2  #水分扩散率
    al=dt*gl*af[:-1]     #左
    ar=dt*gr*af[1:]      #右

    dl=np.zeros(N);d=np.zeros(N);du=np.zeros(N)
    b=u[:-1].copy()

    a0=dt*4*af[0]/dr**2  #圆心差分的系数
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

    x=sanduijiao(dl,d,du,b)
    return np.append(x,p*x[-1]+q*x[-2]+s)

time_start=time.time()  #显示计算耗时用，可删去
T=np.full(N+1,28.0)     #温度初始化
C=np.full(N+1,2.55)     #含水量初始化

To=np.empty((nt//ot+1,nr+1))  #温度
Co=np.empty_like(To)          #含水率
To[0]=T[::N//nr]
Co[0]=C[::N//nr]

"""分段线性插值——【或许可以改为三次样条插值，以便后续问题复用？】
暂无必要，烘房内气温与湿度稳定缓慢上升，线性插值足矣"""
tg=np.arange(nt+1,dtype=float)*dt
Tg=np.interp(tg,t,Ta)  #空气温度插值,60秒一跳
Cg=np.interp(tg,t,Ca)  #空气湿度插值,60秒一跳

for it in range(1,nt+1):  #时间推进
    T=fd(T,np.full(N+1,alpha),h/k,Tg[it])
    #D使用前一时刻水分值，避免非线性Picard迭代（是否可靠？需要说明）
    D=7e-9*np.exp(-0.89/C)
    C=fd(C,D,hm/D[-1],Cg[it])
    if it%ot==0:
        iz=it//ot
        To[iz]=T[::N//nr]
        Co[iz]=C[::N//nr]


def xlsx(nm,z):   #时间、半径对应的温度、水分值
    sh=ans.create_sheet(nm)
    nf="0.0000" 
    sh.cell(1,1,"t(s)/r(cm)")
    for j in range(nr+1): 
        sh.cell(1,j+2,float(j/10)) 
    for i in range(nt//ot+1):
        sh.cell(i+2,1,i)
        for j in range(nr+1):
            c=sh.cell(i+2,j+2,round(float(z[i,j]),4))  #隐式Euler向后差分方法是二阶精度，对应应有二位浮点精度——前面取了N=400，足以保证四位精度^^
            c.number_format=nf
    """sh.freeze_panes="B2"
    sh.column_dimensions["A"].width=14"""  #方便查找特定行列的值，最终result输出记得加注释标

ans=openpyxl.Workbook()
w0=ans.active
ans.remove(w0)
xlsx("温度",To)
xlsx("水分浓度",Co)
ans.save(sc)
time_end=time.time()
time0=time_end-time_start
print("计算完成，耗时{:.4f}秒".format(time0))