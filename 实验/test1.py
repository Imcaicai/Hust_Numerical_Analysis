#coding=utf-8

'''
由给定的关于变量x与y对应的一组数据，且插值节点两两互异，数据以两个列表x_list与y_list
的形式提供，编写函数实现Lagrange插值法（不应调用python三方库中已实现的多项式插值计算
函数），求插值多项式在插值点x的值。
'''
from turtle import color
import numpy as np
import matplotlib.pyplot as plt

#编写下述Lagrange插值函数

def Lagrange_poly(x_list,y_list,x):
#任务一：请在此处添加代码完成该函数，实现Lagrange插值计算功能
######Begin_1######
    ans=0.0
    for i in range(len(y_list)):
        z=y_list[i]
        for j in range(len(y_list)):
            if i!=j:
                z*=(x-x_list[j])/(x_list[i]-x_list[j])
        ans+=z
    return ans
######End_1######


def f(x):
    return 5/(1+x*x)


#任务二：依据下述数据表，调用Lagrange_poly函数，利用插值多项式计算x=11.75的近似值，
#并打印输出，另画图显示插值多项式曲线与各插值节点
'''
x   10       11       12       13
y   2.3026   2.3979   2.4849   2.5649
'''
######Begin_2######
x=[10,11,12,13]
y=[2.3026,2.3979,2.4849,2.5649]
print(Lagrange_poly(x,y,11.75))
x_show=np.arange(10,13,0.1)
y_show=Lagrange_poly(x,y,x_show)
plt.plot(x_show,y_show)
plt.scatter(10,Lagrange_poly(x,y,10),c='r')
plt.scatter(11,Lagrange_poly(x,y,11),c='r')
plt.scatter(12,Lagrange_poly(x,y,12),c='r')
plt.scatter(13,Lagrange_poly(x,y,13),c='r')
plt.title('Lagrange插值')
plt.show()
######End_2######


#任务三：对于被插值函数f(x)=5/(1+x^2),取不同的节点数n=5,10，在区间[-5,5]取n等分节点作为
#插值节点，将f(x)与插值多项式曲线画在同一张图上直观比较，分析Runge现象。
######Begin_3######
x0=np.arange(-5,5,0.01)
y0=f(x0)
x1=np.arange(-5,5,2)
y1=Lagrange_poly(x1,f(x1),x0)
l1,=plt.plot(x0,y0,color='orange',label='f(x)')
l2,=plt.plot(x0,y1,color='green',label='n=5')
plt.legend([l1,l2],['f(x)','Lagrange: n=5'],loc=2)
plt.show()

x2=np.arange(-5,5,1)
y2=Lagrange_poly(x2,f(x2),x0)
l1,=plt.plot(x0,y0,color='orange',label='f(x)')
l3,=plt.plot(x0,y2,color='blue',label='n=10')
plt.legend([l1,l3],['f(x)','Lagrange: n=10'],loc=2)
plt.show()
######End_3######

