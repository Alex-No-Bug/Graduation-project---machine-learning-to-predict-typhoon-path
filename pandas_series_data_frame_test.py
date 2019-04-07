import predict_SST as preSST# 引用预测温度的文件
import predict_PA  as prePa
import predict_Hgt as preHgt
#xt5=f(xt,sst,sp,pa)
#拟合出来的函数
def fun(C,coef,arr):
    sum=0
    for i in range(len(coef[0])):
        sum=sum+coef[0][i]*arr[i]
    return C+sum

#1.数据载入&预处理
import numpy as np
import pandas as pd 
xt=tuple(np.array([0,1,2,3,4,5,6,7,8,9]) )     #不可变的hash类型
ssp=tuple(np.array([22.1,22.1,24.1,21.3,23.4,25.4,25.4,24.8,24.5,24.8]) )
pa=tuple(np.array([34,35,33,33.3,33.5,33.4,37.8,32.1,32,34.2]) )
hgt=tuple(np.array([340,220,222,333,444,555,666,777,888,1111]) )
xt5=tuple(np.array([10,20.2,30.3,40.4,52.5,62.6,72.7,82.8,93.9,130]) )
dx={'xt':xt, 'ssp':ssp, 'pa':pa, 'hgt':hgt, 'xt5':xt5}
data_x=pd.DataFrame(dx)
print(data_x)
xx=data_x[['xt','ssp','pa','hgt']]
yx=data_x[['xt5']]

#2.建模 预测XT+5
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#核心代码
x_train1,x_test1,y_train1,y_test1=train_test_split(xx,yx,random_state=1)#分割train-set test-set
lr1=LinearRegression()
lr1.fit(x_train1,y_train1)#开始拟合
print(lr1.intercept_)     #截距  一定要输出看看是不是数组还是单个变量
print(lr1.coef_)          #系数矩阵
#accept1=input('input the 4 datas to predict x:\n').split()#分割字符串
#print( fun(lr1.intercept_[0],lr1.coef_,accept1) )

#同理预测yT+5值
yt=tuple(np.array([0.1,1.1,2.4,3.6,4.7,5.8,6.8,7.9,8,9.4]) )     #不可变的hash类型
ssp=tuple(np.array([22.1,22.1,24.1,21.3,23.4,25.4,25.4,24.8,24.5,24.8]) )
pa=tuple(np.array([34,35,33,33.3,33.5,33.4,37.8,32.1,32,34.2]) )
hgt=tuple(np.array([340,220,222,333,444,555,666,777,888,1111]) )
yt5=tuple( np.array([1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9,10]) )
dy={'yt':yt, 'ssp':ssp, 'pa':pa, 'hgt':hgt, 'yt5':yt5}
data_y=pd.DataFrame(dy)
print(data_y)
xy=data_y[['yt','ssp','pa','hgt']]
yy=data_y[['yt5']]

#2.建模
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
#核心代码
x_train2,x_test2,y_train2,y_test2=train_test_split(xy,yy,random_state=1)#分割train-set test-set
print(x_train2.shape)
print(x_test2.shape)
lr2=LinearRegression()
lr2.fit(x_train2,y_train2)#开始拟合
print(lr2.intercept_)     #截距
print(lr2.coef_)          #系数矩阵
#accept2=input('input the 4 datas to predict y :\n').split()#分割字符串
#print(fun(lr2.intercept_[0],lr2.coef_,accept2))


#实战预测并绘制图片
print('input the x0 y0 sst pa hgt:\n')
Accept=input().split()       #分割字符串
x_sets=[]
y_sets=[]                    #list
#初始化台风生成向量v1=(xt,sst,pa,hgt)   v2=(yt,sst,pa,hgt) 
x_sets.append(eval(Accept[0]))
y_sets.append(eval(Accept[1]))  
for i in range(2,5):
    x_sets.append(eval(Accept[i]))
    y_sets.append(eval(Accept[i]))

x_result = []               #用于最终绘制散点图
y_result = []
for i in range(200):        #进行迭代生成散点序列
    XT5=fun(lr1.intercept_[0],lr1.coef_,x_sets)
    YT5=fun(lr2.intercept_[0],lr2.coef_,y_sets)
    x_result.append( XT5 )
    y_result.append( YT5 )
    #需要完善向量(xT,sst,pa,hgt) sst pa hgt 均随时间变化
    x_sets[0]=XT5
    y_sets[1]=x_sets[1]= preSST.sst_predict_fun(preSST.lr3.intercept_, preSST.lr3.coef_[0], x_sets[1]) # 调用另一个文件的函数用x_sets[1]来预测下一时刻的SST
    #y_sets[2]=x_sets[2]= prePa.pa_predict_fun  (prePa.lr4.intercept_,  prePa.lr4.coef_[0],  x_sets[2])
    #y_sets[3]=x_sets[3]= preHgt.hgt_predict_fun(preHgt.lr5.intercpt_,  preHgt.lr5.coef_[0], x_sets[3])
    y_sets[0]=YT5


#绘制图片
import matplotlib.pyplot as pl
pl.title('Typhoon track prediction')
pl.xlabel('X coordinate')
pl.ylabel('Y coordinate')
pl.scatter(x_result,y_result)#画布散点显示
pl.tick_params(axis='both', which='major', labelsize=14)
pl.plot(x_result,y_result)   #连线
pl.show()