def sst_predict_fun(b,k,sst0):#拟合出来的SST预测函数
    return k*sst0+b
import numpy as np
from  sklearn.linear_model import LinearRegression
sst=np.array([[1],[2],[3]])#人为构造二维数组
sst5=np.array([2.4,4.6,6.8])
lr3=LinearRegression()
lr3.fit(sst,sst5)                        #需要穿进去二维数组












# print( lr3.intercept_ )
# print( lr3.coef_ )
# sst0=input('input the sst0:\n')
# print( sst_predict_fun(lr3.intercept_,lr3.coef_[0],eval(sst0) ) )