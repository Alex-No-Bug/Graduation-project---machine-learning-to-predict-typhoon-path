import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation

data_SST=[
    [10.1],[10.2],[10.3],[10.4],
    [11.2],[11.3],[11.4],[11.5],
    [12.3],[12.4],[12.5],[12.6],
    [14.5],[14.6],[14.7],[14.8],
    [15.4],[15.6],[15.7],[15.8],
    [16.7],[16.8],[16.9],[17.0],
    [20.1],[20.2],[20.3],[20.4],
    [21.2],[21.3],[21.4],[21.5],
    [22.3],[22.4],[22.5],[22.6],
    [24.5],[24.6],[24.7],[24.8],
    [25.4],[25.6],[25.7],[25.8],
    [26.7],[26.8],[26.9],[27.0]
]
sst0=[ [3100.2],[3100.3],[3100.8] ]

def load_data(sequence_length=4, split=0.5):#sequence_length=4 切片长度
    scaler   = MinMaxScaler()#映射到0,1区间
    data_all = scaler.fit_transform(data_SST)#二维数组
    sst1     = scaler.fit_transform(sst0)

    data = []
    for i in range(len(data_all) - sequence_length - 1):
        data.append(data_all[i: i + sequence_length + 1])
    reshaped_data = np.array(data).astype('float64')
    print('shape:\n')
    print(reshaped_data.shape)
    #np.random.shuffle(reshaped_data)#将数据集随机打乱
    #竖着切一刀
    x = reshaped_data[:, :-1]#用前9个数据预测最后一个
    y = reshaped_data[:, -1]
    split_boundary = int(reshaped_data.shape[0] * split)
    #横着切一刀
    train_x = x[: split_boundary]
    test_x = x[split_boundary:]
    train_y = y[: split_boundary]
    test_y = y[split_boundary:]
    return train_x, train_y, test_x, test_y, scaler ,sst1


#搭建神经网络
def build_model():
    # input_dim是输入的train_x的最后一个维度，train_x的维度为(n_samples, time_steps, input_dim)
    model = Sequential()
    model.add(LSTM(input_dim=1, output_dim=50, return_sequences=True))
    print(model.layers)
    model.add(LSTM(100, return_sequences=False))
    model.add(Dense(output_dim=1))
    model.add(Activation('linear')) #激励函数
    model.compile(loss='mse', optimizer='rmsprop')
    return model

def train_model(train_x, train_y, test_x, test_y,sst1):
    model = build_model()
    try:
        model.fit(train_x, train_y, batch_size=512, nb_epoch=30, validation_split=0.1)
        print('test_x.shape:\n')
        print(test_x.shape)
        print(test_x)
        sst1=sst1.reshape(1,3,1)   #注意这里要改成
        predict     = model.predict(test_x)
        predict_sst = model.predict(sst1)

        predict = np.reshape(predict, (predict.size, ))
        predict_sst=np.reshape(predict_sst,(predict_sst.size,))
    except KeyboardInterrupt:
        print(predict)
        print(test_y)
    try:
        fig = plt.figure(1)
        plt.plot(predict, 'r:')
        plt.plot(test_y, 'g-')
        plt.legend(['predict', 'true'])
    except Exception as e:
        print(e)
    return predict, test_y, predict_sst



train_x, train_y, test_x, test_y, scaler,sst1 = load_data()
train_x = np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))#将其变换为三维数组
test_x  = np.reshape(test_x,  (test_x.shape[0],  test_x.shape[1],  1))
sst1    = np.reshape(sst1 ,   (sst1.shape[0]  ,  sst1.shape[1],    1))
predict_y, test_y,predict_sst1= train_model(train_x, train_y, test_x, test_y,sst1)
predict_sst1 =scaler.inverse_transform([[i] for i in predict_sst1])      #反归一化
predict_y   = scaler.inverse_transform([[i] for i in predict_y])

test_y = scaler.inverse_transform(test_y)
print('predict_y:\n')
print(predict_y)
print('predict_sst:\n')
print(predict_sst1)

fig2 = plt.figure(2)
plt.plot(predict_y, 'g:')
plt.plot(test_y, 'r-')
plt.show()



