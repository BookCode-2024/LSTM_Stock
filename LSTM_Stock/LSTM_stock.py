# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:22:15 2023

@author: zhaodf
"""

#采用LSTM对TATAGLOBAL的股票价格（开盘价）进行预测

#Tensorflow1.13.1 and Keras2.2.4

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

"""
    数据预处理
        1， 读取股票数据。
        2， 对数据进行归一化。
        3， 生成训练集数据
"""
# 1, 读取训练集数据
dataset_train = pd.read_csv('dataset/NSE-TATAGLOBAL.csv') # (2035, 8) 读取.csv表格中的数据
training_set = dataset_train.iloc[:, 1:2].values # (2035, 1) # 获取表格中第二列上的数据值,股票开盘价


# 2， 归一化操作
sc = MinMaxScaler(feature_range=(0, 1)) # 将数据归一化到0~1之间
training_set_scaled = sc.fit_transform(training_set) # 对训练集数据进行归一化

# 3， 生成训练集数据
# 设置一个样本为：60个数据点预测1个数据点
X_train = []
Y_train = []

for i in range(60, 2035):
    X_train.append(training_set_scaled[i-60:i, 0])
    Y_train.append(training_set_scaled[i, 0])

X_train, Y_train = np.array(X_train), np.array(Y_train) # (1975, 60)   (1975, 1)  将list形式转换成矩阵形式

# 转化数据格式与LSTM网络输入格式匹配
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # (1975, 60, 1)


"""
    构建LSTM模型用于训练。
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=20, return_sequences=True, input_shape=(60, 1))) # 当前LSTM层保留每个时刻的输出值
# model.add(tf.keras.layers.Dropout(0.2)) # 该层网络每次训练时，20%神经元失活
model.add(tf.keras.layers.LSTM(units=20, return_sequences=True))
# model.add(tf.keras.layers.Dropout(0.2))
# model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
# model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.LSTM(units=10, return_sequences=False)) # 当前LSTM层只输出最后一个时刻的输出值
# model.add(tf.keras.layers.Dropout(0.2))

# 将上述LSTM层的最后一个时刻的输出接到全连接层。
# 用于数据预测，故不使用BN层与softmax激活函数。
model.add(tf.keras.layers.Dense(16))
model.add(tf.keras.layers.Dense(1))

model.build(input_shape=(None, 60, 1))
model.summary() # 显示网络的架构


"""
    训练LSTM模型
"""
# 绘制训练集与测试集的loss曲线图
def show_train_history(train_history,train,validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.xlabel('epoch')
    plt.ylabel(train)
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# 模型训练参数的配置
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.mse) # 采用均方差损失

# 模型开始训练
history = model.fit(x=X_train, y=Y_train, batch_size=32, epochs=100, verbose=1, validation_split=0.2)

figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
show_train_history(history, 'loss', 'val_loss') # 绘制loss曲线


"""
    进入测试阶段。测试集的股票数据是训练集股票数据的后续
        1， 读取测试数据。
        2， 建立测试集
        2， 使用模型预测数据。
        3， 绘制真实股票曲线与预测股票曲线。
"""
# 1， 读取数据
dataset_test = pd.read_csv('dataset/tatatest.csv') # 读取测试上的数据
real_stock_price = dataset_test.iloc[:, 1:2].values # (16, 1)

# 2， 建立测试集
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values # 结果为训练集上的最后60个数据点与测试集上所有的数据点

inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs) # (76, 1) 对输入数据进行归一化

X_test = []

for i in range(60, 76):
    X_test.append(inputs[i - 60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 3， 预测股票曲线
predicted_stock_price = model.predict(X_test) # 进行预测数据
predicted_stock_price = sc.inverse_transform(predicted_stock_price) # 反归一化

# 3， 进行结果的可视化
figsize = 7,5
figure, ax = plt.subplots(figsize=figsize)
plt.plot(real_stock_price, color='red', label='Real TATA Stock Price')
plt.plot(predicted_stock_price, color='blue', label='Predicted TATA Stock Price')
plt.title('Time')
plt.ylabel('TAT Stock Price')
plt.legend()
plt.show()
