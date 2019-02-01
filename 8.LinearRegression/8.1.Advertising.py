#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


if __name__ == "__main__":
    path = 'E:\\data\\8.Advertising.csv'
    # # 手写读取数据 - 请自行分析，在8.2.Iris代码中给出类似的例子
    # f = file(path)
    # x = []
    # y = []
    # for i, d in enumerate(f):  # 采用enumerate函数，通过枚举的方式获取得到f中的value及索引值
    #     if i == 0:
    #         continue
    #     d = d.strip()  #strip方法去掉文本中空白的部分
    #     if not d:
    #         continue
    #     d = map(float, d.split(','))  # map(function, iterable, ...)  function -- 函数,iterable -- 一个或多个序列
                                        # python2.x 返回列表，3.x返回迭代对象。
    #     x.append(d[1:-1])   # 采用append方法将d中从第二个数据到倒数第二个数据进行追加（左闭右开）
    #     y.append(d[-1])     # 将最后一个数据append
    # print x
    # print y
    # x = np.array(x)    # 将x列表转换为数组
    # y = np.array(y)

    # # Python自带库
    # f = file(path, 'rb')   # 文件读取file(路径，读取方式)
    # print f
    # d = csv.reader(f)     # 采用python自带的csv库进行读取
    # for line in d:        # 遍历每行
    #     print line
    # f.close()             # 关闭文件

    # # numpy读入
    # p = np.loadtxt(path, delimiter=',', skiprows=1)   # 采用loadtxtf方式读取txt文件（传入路径，分隔符，忽略第一行)
    # print p

    # pandas读入
    data = pd.read_csv(path)    # TV、Radio、Newspaper、Sales
    x = data[['TV', 'Radio', 'Newspaper']]   # 读取对应的col的数据
    # x = data[['TV', 'Radio']]
    y = data['Sales']
    # print (x)
    # print (y)

    # # 绘制1
    # plt.plot(data['TV'], y, 'ro', label='TV')
    # plt.plot(data['Radio'], y, 'g^', label='Radio')
    # plt.plot(data['Newspaper'], y, 'mv', label='Newspaer')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()
    # #
    # 绘制2
    # plt.figure(figsize=(12, 12))
    # plt.subplot(311)     # 3x1的子图，并取第一个
    # plt.plot(data['TV'], y, 'ro')   # 以data['TV']数据为X轴，绘制图形
    # plt.title('TV')
    # plt.grid()
    # plt.subplot(312)
    # plt.plot(data['Radio'], y, 'g^')
    # plt.title('Radio')
    # plt.grid()
    # plt.subplot(313)
    # plt.plot(data['Newspaper'], y, 'b*')
    # plt.title('Newspaper')
    # plt.grid()
    # plt.tight_layout()  # 图形紧凑排列在figures中
    # plt.savefig("your_image.svg")
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1,train_size=0.8)   # 分割数据集 设置0.8分割点
    print("X_data:", x_train)
    print("Y_data:", y_train)
    linreg = LinearRegression()     # 调用线性回归模型
    model = linreg.fit(x_train, y_train)   # 对训练集的数据进行拟合（线性回归）
    print("model:", model)
    print("coef:", linreg.coef_)     # 打印出各相关系数
    print("intercept:", linreg.intercept_)     # 打印截距
    #
    y_hat = linreg.predict(np.array(x_test))   # 将测试集数据带入回归方程中得到预测值
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error(将预测值与测试集中的真实值构建均方误差)
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print("MSE:", mse)
    print("RMSE:", rmse)
    #
    t = np.arange(len(x_test))  # 以测试集的数据长度（个数）为横坐标（通过arange方法返回一个adarray结构的数组）
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
