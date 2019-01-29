# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 构造数据
# 构造模型
# 计算损失
# 优化
# 执行会话
# 可视化

"""设置超参"""
# 模型参数
learning_rate = 0.01
step_num = 1000
# 模型参数
h1_num = 10
h2_num = 10
activate_func = tf.nn.tanh
"""构造数据"""
x_data = np.linspace(-2, 2, 500, dtype=np.float32)[:, np.newaxis]
bias = np.random.normal(0, 0.5, x_data.shape).astype(np.float32)
y_true = np.add(
    np.add(
        np.multiply(
            5, np.power(
                x_data, 2)), np.multiply(
                    4, x_data)), bias)               # 构造一个f(x)= ax*x+bx+c的二次函数

"""构造计算图"""
# 构造数据输入的参数；占位符
x_batch = tf.placeholder(tf.float32, [None, 1], name="x_batch")
y_batch = tf.placeholder(tf.float32, [None, 1], name="y_batch")

# 构造模型

def net_layer(input, insize, outsize, activate_function=None):
    # 初始化参数
    w = tf.Variable(tf.random_normal([insize, outsize]), name="w")
    b = tf.Variable(tf.truncated_normal(
        [1, outsize], mean=0, stddev=0.01, name="b"))
    # 构造
    if activate_function is None:
        output = tf.add(tf.matmul(input, w), b)
    else:
        output = activate_function(tf.add(tf.matmul(input, w), b))

    return output


# 添加神经网络
h1 = net_layer(x_batch, 1, h1_num, activate_function=activate_func)   # 激活函数选用
h2 = net_layer(h1, h1_num, h2_num, activate_function=activate_func)
y_pred = net_layer(h2, h2_num, 1)

# 计算损失
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_pred - y_true), axis=1))

# 梯度下降计算损失，迭代优化，得到参数
opt = tf.train.GradientDescentOptimizer(
    learning_rate=learning_rate).minimize(loss)

saver = tf.train.Saver()  # 实例化存储数据的类

init_op = tf.global_variables_initializer()  # 初始化变量


# 执行会话
with tf.Session() as sess:
    sess.run(init_op)     # 运行变量
    for i in range(step_num):
        _, loss_data = sess.run([opt, loss], feed_dict={
                                x_batch: x_data, y_batch: y_true})
        print(loss_data)
        result = sess.run(y_pred, feed_dict={x_batch: x_data})

# 数据保存
    save_path = saver.save(sess, "model/model.ckpt")

# tensorboard
    tf.summary.FileWriter("D:\\hello3", graph=tf.get_default_graph())

    """可视化"""
    plt.scatter(x_data,y_true)
    plt.plot(x_data,result,color="r")
    plt.show()





