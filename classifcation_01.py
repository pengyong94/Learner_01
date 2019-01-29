# -*- coding: utf-8 -*-
import tensorflow as tf
import input_data      # 导入数据

"""定义超参"""
# 定义训练参数
learning_rate = 0.01
batch_size = 5000
step_num = 1000
# 定义模型参数
h1_num = 50
h2_num = 50
activate_func = tf.nn.relu
"""构造图"""
# 数据集中给出了6万张[784,]的数据
mnist = input_data.read_data_sets("mnist_data",one_hot=True)

# 构造图
x_batch = tf.placeholder(tf.float32,[None,784],"x_batch")
y_batch = tf.placeholder(tf.float32,[None,10],"x_batch")  # 预测值为0-9的10个数

# 定义神经网络
def net_layer(input,insize,outsize ,activate_function):
    w = tf.Variable(tf.random_normal([insize,outsize]),name="w")
    b = tf.Variable(tf.add(tf.zeros([1,outsize]),0.1),name="b")

    if activate_function is None:
        output = tf.add(tf.matmul(input,w),b)
    else:
        output = activate_function(tf.add(tf.matmul(input,w),b))
    return  output

# 构造网络
h1 = net_layer(x_batch,784,h1_num,activate_function=activate_func)
h2 = net_layer(h1,h1_num,h2_num,activate_function=activate_func)

pre = net_layer(h2,h2_num,10,activate_function=tf.nn.softmax)

"""计算损失"""
loss = tf.losses.softmax_cross_entropy(y_batch,pre)
# 优化
opt= tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 实例化数据保存函数
saver = tf.train.Saver()

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    for i in range(step_num):
        x_date ,y_data = mnist.train.next_batch(batch_size)
        _,loss_data = sess.run([opt,loss],feed_dict={x_batch:x_date,y_batch:y_data})
        print(loss_data)

    # 数据保存
    save_path = saver.save(sess, "model_01/model.ckpt")

    # tensorboard
    tf.summary.FileWriter("D:\\hello2", graph=tf.get_default_graph())