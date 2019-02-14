# -*-coding:utf-8 -*-
import tensorflow as tf
import input_data

# 1.读取数据，训练值：目标值
# 2.构建模型（卷积神经网络）
# 3.计算损失，优化模型，计算精度
# 4.保存模型，可视化

"""自定义命令行"""
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("flag_num",1,"1代表训练，0代表测试")

"""定义超参数"""
step_size = 1000
batch_size = 500
learning_rate = 0.001
test_size = 1000

# 获取数据
mnist = input_data.read_data_sets("mnist_data",one_hot=True)

# 定义权重变量函数
def weight_variable(shape):
    w = tf.Variable(tf.random_normal(shape=shape,mean=0.0,stddev=1.0,dtype=tf.float32))
    return w

def bias_variable(shape):
    b = tf.Variable(tf.constant(0.0,shape=shape,dtype=tf.float32))
    return b

# 定义偏置变量函数


# 构建模型
def cnn_model():
    # 定义占位符数据
    with tf.variable_scope("train_data"):
        # 训练集中给出的每个样本是维度[784]的向量
        x_train = tf.placeholder(tf.float32,[None,784],"x_train")
        y = tf.placeholder(tf.float32,[None,10],"y_trian")

    # 对输入的data进行reshape后开始卷积
    # 对x进行形状的改变[None, 784]  [None, 28, 28, 1]
    x_data = tf.reshape(x_train,[-1,28,28,1])   # 每批次得到数据量不确定，采用-1代替

    # 定义第一层卷积
    with tf.variable_scope("conv1"):
        # 定义卷积权重（观察窗口的大小为5x5，通道为1 ，个数为32）
        w1 = weight_variable([5,5,1,32])   #
        b1 = bias_variable([32])

        # 先利用w1对x_data进行卷积，然后加上偏置后进行relu函数激活
        # 观察窗口大小; w1[5,5,1,32] s=1 padding ="SAME"  卷积后的x_data --- [-1,28,28,32]
        x_conv1 = tf.nn.relu(tf.nn.conv2d(x_data,w1,strides=[1,1,1,1],padding="SAME")+ b1)
        # 对卷积激活后的函数进行池化，池化后 [-1,14,14,32]
        x_pool1 = tf.nn.max_pool(x_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 定义第二层卷积
    # 第一次卷积后x的shape是 [-1,14,14,32],所以第二次的filter的维数(通道)必须是32
    with tf.variable_scope("conv2"):
        w2 = weight_variable([5,5,32,64])
        b2 = bias_variable([64])

        # 先利用w2对x_pool1进行卷积，然后加上偏置后进行relu函数激活
        # 观察窗口大小; w1[5,5,32,64] s=1 padding ="SAME"  卷积后的x_data --- [-1,14,14,64]
        x_conv2 = tf.nn.relu(tf.nn.conv2d(x_pool1,w2,strides=[1,1,1,1],padding="SAME") + b2)
        # 对卷积激活后的函数进行池化，池化后 [-1,7,7,64]
        x_pool2 = tf.nn.max_pool(x_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

    # 建立全连接层
        with tf.variable_scope("fc"):
            # 定义全连接层的参数
            w_fc = weight_variable([7*7*64,10])
            b_fc = bias_variable([10])

        # 对经过卷积后的数据进行reshape，能够与权重参数进行矩阵运算 y = x*w + b
        x_data_reshape = tf.reshape(x_pool2,[-1,7*7*64])

        # 构建模型进行预测,得到的y_predict的shape是[-1,10],批次大小(-1),每次输出one_hot的10个值
        y_predict = tf.add(tf.matmul(x_data_reshape,w_fc),b_fc)


        return x_train,y,y_predict

# 定义手写识别的主程序
def reco_hand():
    # 获取返回数据 x_train用于作为训练模型的输入参数，y用作数据样本的真实值来计算训练精度，y_predict作为预测值计算精度

    x_train ,y, y_predict = cnn_model()

    # 计算模型训练损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=y_predict))

    # 优化损失
    opt = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # 打印训练精度
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_predict,1)),tf.float32))

    # 收集变量 单个数字值收集
    tf.summary.scalar("losses", loss)
    tf.summary.scalar("accuracy", acc)

    # # 高纬度变量收集
    # tf.summary.histogram("w1", w1)
    # tf.summary.histogram("w2", w2)
    # tf.summary.histogram("b1", b1)
    # tf.summary.histogram("b2", b2)

    # 定义一个合并变量的op：进行变量合并
    merged = tf.summary.merge_all()

    # 定义模型保存
    Saver = tf.train.Saver()

    # 初始化变量
    var = tf.global_variables_initializer()

    #开启会话
    with tf.Session() as sess:
        sess.run(var)
        # 可视化tensorboard
        filewriter = tf.summary.FileWriter("./temp_tensor", graph=tf.get_default_graph())
        # 可视化
        if FLAGS.flag_num ==1:
            for i in range(step_size):
                x_batch,y_batch = mnist.train.next_batch(batch_size)

                _,loss_data = sess.run([opt ,loss],feed_dict = {x_train:x_batch,y:y_batch})
                accuracy = sess.run(acc,feed_dict = {x_train:x_batch,y:y_batch})

                print("第%d训练后,训练精度是%f,模型损失是:%f"%(i,accuracy,loss_data))

                # 运行合并变量op
                summary = sess.run(merged, feed_dict={x_train: x_batch, y: y_batch})
                # 写入每步训练的值
                filewriter.add_summary(summary, i)

                # 保存模型参数(每训练一次，保存一次模型)
                Saver.save(sess, "./temp/ckpt/")
        else:
            # 先加载训练保存好的模型参数
            Saver.restore(sess, "./temp/ckpt/")

            for i in range(test_size):
                x_test, y_test = mnist.test.next_batch(1)

            # 将test集数据传入训练模型得到预测值，[-1,10],对axis=1维度的one-hot取最大值，并通过eval()函数取tensor中的值
                y_pred= tf.argmax(sess.run(y_predict, feed_dict={x_train: x_test, y: y_test}),1).eval()

                y_label = tf.argmax(y_test,1).eval()

                print("第%d次测试，预测结果是%d,实际标签是%d"%(i,y_pred,y_label))


if __name__ == "__main__":
    reco_hand()
