# -*- coding: utf-8 -*-
import  tensorflow as tf
# 1.读取数据集，获取训练样本

# 2.构建训练模型(神经网络)，损失函数，优化模型

# 3.传入数据进行训练

# 4.保存模型，可视化处理

### 5.加载模型，进行测试

# 自定义命令行
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("path","./tfrecords/captcha.tfrecords","数据集的存储路径")
tf.app.flags.DEFINE_integer("batch_size",10,"每批次处理的数据量")
tf.app.flags.DEFINE_integer("letter_num",26,"one_hot编码的深度")
tf.app.flags.DEFINE_integer("label_num",4,"one_hot编码的深度")

# 定义初始化权重变量
def weight_varible(shape):
    weight_varible = tf.Variable(tf.random_normal(shape=shape,mean=0.0,stddev=1.0,dtype=tf.float32))
    return weight_varible

def bias_varible(shape):
    bias_varible = tf.Variable(tf.constant(0.0,shape=shape))
    return bias_varible

def get_data():

    """从tf.reconds中获取图片数据"""
    # 1. 构建文件队列，获取文件列表
    file_queue = tf.train.string_input_producer([FLAGS.path])

    # 2.构建阅读，读取文件中的内容，默认按照每一个样本进行读取。
    reader  = tf.TFRecordReader()

    # 3.读取样本数据
    key ,value = reader.read(file_queue)

    # 4.tf.recondr数据的exmple需要进行解析
    # example进行解析之后返回的是一个字典类型的数据
    features = tf.parse_single_example(value,features={
        "image": tf.FixedLenFeature([],tf.string),
        "label": tf.FixedLenFeature([],tf.string)
    })

    # 5.解码内容：对解析之后的数据进行解码处理，指定数据类型
    image = tf.decode_raw(features["image"],tf.uint8)
    label = tf.decode_raw(features["label"],tf.uint8)

    # 6.读取出来的图片数据是一个列表，根据实际要求对数据的shape进行处理
    image_reshape = tf.reshape(image,[20,80,3])
    label_reshape  = tf.reshape(label,[4])

    # 7.批处理
    image_batch, label_batch = tf.train.batch([image_reshape,label_reshape],batch_size=FLAGS.batch_size,num_threads=1,capacity=10)

    print(image_batch,label_batch)

    return image_batch, label_batch

def  fc_model(image_batch):

    with tf.variable_scope("fc_model"):
        # 对图片数据的shape进行处理,使得能够进行矩阵运算。
        image_reshape  = tf.cast(tf.reshape(image_batch,[-1,20*80*3]),tf.float32)

        # 随机初始化权重及偏置
        # 根据网络模型 y= w*x+b，分别计算data ,w ,b 的shape
        # datd:[-1 ,20*80*3] -->  w:[20*80*3,4*26]  b:[4*26]  y = [-1,4*26]
        w = weight_varible([20*80*3,4*26])
        b = bias_varible([4*26])

        # 收集高维变量
        tf.summary.histogram("weight", w)
        tf.summary.histogram("bias",b)

        # 进行全连接层计算
        y_pred = tf.add(tf.matmul(image_reshape,w),b)
        print("y_pred",y_pred)
    return y_pred

def value_one_hot(label):
    """
       将读取文件当中的目标值转换成one-hot编码
       :param label: [100, 4]      [[13, 25, 15, 15], [19, 23, 20, 16]......]
       :return: one-hot
    """
    label_one_hot = tf.one_hot(label, depth=FLAGS.letter_num , on_value=1)

    return label_one_hot

def reco_valid():
    # 接受训练样本数据
    image_batch, label_batch = get_data()

    # 得到数据集中的真实值的one-hot编码（传入数据集中的标签值）
    y_true = value_one_hot(label_batch)

    # 得到模型训练后的预测值（传入训练样本x）
    y_predict = fc_model(image_batch)

    with tf.variable_scope("loss"):

        # 对预测值进行softmax()处理后与真实值求交叉熵,并对交叉熵取均值，得到损失值
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.reshape(y_true, [FLAGS.batch_size, FLAGS.label_num * FLAGS.letter_num]),
                logits=y_predict))

    with tf.variable_scope("optimizer"):
        # 优化损失
        opt = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    with tf.variable_scope("acc"):
        # 计算精度
        # # 比较每个预测值和目标值是否位置一样    y_predict [100, 4 * 26]---->[100, 4, 26]
        compa_list = tf.equal(tf.argmax(y_true,2), tf.argmax(tf.reshape(y_predict, [FLAGS.batch_size, FLAGS.label_num, FLAGS.letter_num]), 2))

        acc = tf.reduce_mean(tf.cast(compa_list, tf.float32))

        # 定义一个初始化变量的op
        init_op = tf.global_variables_initializer()

        saver = tf.train.Saver()   # 保存模型

        # 收集低纬变量
        tf.summary.scalar("loss",loss)
        tf.summary.scalar("accuracy",acc)

        merged = tf.summary.merge_all()

    # 开启会话
    with tf.Session() as sess:

        sess.run(init_op)
        # 1.定义线程协调器并开启线程
        coord = tf.train.Coordinator()

        # 2. 开启线程进行文件读取
        threads = tf.train.start_queue_runners(sess, coord=coord)

        # 建立events文件，然后写入
        filewriter = tf.summary.FileWriter("./tmp/test/", graph=sess.graph)

        # 开始训练程序
        for i in range(5000):
            sess.run(opt)

            # print("第%d批次的准确率为：%f" % (i, acc.eval()))   # 获取训练精度，acc是一个张量，必须要执行eval()获取其值

            summary = sess.run(merged)

            filewriter.add_summary(summary,i)

        # 保存模型  ,给出文件保存路径（相对路径）
        saver.save(sess, "./ckpt/recognize_validation")

        # 可视化模型


        # 3.回收线程
        coord.request_stop()
        coord.join(threads)

    return None



if __name__ =="__main__":
    reco_valid()



