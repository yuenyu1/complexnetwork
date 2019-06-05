#coding=utf-8
__author__ = '94353'

import tensorflow as tf
import numpy as np

#生成数据集
data_train1 = np.load('data_train_D_0.05_1.npy')
data_test = np.load('data_TSF_D_0.05_1.npy')
train_X = []
train_y = []
test_X = []
test_y = []
for x in data_train1:
    train_X.append(x[1:31])
    train_y.append([x[31]])
for x in data_test:
    test_X.append(x[1:31])
    test_y.append([x[31]])
train_X = np.array(train_X, dtype=np.float32)
train_y = np.array(train_y, dtype=np.float32)
test_X = np.array(test_X, dtype=np.float32)
test_y = np.array(test_y, dtype=np.float32)


#定义神经网络参数
w1 = tf.Variable(tf.random_normal((30, 32), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((32, 1), stddev=1, seed=1))


x = tf.placeholder(tf.float32, shape=(None, 30), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

#前向传播
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#损失函数和反向传播
mse = tf.reduce_mean(tf.square(y_-y))
train_step = tf.train.AdamOptimizer(0.001).minimize(mse)


#创建一个会话来运行tensorflow程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    #初始化变量
    sess.run(init_op)

    # print(sess.run(w1))
    # print(sess.run(w2))

    #设定训练的轮数
    STEPS = 3000
    for i in range(STEPS):
        #每次选区batchsize的样本进行训练
        sess.run(train_step, feed_dict = {x:train_X, y_:train_y})

        if i % 1000 == 0:
            #每隔一段时间计算交叉熵并输出
            total_mse = sess.run(mse, feed_dict={x: train_X, y_: train_y})
            print('After %d training step(s), mse on all data is %g' % (i, total_mse))

    # print(sess.run(w1))
    # print(sess.run(w2))
    predict = sess.run(y, feed_dict = {x:test_X, y_:test_y})
    re = []
    real = []
    for i in range(len(data_test)):
        re.append((data_test[i][0], predict[i]))
        real.append((data_test[i][0], data_test[i][-1]))
    re = [x[0] for x in sorted(re, key=lambda x:x[1], reverse=True)]
    real = [x[0] for x in sorted(real, key=lambda x:x[1], reverse=True)]
    print(len(set(re[:25])&set(real[:25]))/25.0)
    print(len(set(re[:50])&set(real[:50]))/50.0)
