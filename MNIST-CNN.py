##使用tensorflow实现一个简单的卷积神经网络
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("C:/MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()

##定义初始化函数，以便反复使用
def weights_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1) #给权重制造一些随机的噪声来打破完全对称，比如截断的正态分布噪声，标准差设为0.1
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)         ##因为使用ReLU，需要给偏置加上0.1用来避免死亡节点
    return tf.Variable(initial)

##定义卷积层与池化层函数
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
##strides代表卷积模板移动的步长，全为1表示会不遗漏地划过图片中的每一个点
##padding表示边界的处理方式，SAME表示给边界加上padding，使得卷积的输出与输入保持相同的尺寸
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

##卷积神经网络会用到空间结构信息，需要将1D的输入向量转换为2D的图片结构
x = tf.placeholder(tf.float32,[None,784])    ## x为特征
y_ = tf.placeholder(tf.float32,[None,10])     ## y_为真实的label
x_image = tf.reshape(x,[-1,28,28,1])          ## -1表示样本数量不定，尺寸为28*28,1表示颜色通道（channel）为1

##定义第一个卷积层
w_conv1 = weights_variable([5,5,1,32])     ##参数代表卷积核尺寸为5*5，颜色通道为1，共有32个不同的卷积核（即会提取32种特征）
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1) + b_conv1)     ##使用conv2d进行卷积操作，然后再使用Relu激活函数进行非线性处理
h_pool1 = max_pool_2x2(h_conv1)          ##最后使用最大池化函数对卷积的输出结果进行池化操作

##定义第二个卷积层
w_conv2 = weights_variable([5,5,32,64])      ##第一层卷积过后，输出的图像颜色通道变成了32，即第一层卷积层的卷积核个数
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

##定义全连接层（fc)
w_fc1 = weights_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2 , [-1,7*7*64])
h_fc1 = tf.nn.relu( tf.matmul(h_pool2_flat , w_fc1) + b_fc1 )

##为了减轻过拟合，使用dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

##dropout层的输出连接一个softmax层，得到最后的概率输出
w_fc2 = weights_variable([1024,10])
b_fc2 = bias_variable([10])
h_fc2 = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

##定义损失函数
cross_entropy = tf.reduce_mean(-tf.reduce_sum( y_ * tf.log(h_fc2)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(h_fc2, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##开始训练
tf.global_variables_initializer().run()
for i in range(20000):
    batch = mnist.train.next_batch(50)    ##mini-batch大小为50
    if i%100 ==0:              ##每100次训练就对结果进行一次评测
        train_accuracy = accuracy.eval(feed_dict={x : batch[0] ,y_ : batch[1], keep_prob : 1.0}) #.eval()等同于sess.run(),评测时dropout为1
        print("step %d,accuracy %g"%(i,train_accuracy))
    train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob : 0.5})

print("test accuracy %g" %sess.run(accuracy,feed_dict={x:mnist.test.images , y_:mnist.test.labels,keep_prob : 1.0}))
