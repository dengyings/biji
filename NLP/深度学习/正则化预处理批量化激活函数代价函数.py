
# coding: utf-8

# 梯度裁剪如何实现呢？TensorFlow提供了一个函数：tf.clip_by_value,以下是示例代码

# In[ ]:

gradients = optimizer.compute_gradients(loss, var_list)  
capped_gradients = [(tf.clip_by_value(grad, -limit., limit.), var) for grad,var in gradients if grad is not None else (None,var)]  
train_op = optimizer.apply_gradients(capped_gradients)  


# 在程序中如何根据迭代次数或精度提前终止呢? 根据迭代次数比较好处理，只要用循环语句就可，这里仅使用TensorFlow实现根据精度提前终止的部分代码，第20章有详细代码。

# In[ ]:

# 获取测试数据的准确率
acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
# 当准确率大于0.98时保存并退出
if acc > 0.98 :
    saver.save(sess, './train_face_model/train_faces.model')


# 如何用TensorFlow来实现dropout呢? 以下为实现dropout的部分代码：

# In[ ]:

# 第一个卷积层+池化
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二个卷积层+池化
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 连一个全连接层
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout层
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax层
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Adam优化器＋cross entropy＋小学习速率
cross_entropy =tf.reduce_mean(-tf.reduce_sum(y_*tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)


# BN如何用TensorFlow来实现呢? 以下是具体实现代码

# In[ ]:

scale = tf.Variable(tf.ones([1]))    #对应式(7.12)的γ
shift = tf.Variable(tf.zeros([1]))   #对应式(7.12)的β
epsilon = 0.001                      #对应式(7.11)的ϵ，防止分母为0
xs = tf.nn.batch_normalization(xs, fc_mean, fc_var, shift, scale, epsilon)


# 在配置好GPU环境的TensorFlow中，如果没有明确指定使用GPU，TensorFlow也会优先选择GPU来运算，不过此时默认使用/gpu:0。以下代码为TensorFlow使用GPU样例。

# In[ ]:

device_name="/gpu:0"
shape=(int(10000),int(10000))

with tf.device(device_name):
    #形状为shap,元素服从minval和maxval之间的均匀分布
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)


# 下例TensorFlow代码中使用了交叉熵，在这个代码中说明了使用交叉熵及其注意事项

# In[1]:

import tensorflow as tf  

# 神经网络的输出
logits=tf.constant([[1.0,2.0,3.0],[1.0,2.0,3.0],[1.0,2.0,3.0]])  
# 使用softmax的输出
y=tf.nn.softmax(logits)  
# 正确的标签只要一个1
y_=tf.constant([[0.0,0.0,1.0],[1.0,0.0,0.0],[1.0,0.0,0.0]])  
# 计算交叉熵  
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)))  
# 使用tf.nn.softmax_cross_entropy_with_logits()函数直接计算神经网络的输出结果的交叉熵。
# 记得使用tf.reduce_sum()
cross_entropy2 = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=logits)) 

with tf.Session() as sess:
    softmax=sess.run(y)
    ce = sess.run(cross_entropy)
    ce2 = sess.run(cross_entropy2)
    print("softmax result=", softmax)
    print("cross_entropy result=", ce)
    print("softmax_cross_entropy_with_logits result=", ce2)

