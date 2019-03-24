
# coding: utf-8

# 比较CPU与GPU性能

# （1）设备设为GPU

# In[3]:

import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name="/gpu:0"

shape=(int(10000),int(10000))

with tf.device(device_name):
    #形状为shap,元素服从minval和maxval之间的均匀分布
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)

print("\n" * 2)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)


# （2）把设备改为CPU

# In[2]:

import sys
import numpy as np
import tensorflow as tf
from datetime import datetime

device_name="/cpu:0"

shape=(int(10000),int(10000))

with tf.device(device_name):
    #形状为shap,元素服从minval和maxval之间的均匀分布
    random_matrix = tf.random_uniform(shape=shape, minval=0, maxval=1)
    dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
    sum_operation = tf.reduce_sum(dot_operation)

startTime = datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)
        print(result)

print("\n" * 2)
print("Shape:", shape, "Device:", device_name)
print("Time taken:", datetime.now() - startTime)


# 这个实例运算较简单，但即使如此，GPU也是CPU的近3倍

# 实例：单GPU与多GPU性能

# In[8]:

'''
"/cpu:0": 表示CPU
"/gpu:0": 表示第1块 GPU
"/gpu:1":表示第2块 GPU
'''

import numpy as np
import tensorflow as tf
import datetime

# 处理单元日志
log_device_placement = True

# 设置执行乘法次数
n = 10
'''
实例: 在2个GPU上执行： A^n + B^n 
 '''
# 创建随机矩阵
A = np.random.rand(10000, 10000).astype('float32')
B = np.random.rand(10000, 10000).astype('float32')

# 创建两个变量
c1 = []
c2 = []

def matpow(M, n):
    if n < 1: 
        return M
    else:
        return tf.matmul(M, matpow(M, n-1))
'''
使用单个GPU计算
'''
with tf.device('/gpu:0'):
    a = tf.placeholder(tf.float32, [10000, 10000])
    b = tf.placeholder(tf.float32, [10000, 10000])
    # 计算 A^n and B^n 并把结果存储在 c1中
    c1.append(matpow(a, n))
    c1.append(matpow(b, n))

with tf.device('/cpu:0'):
  sum = tf.add_n(c1) #Addition of all elements in c1, i.e. A^n + B^n

t1_1 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum, {a:A, b:B})
t2_1 = datetime.datetime.now()

'''
使用多个GPU进行计算
'''
# 在GPU:0上计算 A^n
with tf.device('/gpu:0'):
    # 计算 A^n 并把结果存储在 c2
    a = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(a, n))

# GPU:1 computes B^n
with tf.device('/gpu:1'):
    # 计算 B^n并把结果存储在 c2
    b = tf.placeholder(tf.float32, [10000, 10000])
    c2.append(matpow(b, n))

with tf.device('/cpu:0'):
  sum = tf.add_n(c2) #对c2中的元素进行累加, 如A^n + B^n

t1_2 = datetime.datetime.now()
with tf.Session(config=tf.ConfigProto(log_device_placement=log_device_placement)) as sess:
    # Run the op.
    sess.run(sum, {a:A, b:B})
t2_2 = datetime.datetime.now()

print("Single GPU computation time: " + str(t2_1-t1_1))
print("Multi GPU computation time: " + str(t2_2-t1_2))



# 由此可见，使用多GPU计算比使用单个GPU计算，快近一倍。这里数据不大，如果数据比较大，差别会更明显

# In[ ]:



