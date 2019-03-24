
# coding: utf-8

# 第一章  Numpy常用操作
# 内容简介：
# NumPy是Python基础，更是数据科学的通用语言，而且与TensorFlow关系密切，所以我们把它列为第一章。
# 	NumPy为何如此重要？实际上Python本身含有列表（list）和数组（array），但对于大数据来说，这些结构有很多不足。因列表的元素可以是任何对象，因此列表中所保存的是对象的指针。这样为了保存一个简单的[1,2,3]，都需要有3个指针和三个整数对象。对于数值运算来说这种结构显然比较浪费内存和CPU计算时间。 至于array对象，它直接保存数值，和C语言的一维数组比较类似。但是由于它不支持多维，也没有各种运算函数，因此也不适合做数值运算。 
# 	NumPy（Numerical Python 的简称）的诞生弥补了这些不足，NumPy提供了两种基本的对象：ndarray（N-dimensional array object）和 ufunc（universal function object）。ndarray是存储单一数据类型的多维数组，而ufunc则是能够对数组进行处理的函数
# 

# 1.1生成ndarray的几种方式

# 1.从已有数据中创建

# In[3]:

import numpy as np
list1 = [3.14,2.17,0,1,2]
nd1 = np.array(list1)
print(nd1)
print(type(nd1))


# In[4]:

list2 = [[3.14,2.17,0,1,2],[1,2,3,4,5]]
nd2 = np.array(list2)
print(nd2)
print(type(nd2))


# 2.利用random模块生成ndarray

# In[5]:

import numpy as np

nd5 = np.random.random([3,3])
print(nd5)
print(type(nd5))


# In[6]:

import numpy as np

np.random.seed(123)
nd5_1 = np.random.randn(2,3)
print(nd5_1)
np.random.shuffle(nd5_1)
print("随机打乱后数据")
print(nd5_1)
print(type(nd5_1))


# 3.创建特定形状的多维数组

# In[7]:

import numpy as np

#生成全是0的3x3矩阵
nd6 = np.zeros([3,3])
#生成全是1的3x3矩阵
nd7 = np.ones([3,3])
#生成3阶的单位矩阵
nd8= np.eye(3)
#生成3阶对角矩阵
print (np.diag([1, 2, 3]))


# In[8]:

import numpy as np
nd9 = np.random.random([5,5])
np.savetxt(X=nd9,fname='./test2.txt')
nd10 = np.loadtxt('./test2.txt')


# 4.利用arange函数

# In[9]:

import numpy as np

print(np.arange(10))
print(np.arange(0,10))
print(np.arange(1, 4,0.5))
print(np.arange(9, -1, -1))


# 1.2存取元素

# In[10]:

import numpy as np
np.random.seed(2018)
nd11 = np.random.random([10])
#获取指定位置的数据，获取第4个元素
nd11[3]
#截取一段数据
nd11[3:6]
#截取固定间隔数据
nd11[1:6:2]
#倒序取数
nd11[::-2]
#截取一个多维数组的一个区域内数据
nd12=np.arange(25).reshape([5,5])
nd12[1:3,1:3]
#截取一个多维数组中，数值在一个值域之内的数据
nd12[(nd12>3)&(nd12<10)]
#截取多维数组中，指定的行,如读取第2,3行
nd12[[1,2]]  #或nd12[1:3,:]
##截取多维数组中，指定的列,如读取第2,3列
nd12[:,1:3]


# 获取数组中的部分元素除通过指定索引标签外，还可以使用一些函数来实现，如通过random.choice函数可以从指定的样本中进行随机抽取数据。

# In[12]:

import numpy as np
from numpy import random as nr

a=np.arange(1,25,dtype=float)
c1=nr.choice(a,size=(3,4))  #size指定输出数组形状
c2=nr.choice(a,size=(3,4),replace=False)  #replace缺省为True，即可重复抽取。
#下式中参数p指定每个元素对应的抽取概率，缺省为每个元素被抽取的概率相同。
c3=nr.choice(a,size=(3,4),p=a / np.sum(a))
print("随机可重复抽取")
print(c1)
print("随机但不重复抽取")
print(c2)
print("随机但按制度概率抽取")
print(c3)


# 1.3矩阵操作

# In[13]:

import numpy as np

nd14=np.arange(9).reshape([3,3])

#矩阵转置
np.transpose(nd14)

#矩阵乘法运算
a=np.arange(12).reshape([3,4])
b=np.arange(8).reshape([4,2])
a.dot(b)

#求矩阵的迹
a.trace()
#计算矩阵行列式
np.linalg.det(nd14)

#计算逆矩阵
c=np.random.random([3,3])
np.linalg.solve(c,np.eye(3))


# 1.4数据合并与展平

# （1）合并一维数组

# In[14]:

import numpy as np
a=np.array([1,2,3])
b=np.array([4,5,6])
c=np.append(a,b)
print(c)
#或利用concatenate
d=np.concatenate([a,b])
print(d)


# （2）多维数组的合并

# In[15]:

import numpy as np
a=np.arange(4).reshape(2,2)
b=np.arange(4).reshape(2,2)
#按行合并
c=np.append(a,b,axis=0)
print(c)
print("合并后数据维度",c.shape)
#按列合并
d=np.append(a,b,axis=1)
print("按列合并结果:")
print(d)
print("合并后数据维度",d.shape)


# (3)矩阵展平

# In[16]:

import numpy as np
nd15=np.arange(6).reshape(2,-1)
print(nd15)
#按照列优先，展平。
print("按列优先,展平")
print(nd15.ravel('F'))
#按照行优先，展平。
print("按行优先,展平")
print(nd15.ravel())


# 1.5通用函数

# （1）使用math与numpy函数性能比较：

# In[17]:

import time
import math
import numpy as np

x = [i * 0.001 for i in np.arange(1000000)]
start = time.clock()
for i, t in enumerate(x):
    x[i] = math.sin(t)
print ("math.sin:", time.clock() - start )

x = [i * 0.001 for i in np.arange(1000000)]
x = np.array(x)
start = time.clock()
np.sin(x)
print ("numpy.sin:", time.clock() - start )


# （2）使用循环与向量运算比较：

# In[18]:

import time
import numpy as np

x1 = np.random.rand(1000000)
x2 = np.random.rand(1000000)
##使用循环计算向量点积
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n for loop----- Computation time = " + str(1000*(toc - tic)) + "ms")
##使用numpy函数求点积
tic = time.process_time()
dot = 0
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n verctor version---- Computation time = " + str(1000*(toc - tic)) + "ms")


# 1.6 广播机制

# In[19]:

import numpy as np
a=np.arange(10)
b=np.arange(10)
#两个shape相同的数组相加
print(a+b)
#一个数组与标量相加
print(a+3)
#两个向量相乘
print(a*b)

#多维数组之间的运算
c=np.arange(10).reshape([5,2])
d=np.arange(2).reshape([1,2])
#首先将d数组进行复制扩充为[5,2],如何复制请参考图1-2，然后相加。
print(c+d)


# In[ ]:



