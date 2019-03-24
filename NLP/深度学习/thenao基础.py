
# coding: utf-8

# 2.2符号变量

# In[1]:

#导入需要的库或模块
import theano
from theano import tensor as T

#初始化张量
x=T.scalar(name='input',dtype='float32')
w=T.scalar(name='weight',dtype='float32')
b=T.scalar(name='bias',dtype='float32')
z=w*x+b

#编译程序
net_input=theano.function(inputs=[w,x,b],outputs=z)
#执行程序
print('net_input: %2f'% net_input(2.0,3.0,0.5))


# （1）使用内置的变量类型创建

# In[2]:

import theano
from theano import tensor as T

x=T.scalar(name='input',dtype='float32')
data=T.vector(name='data',dtype='float64')


# （2）自定义变量类型

# In[3]:

import theano
from theano import tensor as T

mytype=T.TensorType('float64',broadcastable=(),name=None,sparse_grad=False)


# 
# 图2-1（具体图形在书上）中矩阵与向量相加的具体代码如下
# 

# In[4]:

import theano
import numpy as np
import theano.tensor as T
r = T.row()
r.broadcastable
# (True, False)

mtr = T.matrix()
mtr.broadcastable
# (False, False)

f_row = theano.function([r, mtr], [r + mtr])
R = np.arange(1,3).reshape(1,2)
print(R)
#array([[1, 2]])

M = np.arange(1,7).reshape(3, 2)
print(M)
#array([[1, 2],
#       [3, 4],
#       [5, 6]])

f_row(R, M)


# (3) 将Python类型变量或者Numpy类型变量转化为Theano共享变量

# In[5]:

import theano
import numpy as np
import theano.tensor as T

data=np.array([[1,2],[3,4]])
shared_data=theano.shared(data)
type(shared_data)


# 2.3符号计算图模型

# In[6]:

import theano
import numpy as np
import theano.tensor as T

x = T.dmatrix('x')  
y = T.dmatrix('y')  
z = x + y  


# 2.4函数

# （1）函数定义的格式

# In[7]:

import theano  
x, y =theano.tensor.fscalars('x', 'y')  
z1= x + y  
z2=x*y  
#定义x、y为自变量，z1、z2为函数返回值（因变量）
f =theano.function([x,y],[z1,z2])  

#返回当x=2，y=3的时候，函数f的因变量z1，z2的值
print(f(2,3))


# （2）自动求导

# In[8]:

import theano  
x =theano.tensor.fscalar('x')#定义一个float类型的变量x  
y= 1 / (1 + theano.tensor.exp(-x))#定义变量y  
dx=theano.grad(y,x)#偏导数函数  
f= theano.function([x],dx)#定义函数f，输入为x，输出为s函数的偏导数  
print(f(3))#计算当x=3的时候，函数y的偏导数


# （3）更新共享变量参数

# In[9]:

import theano  
w= theano.shared(1)#定义一个共享变量w，其初始值为1  
x=theano.tensor.iscalar('x')  
f=theano.function([x], w, updates=[[w, w+x]])#定义函数自变量为x，因变量为w，当函数执行完毕后，更新参数w=w+x  
print(f(3))#函数输出为w  
print(w.get_value())#这个时候可以看到w=w+x为4


# In[10]:

import numpy  as np
import theano  
import theano.tensor as T  
rng = np.random  

# 我们为了测试，自己生成10个样本，每个样本是3维的向量，然后用于训练 
N = 10     
feats = 3  
D = (rng.randn(N, feats).astype(np.float32), rng.randint(size=N, low=0, high=2).astype(np.float32))    

# 声明自变量x、以及每个样本对应的标签y(训练标签)  
x = T.matrix("x")  
y = T.vector("y")  

#随机初始化参数w、b=0，为共享变量  
w = theano.shared(rng.randn(feats), name="w")  
b = theano.shared(0., name="b")  

#构造代价函数
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # s激活函数  
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # 交叉商代价函数
cost = xent.mean() + 0.01 * (w ** 2).sum()# 代价函数的平均值+L2正则项以防过拟合，其中权重衰减系数为0.01  
gw, gb = T.grad(cost, [w, b])             #对总代价函数求参数的偏导数  

prediction = p_1 > 0.5                    # 大于0.5预测值为1，否则为0.

train = theano.function(inputs=[x,y],outputs=[prediction, xent],updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))#训练所需函数  
predict = theano.function(inputs=[x], outputs=prediction)#测试阶段函数  

#训练  
training_steps = 1000  
for i in range(training_steps):  
    pred, err = train(D[0], D[1])  
    print (err.mean())#查看代价函数下降变化过程  


# 2.5条件与循环

# （1）条件判断

# In[11]:

from theano import tensor as T  
from theano.ifelse import ifelse  
import theano,time,numpy  

a,b=T.scalars('a','b')  
x,y=T.matrices('x','y')  
z_switch=T.switch(T.lt(a,b),T.mean(x),T.mean(y))#lt:a < b?  
z_lazy=ifelse(T.lt(a,b),T.mean(x),T.mean(y))  

#optimizer:optimizer的类型结构（可以简化计算，增加计算的稳定性）  
#linker:决定使用哪种方式进行编译(C/Python) 
f_switch = theano.function([a, b, x, y], z_switch,mode=theano.Mode(linker='vm'))  
f_lazyifelse = theano.function([a, b, x, y], z_lazy,mode=theano.Mode(linker='vm'))  

val1 = 0.  
val2 = 1.  
big_mat1 = numpy.ones((1000, 100))  
big_mat2 = numpy.ones((1000, 100))  

n_times = 10  

tic = time.clock()  
for i in range(n_times):  
    f_switch(val1, val2, big_mat1, big_mat2)  
print('time spent evaluating both values %f sec' % (time.clock() - tic))  

tic = time.clock()  
for i in range(n_times):  
    f_lazyifelse(val1, val2, big_mat1, big_mat2)  
print('time spent evaluating one value %f sec' % (time.clock() - tic))


# （2）循环语句

# In[12]:

import theano
import theano.tensor as T
import numpy as np

# 定义单步的函数,实现a*x^n
# 输入参数的顺序要与下面scan的输入参数对应
def one_step(coef, power, x):
    return coef * x ** power

coefs = T.ivector()  # 每步变化的值,系数组成的向量
powers = T.ivector() # 每步变化的值,指数组成的向量
x = T.iscalar()      # 每步不变的值,自变量

# seq,out_info,non_seq与one_step函数的参数顺序一一对应
# 返回的result是每一项的符号表达式组成的list
result, updates = theano.scan(fn = one_step,
                       sequences = [coefs, powers],
                       outputs_info = None,
                       non_sequences = x)

# 每一项的值与输入的函数关系
f_poly = theano.function([x, coefs, powers], result, allow_input_downcast=True)

coef_val = np.array([2,3,4,6,5])
power_val = np.array([0,1,2,3,4])
x_val = 10

print("多项式各项的值: ",f_poly(x_val, coef_val, power_val))
#scan返回的result是每一项的值，并没有求和，如果我们只想要多项式的值，可以把f_poly写成这样：
# 多项式每一项的和与输入的函数关系
f_poly = theano.function([x, coefs, powers], result.sum(), allow_input_downcast=True)

print("多项式和的值：",f_poly(x_val, coef_val, power_val))


# 2.6共享变量

# In[13]:

import theano
import theano.tensor as T
from theano import shared
import numpy as np

#定义一个共享变量，并初始化为0
state = shared(0)
inc = T.iscalar('inc')
accumulator = theano.function([inc], state, updates=[(state, state+inc)])
# 打印state的初始值
print(state.get_value())
accumulator(1) # 进行一次函数调用
# 函数返回后，state的值发生了变化
print(state.get_value()) 


# In[ ]:



