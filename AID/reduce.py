import numpy as np
import functools as ft


def f1(x, y):
    return x + y


def f2(x, y):
    return x * y


a, b = 1, 2
c = f1(a, b)
print(c)

d = [1, 2, 3, 4, 5, 6]
e = 0
for x in d:
    e += x
print(e)

x = 0
for i in range(-1, -len(d) - 1, -1):
    y = d[i]
    x = f1(x, y)
print(x)

#f = ft.reduce(f1, d)
f = ft.reduce(lambda x, y: x + y, d)
print(f)
#
# 5,6->f1->11,4->f1->15,3->f1->18,2->f1->20,1->f1->21
#
#g = ft.reduce(f2, range(1, 6))
g = ft.reduce(lambda x, y: x * y, range(1, 6))
print(g)

X = np.array([1, 2, 3, 4, 5])
Y = np.array([10, 20, 30, 40, 50])
Z = np.frompyfunc(f1, 2, 1)(X, Y)
print(Z)
