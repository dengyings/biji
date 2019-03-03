import functools as ft


def f1(x):
    return x + 3


def f2(x):
    return x * 6


def f3(x):
    return x - 9


a = 1
print((a + 3) * 6 - 9)
print(f3(f2(f1(a))))
print(ft.reduce(lambda fa, fb: lambda x: fa(fb(x)), [f3, f2, f1])(a))
#
# f2,f1->l1->f2(f1(a)),f3->l1->f3(f2(f1(a)))
#
