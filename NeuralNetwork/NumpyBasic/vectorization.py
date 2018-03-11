import time
import numpy as np

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

tic = time.process_time()
dot = np.dot(x1, x2)         #矩阵乘
toc = time.process_time()
print("dot = %d, computation time = %s" % (dot, 1000 * (toc - tic)))

tic = time.process_time()
outer = np.outer(x1, x2)     #分别乘
toc = time.process_time()
print("outer = %s, computation time = %s" % (outer, 1000 * (toc - tic)))

tic = time.process_time()
mul = np.multiply(x1, x2)    #点乘
toc = time.process_time()
print("multiply = %s, computation time = %s" % (mul, 1000 * (toc - tic)))

W = np.random.rand(3,len(x1))
tic = time.process_time()
dot1 = np.dot(W, x1)
toc = time.process_time()
print("dot1 = %s, computation time = %s" % (dot1, 1000 * (toc - tic)))




