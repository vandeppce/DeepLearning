# 1.1 exp function
# use math.exp()
import math

def sigmoid_f(x):
    result = 1.0 / (1.0 + 1.0 / math.exp(x))
    return result

print(sigmoid_f(3))

# TypeError
'''
x = [1, 2, 3]
sigmiod_f(x)
'''

#use np.exp()
import numpy as np

def sigmoid_f1(x):
    result = 1.0 / (1.0 + 1.0 / np.exp(x))
    return result

x1 = np.array([1, 2, 3])
print(sigmoid_f1(x1))

# 1.2 sigmoid gradient

def sigmoid_grad(x):
    s = 1.0 / (1.0 + 1.0 / np.exp(x))
    result = s * (1 - s)
    return result

x2 = np.array([1, 2, 3])
print(sigmoid_grad(x2))

# 1.3 reshape arrays

def image2vector(image):
    result = image.reshape(image.shape[0] * image.shape[1] * image.shape[2], 1)
    return result

image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

print(image2vector(image))

# 1.4 normalization rows

def normalizationRaws(x):
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / x_norm
    return x

x3 = np.array(
    [
        [0, 3, 4],
        [1, 6, 4]
    ]
)
print(normalizationRaws(x3))
print(normalizationRaws(x3).shape)

# 1.5 broadcast and softmax function

def softmax_f(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)
    result = x_exp / x_sum
    return result

x4 = np.array(
    [[9, 2, 5, 0, 0],
     [7, 5, 0, 0, 0]]
)
print(softmax_f(x4))