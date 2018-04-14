import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from ImproveDeepNN.OptimizationMethods.opt_utils import *
from ImproveDeepNN.OptimizationMethods.testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Arguments:
    :param parameters: python dictionary containing parameters to be updated
    :param grads: python dictionary containing gradients to update each parameters
    :param learning_rate: the learning rate, scalar

    :return:
    parameters: python dictionary containing updated parameters
    """

    L = len(parameters) // 2

    for i in range(1, L + 1):
        parameters["W" + str(i)] = parameters["W" + str(i)] - learning_rate * grads["dW" + str(i)]
        parameters["b" + str(i)] = parameters["b" + str(i)] - learning_rate * grads["db" + str(i)]

    return parameters

'''
parameters, grads, learning_rate = update_parameters_with_gd_test_case()

parameters = update_parameters_with_gd(parameters, grads, learning_rate)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

def gradient_descent(data_input, labels, layers_dims, num_iterations):
    X = data_input
    Y = labels
    parameters = initialize_parameters(layers_dims)
    for i in range(num_iterations):
        a, cache = forward_propagation(X, parameters)
        cost = compute_cost(a, Y)
        grads = backward_propagation(X, Y, cache)
        parameters = update_parameters_with_gd(parameters, grads, learning_rate=0.01)
        if i % 100 == 0:
            print(cost)
    return parameters

def stochastic_gradient_descent(data_input, labels, layers_dims, num_iterations):
    X = data_input
    Y = labels
    parameters = initialize_parameters(layers_dims)
    m_size = X.shape[1]

    for i in range(num_iterations):
        for j in range(m_size):
            x = X[:, j].reshape(12288, 1)
            y = Y[:, j].reshape(1, 1)
            a, cache = forward_propagation(x, parameters)
            cost = compute_cost(a, y.reshape(1, 1))
            grads = backward_propagation(x, y, cache)
            parameters = update_parameters_with_gd(parameters, grads, learning_rate=0.01)
        if i % 100 == 0:
            print(cost)
    return parameters

'''
Note also that implementing SGD requires 3 for-loops in total: 
1. Over the number of iterations 
2. Over the m
 training examples 
3. Over the layers (to update all parameters, from (W[1],b[1])
 to (W[L],b[L])
)
'''


X = np.random.randn(12288, 148)
Y = np.random.randn(1, 148) < 0.5
parameters1 = gradient_descent(X, Y, [12288, 20, 7, 1], 100)
parameters2 = stochastic_gradient_descent(X, Y, [12288, 20, 7, 1], 100)

predict(X, Y, parameters1)
predict(X, Y, parameters2)


'''
What you should remember: 
- The difference between gradient descent, mini-batch gradient descent and stochastic gradient descent is the number of examples you use to perform one update step. 
- You have to tune a learning rate hyperparameter α
. 
- With a well-turned mini-batch size, usually it outperforms either gradient descent or stochastic gradient descent (particularly when the training set is large).
'''

