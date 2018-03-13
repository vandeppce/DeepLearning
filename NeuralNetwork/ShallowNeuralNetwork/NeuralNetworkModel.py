import numpy as np
import matplotlib.pylab as plt
from NeuralNetwork.testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from NeuralNetwork.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

# 1 defining the neural network structure
def layer_sizes(X, Y):
    """
    Arguments:
    :param X: input dataset of shape (input size, number of examples)
    :param Y: labels of shape (output size, number of examples)

    :return:
    n_x: the size of the input layer
    n_h: the size of hidden layer
    n_y: the size of the output layer
    """

    n_x = X.shape[0]
    n_h = 4        #隐含层含有4个节点
    n_y = Y.shape[0]  #输出层根据类别数确定节点个数

    return n_x, n_h, n_y

X_assess, Y_assess = layer_sizes_test_case()
n_x, n_h, n_y = layer_sizes(X_assess, Y_assess)

print("The number of examples is: " + str(X_assess.shape[1]))
print("The size of the input layer is: " + str(n_x))
print("The size of the hidden layer is: " + str(n_h))
print("The size of the output layer is: " + str(n_y))

# 2 initialize the model's parameters
def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    :param n_x: size of the input layer
    :param n_h: size of the hidden layer
    :param n_y: size of the output layer

    :return:
    params: python dictionary containing parameters
        W1: weight matrix of shape (n_h, n_x)
        b1: bias vector of shape (n_h, 1)
        W2: weight matrix of shape (n_y, n_h)
        b2: bias vector of shape (n_y, 1)
    """

    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x)   #randn() 标准正态分布     rand() [0,1]之间
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros(shape=(n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2
              }

    return params

n_x, n_h, n_y = initialize_parameters_test_case()
params = initialize_parameters(n_x, n_h, n_y)

print("W1 = " + str(params['W1']))
print("b1 = " + str(params['b1']))
print("W2 = " + str(params['W2']))
print("b2 = " + str(params['b2']))

# 3 the loop
def forward_propagation(X, parameters):
    """
    Arguments:
    :param X: input data of size (n_x, m)
    :param parameters: python dictionary containing parameters (output of initialization function)

    :return:
    A2: the sigmoid output of the second activation
    cache: a dictionary containing "Z1", "A1", "Z2" and "A2"

    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(Z1.shape == (n_h, X.shape[1]))
    assert(A1.shape == (n_h, X.shape[1]))

    assert(Z2.shape == (n_y, X.shape[1]))
    assert(A2.shape == (n_y, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2
             }

    return A2, cache

X_assess1, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess1, parameters)

print(np.mean(cache['Z1']), np.mean(cache['A1']), np.mean(cache['Z2']), np.mean(cache['A2']))