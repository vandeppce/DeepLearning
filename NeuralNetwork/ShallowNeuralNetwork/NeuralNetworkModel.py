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

def compute_cost(A2, Y, parameters):
    """
    Arguments:
    :param A2: The sigmoid output of the second activation, of shape(1, m)
    :param Y: "true" labels vector of shape (1, m)
    :param parameters: python dictionary containing parameters W1, b1, W2, b2

    :return:
    cost: cross-entropy cost
    """

    m = A2.shape[1]
    logprobs = np.multiply(Y, np.log(A2))
    cost = -1.0 / m * np.sum(logprobs)

    cost = np.squeeze(cost)
    assert(isinstance(cost, float))
    # print(cost.dtype)   #float64

    return cost

A2_1, Y_assess_1, parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2_1, Y_assess_1, parameters)))

def backword_propagation(parameters, cache, X, Y):
    """
    Arguments:
    :param parameters: python dictionary containing parameters
    :param cache: a dictionary containing "Z1", "A1", "Z2" and "A2"
    :param X: input data of shape (2, m)
    :param Y: "true" labels vector of shape (1, m)

    :return:
    grads: python dictionary containing gradients
    """

    m = X.shape[1]

    W1 = parameters['W1']
    W2 = parameters['W2']

    A1 = cache["A1"]
    A2 = cache["A2"]

    Z1 = cache["Z1"]
    Z2 = cache["Z2"]

    dZ2 = A2 - Y
    dW2 = 1.0 / m * np.dot(dZ2, A1.T)
    db2 = 1.0 / m * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1.0 / m * np.dot(dZ1, X.T)
    db1 = 1.0 / m * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2
             }
    return grads

parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backword_propagation(parameters, cache, X_assess, Y_assess)

print("dW1 = " + str(grads['dW1']))
print("db1 = " + str(grads['db1']))
print("dW2 = " + str(grads['dW2']))
print("db2 = " + str(grads['db2']))

def update_parameters(parameters, grads, learning_rate = 1.2):
    """
    Arguments:
    :param parameters: python dictionary containing parameters
    :param grads: python dictionary containing gradients
    :param learning_rate:

    :return:
    parameters: python dictionary containing updated parameters
    """

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1  = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
                  }

    return parameters

parameters, grads = update_parameters_test_case()
params = update_parameters(parameters, grads)

print("W1 = " + str(params["W1"]))
print("b1 = " + str(params["b1"]))
print("W2 = " + str(params["W2"]))
print("b2 = " + str(params["b2"]))


