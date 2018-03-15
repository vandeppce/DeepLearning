import numpy as np
import h5py
import matplotlib.pylab as plt
from NeuralNetwork.DeepNeuralNetwork.testCases_v2 import *
from NeuralNetwork.DeepNeuralNetwork.dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# 1.1 initialization for 2-layer neural network
# the model's structure is: linear -> relu -> linear -> sigmoid

def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    :param n_x: size of the input layer
    :param n_h: size of the hidden layer
    :param n_y: size of the output layer

    :return:
    parameters: python dictionary containing parameters:
        W1: weight matrix of shape (n_h, n_x)
        b1: bias vector of shape (n_h, 1)
        W2: weight matrix of shape (n_y, n_h)
        b2: bias vector of shape (n_y, 1)
    """

    np.random.seed(1)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    assert(W1.shape == (n_h, n_x))
    assert(b1.shape == (n_h, 1))
    assert(W2.shape == (n_y, n_h))
    assert(b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2
                  }

    return parameters

'''
parameters = initialize_parameters(3, 2, 1)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

# 1.2 initialize L-layer neural network
# structure: [linear -> relu * (L - 1) -> [linear -> sigmoid] * 1
'''
if L == 1:
    parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
    parameters["b" + str(L)] = np.zeros(shape=(layer_dims[1], 1))
'''

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    :param layer_dims: python array (list) containing the dimension of layer in network

    :return:
    parameter: python dictionary containing parameters "W1", "b1", ... , "WL", "bL"
        Wl: weight matrix of shape (layer_dims[l], layer_dims[l - 1]
        bl: bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    L = len(layer_dims) # the depth of the network
    parameter = {}

    for i in range(1, L):
        parameter["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * 0.01
        parameter["b" + str(i)] = np.zeros(shape=(layer_dims[i], 1))

        assert(parameter["W" + str(i)].shape == (layer_dims[i], layer_dims[i - 1]))
        assert(parameter["b" + str(i)].shape == (layer_dims[i], 1))

    return parameter

'''
parameters = initialize_parameters_deep([5, 4, 3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

# 2 forward propagation module
# 2.1 linear forward

def linear_forward(A, W, b):
    """
    Arguments:
    :param A: activations from previous (or input data): (size of previous layer, number of examples)
    :param W: weight matrix: numpy array of shape (size of current layer, size of previous layer)
    :param b: bias vector, numpy array of shape (size of current layer, 1)

    :return:
    Z: the input of the activation function, also called pre-activation parameter
    cache: a python dictionary containing "A", "b" and "W"; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b
    assert(Z.shape == (W.shape[0], A.shape[1]))

    cache = {"A": A,
             "W": W,
             "b": b
             }

    return Z, cache

'''
A, W, b = linear_forward_test_case()

Z, linear_cache = linear_forward(A, W, b)
print("Z = " + str(Z))
'''

# 2.2 linear-activation forward

def linear_activation_forward(A_prev, W, b, activation):
    """
    Arguments:
    :param A_prev: activations from previous layer (or input data): (size of previous layer, number of examples)
    :param W: weight matrix: numpy array of shape (size of current layer, size of previous layer)
    :param b: bias vector, numpy array of shape (size of current layer, 1)
    :param activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    :return:
    A: the output of the activation function, also called the post-activation parameter
    cache: a python dictionary containing "linear_cache" and "activation_cache"; stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    assert(A.shape == Z.shape)
    cache = {"linear_cache": linear_cache,
             "activation_cache": activation_cache}

    return A, cache

'''
A_prev, W, b = linear_activation_forward_test_case()

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "sigmoid")
print("With sigmoid: A = " + str(A))

A, linear_activation_cache = linear_activation_forward(A_prev, W, b, activation = "relu")
print("With ReLU: A = " + str(A))
'''

# 2.3 L-layer model

def L_model_forward(X, parameters):
    """
    Arguments:
    :param X: data, numpy array of shape (input size, number of examples)
    :param parameters: output of initialize_parameters_deep()

    :return:
    AL: last post-activation value
    caches: list of caches containing:
        every cache of linear_relu_forward() (there are L - 1 of them, indexed from 0 to L - 2)
        the cache of linear_sigmoid_forward(0 (there is one, indexed L - 1)
    """
    caches = []
    A = X
    L = len(parameters) // 2
    for i in range(1, L):
        A_prev = A
        Wi = parameters["W" + str(i)]
        bi = parameters["b" + str(i)]
        A, cache = linear_activation_forward(A_prev, Wi, bi, "relu")

        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters["W" + str(L)], parameters["b" + str(L)], activation="sigmoid")
    caches.append(cache)

    assert(AL.shape == (1, X.shape[1]))

    return AL, caches

'''
X, parameters = L_model_forward_test_case()
AL, caches = L_model_forward(X, parameters)
print("AL = " + str(AL))
print("Length of caches list = " + str(len(caches)))
'''

# 3 cost function

def compute_cost(AL, Y):
    """
    Arguments:
    :param AL: probability vector corresponding to label predictions, shape (1, m)
    :param Y: true "label" vector, shape (1, m)

    :return:
    cost: cross-entropy cost
    """

    m_size = Y.shape[1]

    cost = -1.0 / m_size * np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL)))   # *对数组执行元素乘，对矩阵执行矩阵乘, multiply对数组和矩阵均执行元素乘

    cost = np.squeeze(cost)
    assert (isinstance(cost, float))

    return cost

'''
Y, AL = compute_cost_test_case()

print("cost = " + str(compute_cost(AL, Y)))
'''

# 4 backward propagation module
# 4.1 linear backward

def linear_backward(dZ, cache):
    """
    Arguments:
    :param dZ: gradient of the cost with respect to the linear output (of current layer l)
    :param cache: dictionary of value values (A_prev, W, b) coming from the forward propagation in the current layer

    :return:
    dA_prev: gradient of the cost with respect to the activation (of the previous layer l - 1, same shape as A_prev)
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    A_prev = cache["A"]
    W = cache["W"]
    b = cache["b"]

    m_size = dZ.shape[1]

    dW = 1.0 / m_size * np.dot(dZ, A_prev.T)
    db = 1.0 / m_size * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

'''
dZ, linear_cache = linear_backward_test_case()

dA_prev, dW, db = linear_backward(dZ, linear_cache)
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
'''

# 4.2 linear-activation backward

def linear_activation_backward(dA, cache, activation):
    """
    Arguments:
    :param dA: post-activation gradient for current layer l
    :param cache: dictionary of values (linear_cache, activation_cache) we store for computing
    :param activation: the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    :return:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache = cache["linear_cache"]
    activation_cache = cache["activation_cache"]

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db

'''
AL, linear_activation_cache = linear_activation_backward_test_case()

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "sigmoid")
print ("sigmoid:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db) + "\n")

dA_prev, dW, db = linear_activation_backward(AL, linear_activation_cache, activation = "relu")
print ("relu:")
print ("dA_prev = "+ str(dA_prev))
print ("dW = " + str(dW))
print ("db = " + str(db))
'''

# 4.3 L-Model backward

def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                    every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                    the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
                 grads["dA" + str(l)] = ...
                 grads["dW" + str(l)] = ...
                 grads["db" + str(l)] = ...
    """

    grads = {}
    L = len(caches)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    linear_cache = caches[L - 1]["linear_cache"]
    activation_cache = caches[L - 1]["activation_cache"]
    dA_prev, dWL, dbL = linear_activation_backward(dAL, caches[L - 1], activation="sigmoid")
    grads["dA" + str(L)] = dAL
    grads["dW" + str(L)] = dWL
    grads["db" + str(L)] = dbL

    for i in range(1, L):
        dA = dA_prev
        dA_prev, dW, db = linear_activation_backward(dA, caches[L - 1 - i], activation="relu")
        grads["dA" + str(L - i)] = dA
        grads["dW" + str(L - i)] = dW
        grads["db" + str(L - i)] = db

    return grads

'''
AL, Y_assess, caches = L_model_backward_test_case()
grads = L_model_backward(AL, Y_assess, caches)
print(grads)
'''

# 4.4 update parameters

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]

    return parameters

'''
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads, 0.1)

print ("W1 = "+ str(parameters["W1"]))
print ("b1 = "+ str(parameters["b1"]))
print ("W2 = "+ str(parameters["W2"]))
print ("b2 = "+ str(parameters["b2"]))
'''

