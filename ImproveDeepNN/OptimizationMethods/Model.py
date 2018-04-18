import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from ImproveDeepNN.OptimizationMethods.opt_utils import *
from ImproveDeepNN.OptimizationMethods.testCases import *

def initialize_velocity(parameters):
    """
    Initializes the velocity as a python dictionary with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl

    Returns:
    v -- python dictionary containing the current velocity.
                    v['dW' + str(l)] = velocity of dWl
                    v['db' + str(l)] = velocity of dbl
    """
    L = len(parameters) // 2
    v = {}

    for i in range(L):
        v["dW" + str(i + 1)] = np.zeros(parameters["W" + str(i + 1)].shape)
        v["db" + str(i + 1)] = np.zeros(parameters["b" + str(i + 1)].shape)

    return v

def initialize_adam(parameters) :
    """
    Initializes v and s as two python dictionaries with:
                - keys: "dW1", "db1", ..., "dWL", "dbL"
                - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.

    Arguments:
    parameters -- python dictionary containing your parameters.
                    parameters["W" + str(l)] = Wl
                    parameters["b" + str(l)] = bl

    Returns:
    v -- python dictionary that will contain the exponentially weighted average of the gradient.
                    v["dW" + str(l)] = ...
                    v["db" + str(l)] = ...
    s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                    s["dW" + str(l)] = ...
                    s["db" + str(l)] = ...

    """
    L = len(parameters) // 2
    v = {}
    s = {}

    for i in range(L):
        v["dW" + str(i + 1)] = np.zeros(parameters["W" + str(i + 1)].shape)
        v["db" + str(i + 1)] = np.zeros(parameters["b" + str(i + 1)].shape)
        s["dW" + str(i + 1)] = np.zeros(parameters["W" + str(i + 1)].shape)
        s["db" + str(i + 1)] = np.zeros(parameters["b" + str(i + 1)].shape)

    return v, s

def random_mini_batches(X, Y, mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    mini_batch_size -- size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    np.random.seed(seed)
    m = X.shape[1]

    permutation = np.random.permutation(m)

    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    num_complete_minibatches = m // 64
    mini_batches = []

    for i in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, i * mini_batch_size : (i + 1) * mini_batch_size]
        mini_batch_Y = shuffled_Y[:, i * mini_batch_size : (i + 1) * mini_batch_size]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

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

    for i in range(L):
        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * grads["dW" + str(i + 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * grads["db" + str(i + 1)]

    return parameters

def update_parameters_with_momentum(parameters, grads, v, beta, learning_rate):
    """
    Update parameters using Momentum

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- python dictionary containing the current velocity:
                    v['dW' + str(l)] = ...
                    v['db' + str(l)] = ...
    beta -- the momentum hyperparameter, scalar
    learning_rate -- the learning rate, scalar

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- python dictionary containing your updated velocities
    """

    L = len(parameters) // 2

    for i in range(L):
        v["dW" + str(i + 1)] = beta * v["dW" + str(i + 1)] + (1 - beta) * grads["dW" + str(i + 1)]
        v["db" + str(i + 1)] = beta * v["db" + str(i + 1)] + (1 - beta) * grads["db" + str(i + 1)]

        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * v["dW" + str(i + 1)]
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * v["db" + str(i + 1)]

    return parameters, v

def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam

    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates
    beta2 -- Exponential decay hyperparameter for the second moment estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """

    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for i in range(L):

        v["dW" + str(i + 1)] = beta1 * v["dW" + str(i + 1)] + (1 - beta1) * grads["dW" + str(i + 1)]
        v["db" + str(i + 1)] = beta1 * v["db" + str(i + 1)] + (1 - beta1) * grads["db" + str(i + 1)]

        v_corrected["dW" + str(i + 1)] = v["dW" + str(i + 1)] / (1 - beta1 ** t)
        v_corrected["db" + str(i + 1)] = v["db" + str(i + 1)] / (1 - beta1 ** t)

        s["dW" + str(i + 1)] = beta2 * s["dW" + str(i + 1)] + (1 - beta2) * (grads["dW" + str(i + 1)] ** 2)
        s["db" + str(i + 1)] = beta2 * s["db" + str(i + 1)] + (1 - beta2) * (grads["db" + str(i + 1)] ** 2)

        s_corrected["dW" + str(i + 1)] = s["dW" + str(i + 1)] / (1 - beta2 ** t)
        s_corrected["db" + str(i + 1)] = s["db" + str(i + 1)] / (1 - beta2 ** t)

        parameters["W" + str(i + 1)] = parameters["W" + str(i + 1)] - learning_rate * v_corrected["dW" + str(i + 1)] / (np.sqrt(s_corrected["dW" + str(i + 1)]) + epsilon)
        parameters["b" + str(i + 1)] = parameters["b" + str(i + 1)] - learning_rate * v_corrected["db" + str(i + 1)] / (np.sqrt(s_corrected["db" + str(i + 1)]) + epsilon)

    return parameters, v, s

def model(X, Y, layers_dims, optimizer, learning_rate = 0.0007, mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8, num_epochs = 10000, print_cost = True):
    """
    3-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters
    """
    seed = 10
    t = 0
    costs = []
    parameters = initialize_parameters(layers_dims)

    if optimizer == "gd":
        pass
    if optimizer == "momentum":
        v = initialize_velocity(parameters)
    if optimizer == "adam":
        v, s = initialize_adam(parameters)

    for i in range(num_epochs):
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)

        for batch in minibatches:
            (minibatch_X, minibatch_Y) = batch

            a3, cache = forward_propagation(minibatch_X, parameters)
            cost = compute_cost(a3, minibatch_Y)

            grads = backward_propagation(minibatch_X, minibatch_Y, cache)

            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta1, learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)

        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)


    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters

train_X, train_Y = load_dataset()


# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]

#mini-batch gradient descent
#parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

#mini-batch gradient descent with momentum
#parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

#mini-batch gradient descent with adam mode
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")
# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

"""
Momentum usually helps, but given the small learning rate and the simplistic dataset, its impact is almost negligeable. 
Also, the huge oscillations you see in the cost come from the fact that some minibatches are more difficult than others for the optimization algorithm.

Adam on the other hand, clearly outperforms mini-batch gradient descent and Momentum. 
If you run the model for more epochs on this simple dataset, all three methods will lead to very good results. 
However, you’ve seen that Adam converges a lot faster.

Some advantages of Adam include: 
- Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum) 
- Usually works well even with little tuning of hyperparameters (except α)
"""
