import numpy as np
import matplotlib.pylab as plt
import sklearn
import sklearn.datasets
from ImproveDeepNN.PracticalAspects.init_utils import sigmoid, relu, compute_loss, forward_propagation, backward_propagation
from ImproveDeepNN.PracticalAspects.init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_dataset()

def initialize_parameters_zeros(layers_dims):
    """
    What you should remember:
        - The weights W[l] should be initialized randomly to break symmetry.
        - It is however okay to initialize the biases b[l] to zeros. Symmetry is still broken so long as W[l]is initialized randomly.

    Arguments:
    layers_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    parameters = {}
    L = len(layers_dims)

    for i in range(1, L):
        parameters['W' + str(i)] = np.zeros(shape=(layers_dims[i], layers_dims[i - 1]))
        parameters['b' + str(i)] = np.zeros(shape=(layers_dims[i], 1))

    return parameters

def initialize_parameters_random(layers_dims):
    """
    In summary:
        - Initializing weights to very large random values does not work well.
        - Hopefully intializing with small random values does better. The important question is: how small should be these random values be?
    Arguments:
    layers_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                        ...
                        WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                        bL -- bias vector of shape (layers_dims[L], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for i in range(1, L):
        parameters["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * 10
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))

    return parameters

def initialize_parameters_he(layers_dims):
    """
    Arguments:
    layers_dims -- python array (list) containing the size of each layer.

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    W1 -- weight matrix of shape (layers_dims[1], layers_dims[0])
                    b1 -- bias vector of shape (layers_dims[1], 1)
                    ...
                    WL -- weight matrix of shape (layers_dims[L], layers_dims[L-1])
                    bL -- bias vector of shape (layers_dims[L], 1)
    """
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)

    for i in range(1, L):
        parameters["W" + str(i)] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * np.sqrt(2.0 / layers_dims[i - 1])
        parameters["b" + str(i)] = np.zeros((layers_dims[i], 1))

    return parameters

'''
parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

def model(X, Y, learning_rate = 0.01, num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")

    Returns:
    parameters -- parameters learnt by the model
    """

    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0], 10, 5, 1]
    parameters = {}

    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dims)

    for i in range(num_iterations):
        # forward_propagation
        y_hat, cache = forward_propagation(X, parameters)

        # compute loss
        cost = compute_loss(y_hat, Y)

        # back_propagation
        grads = backward_propagation(X, Y, cache)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 1000 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration {}: {}".format(i, cost))

    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


parameters = model(train_X, train_Y, initialization="he")
print("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

print ("predictions_train = " + str(predictions_train))
print ("predictions_test = " + str(predictions_test))

plt.title("Model with he initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

'''
What you should remember from this notebook: 
- Different initializations lead to different results 
- Random initialization is used to break symmetry and make sure different hidden units can learn different things 
- Don’t intialize to values that are too large 
- He initialization works well for networks with ReLU activations.
'''
