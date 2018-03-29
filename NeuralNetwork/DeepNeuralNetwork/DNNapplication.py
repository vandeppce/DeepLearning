import time
import numpy as np
import h5py
import matplotlib.pylab as plt
import scipy
from PIL import Image
from scipy import ndimage
#from NeuralNetwork.DeepNeuralNetwork.dnn_app_utils_v2 import *

from NeuralNetwork.DeepNeuralNetwork.DNNmodel import *
from NeuralNetwork.DeepNeuralNetwork.dnn_app_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward, load_data, predict, print_mislabeled_images

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
# print(train_x_orig.shape) #(209, 64, 64, 3)
# view the dataset
'''
index = 10
plt.imshow(train_x_orig[index])
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
'''

m_train = train_x_orig.shape[0]         # the number of examples
num_px = train_x_orig.shape[1]          # the number of px
m_test = test_x_orig.shape[0]           # the number of test examples

'''
print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))
'''

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

''' wrong reshape
train_x_flatten1 = train_x_orig.reshape(num_px * num_px * 3, m_train)
test_x_flatten1 = test_x_orig.reshape(num_px * num_px * 3, m_test)
'''
# right reshape

train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

'''
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
'''

# 1 architecture of model
'''
As usual you will follow the Deep Learning methodology to build the model: 
1. Initialize parameters / Define hyperparameters 
2. Loop for num_iterations: 
a. Forward propagation 
b. Compute cost function 
c. Backward propagation 
d. Update parameters (using parameters, and grads from backprop) 
4. Use trained parameters to predict labels
'''

# 1.1 2-layer neural network

def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    """
    Arguments:
    :param X: input data, shape (n_x, number of examples)
    :param Y: true "label" vector, shape (1, number of examples)
    :param layers_dims: dimensions of the layers (n_x, n_h, n_y)
    :param learning_rate: learning rate of the gardient descent update rule
    :param num_iterations: number of iterations of the optimization loop
    :param print_cost: if set to True, this will print the cost every 100 iterations

    :return:
    parameters: a dictionary containing W1, W2, b1, b2
    """

    np.random.seed(1)
    costs = []
    grads = {}

    # 1 initialize
    (n_x, n_h, n_y) = layers_dims
    parameters = initialize_parameters(n_x, n_h, n_y)

    # 2 loop
    for i in range(num_iterations):
        caches = []
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # a forward propagation
        A1, cache = linear_activation_forward(X, W1, b1, activation="relu")
        caches.append(cache)

        A2, cache = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        caches.append(cache)

        # b cost function
        cost = compute_cost(A2, Y)
        costs.append(cost)

        # c backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))

        dA1, dW2, db2 = linear_activation_backward(dA2, caches[1], activation="sigmoid")
        dX, dW1, db1 = linear_activation_backward(dA1, caches[0], activation="relu")

        grads["dW1"] = dW1
        grads["db1"] = db1
        grads["dW2"] = dW2
        grads["db2"] = db2

        # d update
        parameters = update_parameters(parameters, grads, learning_rate)

        # e print
        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("cost after %i iterations is: %f" % (i, cost))


    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

'''
# train the parameters
n_x = 12288
n_h = 7
n_y = 1
layers_dims = (n_x, n_h, n_y)
parameters = two_layer_model(train_x, train_y, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
# predict
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)
'''

# 1.2 L-layer neural network

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost = False):
    """
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []
    # 1 initialize
    parameters = initialize_parameters_deep(layers_dims)
    L = len(layers_dims) - 1
    AL = "A" + np.str(L)
    # 2 loop
    for i in range(num_iterations):
        # a forward propagation
        AL, caches = L_model_forward(X, parameters)

        # b cost
        cost = compute_cost(AL, Y)

        # c backward propagation
        grads = L_model_backward(AL, Y, caches)

        # d update
        parameters = update_parameters(parameters, grads, learning_rate)

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("cost after %i iterations is: %f" % (i, cost))
    '''
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    '''
    return parameters


# train
layers_dims = [12288, 20, 7, 5, 1]
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)
# predict
predictions_train = predict(train_x, train_y, parameters)
predictions_test = predict(test_x, test_y, parameters)
'''

# 2 results analysis

# print_mislabeled_images(classes, test_x, test_y, predictions_test)

# 3 test on own image

'''
my_image = "my_image2.jpg" # change this to the name of your image file
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))   #重构
my_predicted_image = predict(my_image, my_label_y, parameters)

plt.imshow(image)
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")