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

m_train = train_x_orig.shape[0]         # the number of examples
num_px = train_x_orig.shape[1]          # the number of px
m_test = test_x_orig.shape[0]           # the number of test examples

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
test_x = test_x_flatten / 255

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

layers_dims = [12288, 20, 7, 5, 1]
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True)

def cat_generization(parameter, iteration_num = 20000, target = 1, learning_rate = 500):
    #cat = np.zeros(shape=(12288, 1)) * 255
    cat = np.random.randint(low=0, high=255, size=(12288, 1))
    for i in range(iteration_num):
        cat = np.clip(cat, 0.0, 255.0)
        AL, caches = L_model_forward(cat / 255.0, parameter)
        cost = -1.0 / 1 * np.sum(np.multiply(target, np.log(AL)) + np.multiply(1 - target, np.log(1 - AL)))
        grads = L_model_backward(AL, target, caches)
        cat_grads = grads["dA_prev"]
        cat -= learning_rate * cat_grads
        if i % 1000 == 0:
            print(cost)
    return cat


cat = cat_generization(parameters) / 255.0
cat = np.clip(cat, 0.0, 1.0)
prob, caches = L_model_forward(cat, parameters)
my_predicted_image = predict(cat, 1, parameters)
ad_image = cat.reshape(64, 64, 3)
print(prob)
#plt.imshow(image)
plt.imshow(ad_image)

plt.show()
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
