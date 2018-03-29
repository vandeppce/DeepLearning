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
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 1500, print_cost = True)

def noise_generization(origin_image, parameter, iteration_num = 10000, noise_limit = 3, target = 0, learning_rate = 0.05):
    noise = np.random.randn(origin_image.shape[0], origin_image.shape[1])     #随机初始化噪声
    for i in range(iteration_num):
        noise_image = origin_image + noise          #噪声图像
        noise_image = np.clip(noise_image, 0.0, 255.0)
        AL, caches = L_model_forward(noise_image / 255.0, parameter)       #噪声图像前向传播
        #AL = np.clip(AL, 0.1, 0.7)
        cost = -1.0 / 1 * np.sum(np.multiply(target, np.log(AL)) + np.multiply(1 - target, np.log(1 - AL)))
        grads = L_model_backward(AL, target, caches)
        noise_grads = grads["dA_prev"]               #计算损失函数相对噪声对梯度，这里由于噪声和原图直接相加，因此噪声的梯度就是输入A0的梯度
        noise -= learning_rate * noise_grads         #更新噪声
        noise = np.clip(noise, -noise_limit, noise_limit)     #压缩到限制的范围内
        if i % 1000 == 0:
            print(cost)
    return noise


my_image = "my_image2.jpg"
my_label_y = [1] # the true class of your image (1 -> cat, 0 -> non-cat)

fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(num_px,num_px)).reshape((num_px*num_px*3,1))   #重构
noise = noise_generization(my_image, parameters)
for i in range(my_image.shape[0]):
    my_image[i][0] = np.float(my_image[i][0])
noise_image = (my_image + noise) / 255.0
noise_image = np.clip(noise_image, 0.0, 1.0)
prob, caches = L_model_forward(noise_image, parameters)
ppp, ccc = L_model_forward(my_image / 255.0, parameters)
print(prob, ppp)
my_predicted_image = predict(noise_image, my_label_y, parameters)
ad_image = noise_image.reshape(64, 64, 3)
#plt.imshow(image)
show_image = my_image.reshape(64, 64, 3)
noise_show = noise.reshape(64, 64, 3)
plt.subplot(221)
plt.imshow(show_image)
plt.subplot(222)
plt.imshow(ad_image)
plt.subplot(223)
plt.imshow(noise_show)
plt.show()
print ("y = " + str(np.squeeze(my_predicted_image)) + ", your L-layer model predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")