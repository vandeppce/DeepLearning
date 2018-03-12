import numpy as np
import scipy
import matplotlib.pyplot as plt
from PIL import Image
import h5py
from scipy import ndimage
from NeuralNetwork.lr_utils import load_dataset

def initialize_with_zeros(x):
    w = np.zeros(shape=(x, 1))
    b = 0

    assert(w.shape == (x, 1))
    assert(isinstance(b, int) or isinstance(b, float))

    return w, b

def sigmoid(x):
    s = 1.0 / (1.0 + np.exp(-x))
    return s

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
        This function optimizes w and b by running a gradient descent algorithm

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of shape (num_px * num_px * 3, number of examples)
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
        num_iterations -- number of iterations of the optimization loop
        learning_rate -- learning rate of the gradient descent update rule
        print_cost -- True to print the loss every 100 steps

        Returns:
        params -- dictionary containing the weights w and bias b
        grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
        costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

        Tips:
        You basically need to write down two steps and iterate through them:
            1) Calculate the cost and the gradient for the current parameters. Use propagate().
            2) Update the parameters using gradient descent rule for w and b.
        """

    costs = []

    for i in range(num_iterations):
        m_size = X.shape[1]
        z = np.dot(w.T, X) + b
        A = sigmoid(z)

        cost = -1.0 / m_size * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        dw = 1.0 / m_size * np.dot(X, (A - Y).T)
        db = 1.0 / m_size * np.sum(A - Y)

        assert(dw.shape == w.shape)
        assert(db.dtype == float)

        w = w - learning_rate * dw
        b = b - learning_rate * db
        costs.append(cost)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {
        'w': w,
        'b': b
    }

    grads = {
        'dw': dw,
        'db': db
    }

    return params, grads, costs

def predict(w, b, X):
    '''
        Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

        Arguments:
        w -- weights, a numpy array of size (num_px * num_px * 3, 1)
        b -- bias, a scalar
        X -- data of size (num_px * num_px * 3, number of examples)

        Returns:
        Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
        '''

    m_size = X.shape[1]
    Y_prediction = np.zeros(shape=(1, m_size))

    z = np.dot(w.T, X) + b
    A = sigmoid(z)
    assert(A.shape == (1, m_size))

    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0
    assert(Y_prediction.shape == (1, m_size))

    return Y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # 1 initialize
    w, b = initialize_with_zeros(X_train.shape[0])

    # 2 propagation and gradient decent
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost = True)

    # 3 prediction
    prediction_w = params['w']
    prediction_b = params['b']

    Y_prediction_test = predict(prediction_w, prediction_b, X_test)
    Y_prediction_train = predict(prediction_w, prediction_b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_x_orig.shape[0]     #训练集大小
m_test = test_set_x_orig.shape[0]       #测试集大小
num_px = train_set_x_orig.shape[1]      #像素

train_set_x_flatten = train_set_x_orig.reshape(m_train, -1)    #等价于 reshape(m_train, m_size)
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1)

train_set_x_flatten = train_set_x_flatten.T                    #将样本转化为列向量
test_set_x_flatten = test_set_x_flatten.T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.show()
