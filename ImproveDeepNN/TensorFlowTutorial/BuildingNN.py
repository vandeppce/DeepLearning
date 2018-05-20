import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from ImproveDeepNN.TensorFlowTutorial.tf_utils import *
import scipy
from PIL import Image
from scipy import ndimage


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()
#print(X_train_orig.shape, Y_train_orig.shape, classes.shape)

'''
index = 0
plt.imshow(X_train_orig[index])
print("y = " + str(np.squeeze(Y_train_orig[:, index])))
plt.show()
'''

# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

'''
print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
'''

'''
Your goal is to build an algorithm capable of recognizing a sign with high accuracy. 
To do so, you are going to build a tensorflow model that is almost the same as one you have previously built in numpy for cat recognition 
(but now using a softmax out). It is a great occasion to compare your numpy implementation to the tensorflow one.

The model is LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX. 
The SIGMOID out layer has been converted to a SOFTMAX. 
A SOFTMAX layer generalizes SIGMOID to when there are more than two classes.
'''

#1 create placeholders

def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    n_y -- scalar, number of classes (from 0 to 5, so -> 6)

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])

    return X, Y

'''
X, Y = create_placeholders(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))
'''

#2 initializing the parameters

def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    tf.set_random_seed(1)

    W1 = tf.get_variable("W1", [25, 12288], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [6, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [6, 1], initializer=tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3
                  }
    return parameters

'''
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
'''

#3 forward propagetion
'''
It is important to note that the forward propagation stops at z3. 
The reason is that in tensorflow the last linear layer out is given as input to the function computing the loss. 
Therefore, you don’t need a3!
'''

def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the out of the last LINEAR unit
    """

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3

    return Z3

'''
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))
'''

#You may have noticed that the forward propagation doesn’t out any cache. You will understand why below, when we get to back propagation.

#4 compute cost

'''
- It is important to know that the “logits” and “labels” inputs of tf.nn.softmax_cross_entropy_with_logits 
are expected to be of shape (number of examples, num_classes).  (1080, 6)
We have thus transposed Z3 and Y for you. 
- Besides, tf.reduce_mean basically does the summation over the examples.
'''

def compute_cost(Z3, Y):
    """
    Computes the cost

    Arguments:
    Z3 -- out of forward propagation (out of the last LINEAR unit), of shape (6, number of examples)
    Y -- "true" labels vector placeholder, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    logits = tf.transpose(Z3)            #(6, 1080) -> (1080, 6)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost

'''
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))
'''

#5 backward propagetion & parameter updates

'''
After you compute the cost function. You will create an “optimizer” object. 
You have to call this object along with the cost when running the tf.session. 
When called, it will perform an optimization on the given cost with the chosen method and learning rate.

for instance, the GD the optimizer would be:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)

to make the optimization you will do:
    _, c = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
    
Note When coding, we often use _ as a “throwaway” variable to store values that we won’t need to use later. 
Here, _ takes on the evaluated value of optimizer, 
which we don’t need (and c takes the value of the cost variable).
'''

def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (out size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (out size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : out size
    costs = []  # To keep track of the cost

    #defining computing graphs

    #1 create placeholders of (nx, ny)
    X, Y = create_placeholders(n_x, n_y)

    #2 initialize parameters
    parameters = initialize_parameters()

    #3 forward propagation
    Z3 = forward_propagation(X, parameters)
    #Z3 = tf.nn.softmax(forward_propagation(X, parameters))

    #4 computing cost
    cost = compute_cost(Z3, Y)

    #5 back propagation
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    #initialize all variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):

            epoch_cost = 0.
            seed += 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            num_minibatches = int(m / minibatch_size)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch

                #run the graph on the minibatch

                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})

                epoch_cost += minibatch_cost / num_minibatches

            if print_cost and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        #save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        #calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        #calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))              #eval()是sess.run()的另一种写法

        return parameters

parameters = model(X_train, Y_train, X_test, Y_test)

my_image = "thumbs_up.jpg"
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))