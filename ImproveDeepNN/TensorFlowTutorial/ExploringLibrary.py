import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from ImproveDeepNN.TensorFlowTutorial.tf_utils import *

np.random.seed(1)

#0 example
'''
y_hat = tf.constant(36, name='y_hat')
y = tf.constant(39, name='y')

loss = tf.Variable((y - y_hat) ** 2, name='loss')

init = tf.global_variables_initializer()

x = tf.placeholder(tf.int64, name='x')
with tf.Session() as session:
    session.run(init)
    print(session.run(loss))
    print(session.run(2 * x, feed_dict={x:3}))
    session.close()
'''

'''
Writing and running programs in TensorFlow has the following steps:

Create Tensors (variables) that are not yet executed/evaluated.
Write operations between those Tensors.
Initialize your Tensors.
Create a Session.
Run the Session. This will run the operations you’d written above.

'''

#1 linear function

'''
def linear_function():
    """
        Implements a linear function:
            Initializes W to be a random tensor of shape (4,3)
            Initializes X to be a random tensor of shape (3,1)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- runs the session for Y = WX + b
    """

    np.random.seed(1)

    X = tf.constant(np.random.randn(3, 1), name='X')
    W = tf.constant(np.random.randn(4, 3), name='W')
    b = tf.constant(np.random.randn(4, 1), name='b')

    Y = tf.add(tf.matmul(W, X), b)

    sess = tf.Session()
    result = sess.run(Y)

    sess.close()
    return result

print("result = " + str(linear_function()))
'''

#2 sigmoid computing

'''
method 1:
sess = tf.Session()
# Run the variables initialization (if needed), run the operations
result = sess.run(..., feed_dict = {...})
sess.close() # Close the session

method 2:
with tf.Session() as sess: 
    # run the variables initialization (if needed), run the operations
    result = sess.run(..., feed_dict = {...})
    # This takes care of closing the session for you :)
'''

def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    results -- the sigmoid of z
    """

    x = tf.placeholder(tf.float32, name='x')
    sigmoid = tf.sigmoid(x)

    with tf.Session() as sess:
        result = sess.run(sigmoid, feed_dict={x: z})

    return result

'''
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))
'''

#3 computing the cost

def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy

    Arguments:
    logits -- vector containing z, out of the last linear unit (before the final sigmoid activation)
    labels -- vector of labels y (1 or 0)

    Note: What we've been calling "z" and "y" in this class are respectively called "logits" and "labels"
    in the TensorFlow documentation. So logits will feed into z, and labels into y.

    Returns:
    cost -- runs the session of the cost (formula (2))
    """

    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    with tf.Session() as sess:
        cost = sess.run(cost, feed_dict={z: logits, y: labels})

    return cost

'''
logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))
'''

#4 using one hot encodings

def one_hot_matrix(labels, C):
    """
    Creates a matrix where the i-th row corresponds to the ith class number and the jth column
                     corresponds to the jth training example. So if example j had a label i. Then entry (i,j)
                     will be 1.

    Arguments:
    labels -- vector containing the labels
    C -- number of classes, the depth of the one hot dimension

    Returns:
    one_hot -- one hot matrix
    """

    C = tf.constant(value = C, name='C')

    one_hot_matrix = tf.one_hot(labels, C, axis=0)

    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)

    return one_hot

'''
labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = " + str(one_hot))
'''

#5 initialize with zeros and ones

def ones(shape):
    """
    Creates an array of ones of dimension shape

    Arguments:
    shape -- shape of the array you want to create

    Returns:
    ones -- array containing only ones
    """

    ones = tf.ones(shape)

    with tf.Session() as sess:
        ones = sess.run(ones)

    return ones

'''
print ("ones = " + str(ones([3])))
'''