import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from ImproveDeepNN.OptimizationMethods.opt_utils import *
from ImproveDeepNN.OptimizationMethods.testCases import *

plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

"""
There are two steps to build mini batches:
1. Shuffle: Create a shuffled version of the training set (X, Y) as shown below. 
            Each column of X and Y represents a training example. 
            Note that the random shuffling is done synchronously between X and Y. Such that after the shuffling the ith
            column of X is the example corresponding to the ith label in Y. 
            The shuffling step ensures that examples will be split randomly into different mini-batches.
            X, Y随机列混合，同步变换

2. Partition: Partition the shuffled (X, Y) into mini-batches of size mini_batch_size (here 64). 
              Note that the number of training examples is not always divisible by mini_batch_size. 
              The last mini batch might be smaller, but you don’t need to worry about this.
"""

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

    num_complete_minibatches = m // mini_batch_size  #地板除
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

'''
X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)
print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
'''
'''
What you should remember: 
- Shuffling and Partitioning are the two steps required to build mini-batches 
- Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.
'''

