import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from ConvolutionalNeuralNetwork.NeuralStyleTransfer.nst_utils import *
import numpy as np
import tensorflow as tf

# 1. Transfer Learning -- using vgg-19
# model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")

# 2. Neural Style Transfer
# 2.1 Computing the content cost

# content_image = scipy.misc.imread("images/louvre.jpg")
# imshow(content_image)

"""
3 steps to implement this function:
a. retrieve dimensions from a_G
b. unroll a_C and a_G
c. compute the content cost
"""

def compute_cost(a_C, a_G):
    """
    Computes the content cost

    Arguments:
    :param a_C: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content the image C
    :param a_G: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content the image G

    :return:
    J_content -- scalar that you compute using equation above
    """

    # Retrieve dimensions from a_G
    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    # Reshape a_C and a_G
    a_C_unrolled = tf.transpose(tf.reshape(a_C, [n_H * n_W, n_C]))
    a_G_unrolled = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    # compute the cost with tensorflow
    J_content = 1. / (4. * n_H * n_W * n_C) * tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))

    return J_content

'''
tf.reset_default_graph()

with tf.Session() as sess:
    tf.set_random_seed(1)
    a_C = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)
    a_G = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)
    J_content = compute_cost(a_C, a_G)
    print("J_content = " + str(J_content.eval()))
'''

# 2.2 Computing the style cost
# style_image = scipy.misc.imread("images/monet_800600.jpg")      # (600, 800, 3)
# imshow(style_image)
# plt.show()

# 2.2.1 Style matrix

def gram_matrix(A):
    """
    Arguments:
    :param A: matrix of shape (n_C, n_H * n_W)

    :return:
    GA -- gram matrix of A, of shape (n_C, n_C)
    """

    GA = tf.matmul(A, tf.transpose(A))

    return GA

'''
tf.reset_default_graph()

with tf.Session() as sess:
    tf.set_random_seed(1)
    A = tf.random_normal([3, 2 * 1], mean = 1, stddev = 4)
    GA = gram_matrix(A)

    print("GA = " + str(GA.eval()))
'''

# 2.2.2 Style cost

"""
3 steps to implement this function:
a. retrieve dimensions from the hidden layer activations a_G
b. Unroll the hidden layer activations a_S and a_G into 2D matrices
c. compute the style matrix of the images S and G
d. compute the style cost
"""

def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    :param a_S: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    :param a_G: tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    :return:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation
    """

    m, n_H, n_W, n_C = a_G.get_shape().as_list()

    a_S = tf.transpose(tf.reshape(a_S, [n_H * n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, [n_H * n_W, n_C]))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    J_style_layer = 1. / (4. * n_C * n_C * n_H * n_H * n_W * n_W) * tf.reduce_sum(tf.square(tf.subtract(GS, GG)))

    return J_style_layer

'''
tf.reset_default_graph()

with tf.Session() as sess:
    tf.set_random_seed(1)
    a_S = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)
    a_G = tf.random_normal([1, 4, 4, 3], mean = 1, stddev = 4)
    J_style_layer = compute_layer_style_cost(a_S, a_G)

    print("J_style_layer = " + str(J_style_layer.eval()))
'''

# 2.2.3 Style weights

# "merge" several different layers

STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2),
]

def compute_style_cost(model, STYLE_LAYERS):
    """
    computes the overall style cost from several chosen layers

    Arguments:
    :param model: our tensorflow model
    :param STYLE_LAYERS: a python list containing:
                            - the name of the layers we would like to extract style from
                            - a coefficient for each of them
    :return:
    J_style -- tensor representing a scalar value, style cost defined above by equation
    """

    # initialize the overall style cost
    J_style = 0.

    for layer_name, coeff in STYLE_LAYERS:

        # select the output tensor of the currently selected layer
        out = model[layer_name]

        # set a_S to be the hidden layer activation from the layer we have selected, by running the session on out
        a_S = sess.run(out)

        # set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
        # and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
        # when we run the session, this will be the activations drawn from the appropriate layer, with G as input
        a_G = out

        # compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Add coeff * J_style_layer of this layer to overall style cost
        J_style += coeff * J_style_layer

    return J_style

# Note: In the inner-loop of the for-loop above, a_G is a tensor and hasn't been evaluated yet. It will be evaluated and updated at
# each iteration when we run the TensorFlow graph in model_nn() below.

# 2.3 Defining the total cost to optimize

def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function

    Arguments:
    :param J_content: content cost coded above
    :param J_style: style cost coded above
    :param alpha: hyperparameter weighting the importance of the content cost
    :param beta: hyperparameter weighting the importance of the style cost

    :return:
    J -- total cost as defined by the formula above
    """

    J = alpha * J_content + beta * J_style

    return J

'''
tf.reset_default_graph()

with tf.Session() as sess:
    np.random.seed(3)
    J_content = np.random.randn()
    J_style = np.random.randn()
    J = total_cost(J_content, J_style)
    print("J = " + str(J))
'''

# 3. Solving the optimization problem

"""
steps:
a. create an interactive session
b. load the content image
c. load the style image
d. randomly initialize the image to be generated
e. load the VGG19 model
f. build the tensorflow graph
    i. run the content image through the VGG19 model and compute the content cost
    ii. run the style image through the VGG19 model and compute the style cost
    iii. compute the total cost
    iiii. define the optimizer and the learning rate
g. initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step
"""

# reset the graph
tf.reset_default_graph()

# start interactive session
sess = tf.InteractiveSession()

# load, reshape, and normalize C
content_image = scipy.misc.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)

# load, reshape, and normalize S
style_image = scipy.misc.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)

# initialize the G, as a noisy image created from the content_image
generated_image = generate_noise_image(content_image)

# load VGG19 model
model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
print(model)

# assign
# 1. assign the content image to be the input to the VGG model
# 2. set a_C to be the tensor giving the hidden layer activation for layer "conv4_2"
# 3. set a_G to be the tensor giving the hidden layer activation for the same layer
# 4. compute the content cost using a_C and a_G

# assign the content image to be the input of the VGG model
sess.run(model['input'].assign(content_image))
"""
这里的sess.run是指把content_image赋值到model['input']中，而model['input']是定义的一个variable，assign的使用条件也是variable
下面两行的作用是取出中间层，然后执行sess.run得到这一层的输出结果，也就是输入C图片之后中间层的输出结果，作为a_C
再下面一行是给a_G一个临时变量，将a_G指向相同layer的输出，但是注意这里并没有run，也就是说并没有实际得到a_G的结果
虽然下面一直都使用了a_G来计算J_content，但是都没有run，所以保留的是临时变量
一直到model_nn中，将input_image赋值到model的输入中，这时候才真正将G的值写入图中，然后执行sess.run(optimizer)
这时会将所有的变量均赋值并执行，也就是说会执行total_cost, 然后检查两个损失函数，然后检查损失函数中a_G没赋值，然后检查到out，然后执行run(out)给a_G赋值
至于为什么这个网络优化的不是参数而是输入，optimizer会将定义的loss给优化，默认会优化全局的variable，那么这里的variable就是model的input，由于我们给input
赋了值，所以网络就以此为初始值，不断迭代优化input
"""
# select the output tensor of layer conv4_2
out = model['conv4_2']

# set a_C to be the hidden layer activation from the layer we have selected
a_C = sess.run(out)

# set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name]
# and isn't evaluated yet. Later in the code, we'll assign the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input
a_G = out

# compute the content cost
J_content = compute_cost(a_C, a_G)

# assign the input of the model to be the "style" image
sess.run(model['input'].assign(style_image))

# compute the style cost
J_style = compute_style_cost(model, STYLE_LAYERS)

# total cost
J = total_cost(J_content, J_style, 10, 40)

# define optimizer
optimizer = tf.train.AdamOptimizer(2.0)

# define train_step
train_step = optimizer.minimize(J)

# implement the model_nn() function: 1. initializes the variables of the tensorflow graph, 2. assigns the input image (initial generated image)
# as the input of the VGG19, 3. runs the train_step for a large number of steps

def model_nn(sess, input_image, num_iterations = 200):

    # Initialize global variables
    sess.run(tf.global_variables_initializer())

    # Run the noisy input image (initial generated image) through the model
    sess.run(model['input'].assign(input_image))

    for i in range(num_iterations):

        # Run the session on the train_step to minimize the total cost
        sess.run(train_step)

        # Compute the generated image by running the session on the current model['input']
        generated_image = sess.run(model['input'])

        # print every 20 iteration
        if i % 20 == 0:
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))

            save_image("output/" + str(i) + ".png", generated_image)

    save_image("output/generate_image.jpg", generated_image)

    return generated_image

model_nn(sess, generated_image)