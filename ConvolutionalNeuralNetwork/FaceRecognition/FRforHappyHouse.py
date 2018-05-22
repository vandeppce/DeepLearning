from keras.models import Sequential
from keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate
from keras.layers.core import Lambda, Flatten, Dense
from keras.initializers import glorot_uniform
from keras.engine.topology import Layer
from keras import backend as K
K.set_image_data_format('channels_first')
import cv2
import os
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from ConvolutionalNeuralNetwork.FaceRecognition.fr_utils import *
from ConvolutionalNeuralNetwork.FaceRecognition.inception_blocks_v2 import *

np.set_printoptions(threshold=np.nan)

# 0. Naive Face Verification

"""
The simplest way: compare the two images pixel-to-pixel. 
If the distance are less than a chosen threshold, it may be the same persion.
Performing poorly.
"""

# 1. Encoding face images into a 128-djmensional vector
# 1.1 Using an ConvNet to compute encodings

"""
1. using inception network
2. using 96x96 dimensional RGB images as inputs, channel_first, tensor of shape (m, nC, nH, nW) = (m, 3, 96, 96)
3. outputing a matrix of shape (m, 128) that encodes each input face image into a 128-dimensional vector
"""


FRmodel = faceRecoModel(input_shape=(3, 96, 96))
# print("Total Params: ", FRmodel.count_params())


# 1.2 The Triplet Loss

def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Arguments:
    :param y_true: true labels, required when you define a loss in Keras, you don't need it in this function
    :param y_pred: python list containing three objects:
                   anchor -- the encodings for the anchor images, of shape (None, 128)
                   positive -- the encodings for the positive images, of shape (None, 128)
                   negative -- the encodings for the negative images, of shape (None, 128)
    :param alpha: margin

    :return:
    loss -- real number, value of the loss
    """

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    # step 1, compute the (encoding) distance between the anchor and positive, sum over axis = -1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)  # why axis = 1 wrong, 个人猜想：
    # 在compile的时候还没有对模型喂数据，所以FRmodel的输出应该是一个(128)的数据，也就是说，只有1个维度(dimension)，而我们在测试的时候喂进去的数据是(None, 1),
    # 有两个维度，所以我们测试的时候可以使用axis=1，而compile的时候如果使用axis=1，那么因为只有一个dimension，也就自然无法reduce了，所以应该使用axis=-1，反正无论如何sum的都是最后一个维度

    # step 2, compute the (encoding) distance between the anchor and negative, sum over axis = 1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)

    # step 3, subtract the two previous distances and add alpha
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)

    # step 4, take the maximum of basic_loss and 0.0. sum over the training examples
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))

    return loss

'''
with tf.Session() as sess:
    tf.set_random_seed(1)
    y_true = (None, None, None)
    y_pred = (tf.random_normal([3, 128], mean=6, stddev=0.1, seed=1),
              tf.random_normal([3, 128], mean=1, stddev=1, seed=1),
              tf.random_normal([3, 128], mean=3, stddev=4, seed=1),)

    loss = triplet_loss(y_true, y_pred)

    print("loss = " + str(loss.eval()))
'''


# 2. loading the trained model
FRmodel.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
load_weights_from_FaceNet(FRmodel)

# 3. Applying the model
# 3.1 Face Verification

database = {}
database["danielle"] = img_to_encoding("images/danielle.png", FRmodel)
database["younes"] = img_to_encoding("images/younes.jpg", FRmodel)
database["tian"] = img_to_encoding("images/tian.jpg", FRmodel)
database["andrew"] = img_to_encoding("images/andrew.jpg", FRmodel)
database["kian"] = img_to_encoding("images/kian.jpg", FRmodel)
database["dan"] = img_to_encoding("images/dan.jpg", FRmodel)
database["sebastiano"] = img_to_encoding("images/sebastiano.jpg", FRmodel)
database["bertrand"] = img_to_encoding("images/bertrand.jpg", FRmodel)
database["kevin"] = img_to_encoding("images/kevin.jpg", FRmodel)
database["felix"] = img_to_encoding("images/felix.jpg", FRmodel)
database["benoit"] = img_to_encoding("images/benoit.jpg", FRmodel)
database["arnaud"] = img_to_encoding("images/arnaud.jpg", FRmodel)

def verify(image_path, identity, database, model):
    """
    Function that verifies if the person on the "image_path" image is "identity"

    :param image_path: path to an image
    :param identity: string, name of the persion you'd like to verify the identity. Has to be a resident of the Happy House
    :param database: python dictionary mapping names of allowed people's names (strings) to their encodings (vectors)
    :param model: your Inception model instance in Keras

    :return:
    dist -- distance between the image_path and the image of "identity" in the database
    door_open -- True, if the door should open. False otherwise
    """

    # step 1: compute the encoding for the image. Use img_to_encoding()
    encoding = img_to_encoding(image_path, model)

    # step 2: compute distance with identity's image. Use np.linalg.norm() 二范数
    dist = np.linalg.norm(encoding - database[identity])

    # step 3: compute whether the door should open
    if dist < 0.7:
        print("It's " + str(identity) + ", welcome home!")
        door_open = True
    else:
        print("It's not " + str(identity) + ", please go away")
        door_open = False

    return dist, door_open

verify("images/camera_0.jpg", "younes", database, FRmodel)
verify("images/camera_2.jpg", "kian", database, FRmodel)

# 3.2 Face Recognition

def who_is_it(image_path, database, model):
    """
    Implements face recognition for the Happy House by finding who is the person on the image_path image

    :param image_path: path to an image
    :param database: database containing image encodings along with the name of the person on the image
    :param model: your Inception model instance in Keras

    :return:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediciton for the person on image_path
    """

    # step 1: compute the target "encoding" for the image
    encoding = img_to_encoding(image_path, model)

    # step 2: find the closest encoding

    # Initialize "min_dist" to a large value, say 100
    min_dist = 100

    # Loop over the database dictionary's names and encodings
    for (name, db_enc) in database.items():
        # compute L2 distance between the target "encoding" and the current "emb" from the database
        dist = np.linalg.norm(encoding - db_enc)

        # check
        if dist < min_dist:
            min_dist = dist
            identity = name

    if min_dist > 0.7:
        print("Not in the database.")
    else:
        print("It's " + str(identity) + ", the distance is " + str(min_dist))

    return min_dist, identity

who_is_it("images/camera_0.jpg", database, FRmodel)

"""
Some ways to further improve the algorithm

- Put more images of each person (under different lighting conditions, 
taken on different days, etc.) into the database. 
Then given a new image, compare the new face to multiple pictures of the person. 
This would increase accuracy. 
- Crop the images to just contain the face, 
and less of the “border” region around the face. 
This preprocessing removes some of the irrelevant pixels around the face, 
and also makes the algorithm more robust.

What you should remember: 
- Face verification solves an easier 1:1 matching problem; face recognition addresses a harder 1:K matching problem. 
- The triplet loss is an effective loss function for training a neural network to learn an encoding of a face image. 
- The same encoding can be used for verification and recognition. Measuring distances between two images’ encodings 
allows you to determine whether they are pictures of the same person.
"""