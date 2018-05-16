import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Add
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from ConvolutionalNeuralNetwork.KerasTutorial.resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

# 1. The identity block ---- for when the dimension of input and output is same


def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block

    Arguments:
    :param X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param f: integer, specifying the shape of the middle CONV's window for the main path
    :param filters: python list of integers, defining the number of filters in the CONV layers of the main path
    :param stage: integer, used to name the layers, depending on their position in the network
    :param block: string/character, used to name the layers, depending on their position in the network

    :return:
    X: output of the identity block, tensor of shape(n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # retrieve filters
    F1, F2, F3 = filters

    # save the input value, you'll need this layer to add back to the main path
    X_shortcut = X

    # first component of main path
    X = Conv2D(F1, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # second component of main path
    X = Conv2D(F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # third component of main path
    X = Conv2D(F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # final step: add shortcut value to main path, and pass it through a relu activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

'''
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = identity_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))
'''

# 2. The convolutional block ---- for when the dimension of input doesn't match dimension of output

def convolutional_block(X, f, filters, stage, block, s = 2):
    """
    Implementation of the convolutional block

    Arguments:
    :param X: input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    :param f: integer, specifying the shape of the middle CONV's window for the main path
    :param filters: python list of integers, defining the number of filters in the CONV layers of the main patu
    :param stage: integer, used to name the layers, depending on their position in the network
    :param block: string/character, used to name the layers, depending on their position in the network
    :param s: integer, specifying the stride to be used

    :return:
    X: output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # retrieve filters
    F1, F2, F3 = filters

    # save the input value
    X_shortcut = X

    # first component of main path
    X = Conv2D(F1, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '2a',  padding = "valid", kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # second component of main path
    X = Conv2D(F2, kernel_size=(f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same', kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # third component of main path
    X = Conv2D(F3, kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid', kernel_initializer=glorot_uniform(0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # shortcut path
    X_shortcut = Conv2D(F3, kernel_size=(1, 1), strides=(s, s), name=conv_name_base + '1', kernel_initializer=glorot_uniform(0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # final step: add shortcut value to main path, and pass it through a relu activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

'''
tf.reset_default_graph()

with tf.Session() as test:
    np.random.seed(1)
    A_prev = tf.placeholder("float", [3, 4, 4, 6])
    X = np.random.randn(3, 4, 4, 6)
    A = convolutional_block(A_prev, f = 2, filters = [2, 4, 6], stage = 1, block = 'a')
    test.run(tf.global_variables_initializer())
    out = test.run([A], feed_dict={A_prev: X, K.learning_phase(): 0})
    print("out = " + str(out[0][1][1][0]))
'''

# 3. Building ResNet model (50 layers)

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-padding
    X = ZeroPadding2D((3, 3))(X_input)

    # stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # stage 2
    X = convolutional_block(X, f = 3, filters=[64, 64, 256], stage=2, s = 1, block='a')
    X = identity_block(X, f = 3, filters=[64, 64, 256], stage=2, block='b')
    X = identity_block(X, f = 3, filters=[64, 64, 256], stage=2, block='c')

    # stage 3
    X = convolutional_block(X, f = 3, filters=[128, 128, 512], stage=3, block='a', s = 2)
    X = identity_block(X, f = 3, filters=[128, 128, 512], stage=3, block='b')
    X = identity_block(X, f = 3, filters=[128, 128, 512], stage=3, block='c')
    X = identity_block(X, f = 3, filters=[128, 128, 512], stage=3, block='d')

    # stage 4
    X = convolutional_block(X, f = 3, filters=[256, 256, 1024], stage=4, block='a', s=2)
    X = identity_block(X, f = 3, filters=[256, 256, 1024], stage=4, block='b')
    X = identity_block(X, f = 3, filters=[256, 256, 1024], stage=4, block='c')
    X = identity_block(X, f = 3, filters=[256, 256, 1024], stage=4, block='d')
    X = identity_block(X, f = 3, filters=[256, 256, 1024], stage=4, block='e')
    X = identity_block(X, f = 3, filters=[256, 256, 1024], stage=4, block='f')

    # stage 5
    X = convolutional_block(X, f = 3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='b')
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block='c')

    # avgpool
    X = AveragePooling2D(pool_size=(2, 2), name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

'''
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
'''

# create model
model = ResNet50(input_shape=(64, 64, 3), classes = 6)

# compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

'''
# train the model
model.fit(X_train, Y_train, epochs=2, batch_size=32)

preds = model.evaluate(X_test, Y_test)
print("loss = " + str(preds[0]))
print("test accuracy = " + str(preds[1]))

img_path = 'images/my_image1.jpg'
img = image.load_img(img_path, target_size=(64, 64))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
print('Input image shape:', x.shape)
my_image = scipy.misc.imread(img_path)
imshow(my_image)
print("class prediction vector [p(0), p(1), p(2), p(3), p(4), p(5)] = ")
print(model.predict(x))
'''

model.summary()

