import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from ConvolutionalNeuralNetwork.KerasTutorial.kt_utils import *
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

'''
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
'''

def model(input_shape):
    X_input = Input(input_shape)

    # zero-padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # Conv -> BN -> RELU block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # create model, this creates your keras model instance, you'll use this instance to train/test the model
    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model

def HappyModel(input_shape):
    """
    Implementation of the HappyModel

    Arguments:
    :param input_shape: shape of the images of the dataset

    :return:
    model: a Model() instance in Keras
    """

    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


### steps to train the model
# 1. create the model
happyModel = HappyModel(X_train.shape[1:])

# 2. compile the model
happyModel.compile(optimizer="Adam", loss="binary_crossentropy", metrics=["accuracy"])

# 3. train the model
happyModel.fit(x=X_train, y=Y_train, epochs=10, batch_size=32)

'''
# 4. test the model
preds = happyModel.evaluate(X_test, Y_test)

print()
print("Loss = " + str(preds[0]))
print("Test Accuracy = " + str(preds[1]))
'''

img_path = "images/my_image.jpg"

img = image.load_img(img_path, target_size=(64, 64))
imshow(img)
plt.show()

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

print(happyModel.predict(x))