import numpy as np
import matplotlib.pyplot as plt
from NeuralNetwork.LogisticRegression.lr_utils import load_dataset

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()     #加载数据

index = 25
plt.imshow(train_set_x_orig[index])     #示例
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

m_train = train_set_x_orig.shape[0]     #训练集大小
m_test = test_set_x_orig.shape[0]       #测试集大小
num_px = train_set_x_orig.shape[1]      #像素

#打印各矩阵对维度
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("classes shape: " + str(classes.shape))

m_size = train_set_x_orig.shape[1] * train_set_x_orig.shape[2] * train_set_x_orig.shape[3]
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1)    #等价于 reshape(m_train, m_size)
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1)

train_set_x_flatten = train_set_x_flatten.T                    #将样本转化为列向量
test_set_x_flatten = test_set_x_flatten.T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

#standardize
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255