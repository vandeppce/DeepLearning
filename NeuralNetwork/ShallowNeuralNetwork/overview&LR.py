import numpy as np
import matplotlib.pylab as plt
from NeuralNetwork.testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from NeuralNetwork.planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

X, Y = load_planar_dataset()
plt.scatter(X[0, :], X[1, :], c=np.squeeze(Y), s=40, cmap=plt.cm.Spectral)

shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

print("The shape of X is: " + str(shape_X))
print("The shape of Y is: " + str(shape_Y))
print("The number of training examples is: " + str(m))

clf = sklearn.linear_model.LogisticRegressionCV()    #使用交叉验证选择正则化参数C
clf.fit(X.T, Y.T)   #X和Y使用行矩阵，所以转置

plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

LR_predictions = clf.predict(X.T)
print("Accuracy of logistic regression: %d" % float((np.dot(Y, LR_predictions) + np.dot(1 - Y, 1 - LR_predictions)) / float(Y.size) * 100) + '%' + "(percentage of correctly labelled datapoints)")

#plt.show()

#LogisticRegression did not work well on non-linear model like "flower dataset"
