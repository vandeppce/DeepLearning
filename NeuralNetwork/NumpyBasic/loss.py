import numpy as np
# l1 loss
def loss_function(y_hat, y):
    loss1 = np.sum(np.abs(y - y_hat))
    loss2 = np.sum((y - y_hat) ** 2)         #元素乘方
    return loss1, loss2

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
loss1, loss2 = loss_function(yhat, y)
print("L1 = " + str(loss1))
print("L2 = " + str(loss2))




