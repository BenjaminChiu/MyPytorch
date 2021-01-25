import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utils.d2lzh_pytorch as d2l
import numpy as np
import sys
import time

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化模型参数
num_inputs = 784
num_outputs = 10

W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)

W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

# 直接根据数值创建1个张量
X = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 中括号外面的逗号为一行的 分行标志，一行里面有多少个数为多少个列
print(X)
print(X[0, :])  # 索引第1行
print(X[1, :])  # 索引第2行
print(X[:, 0])  # 索引第1列
print(X[:, 1])  # 索引第2列
print(X.sum(dim=0, keepdim=True))  # dim=1 同一行，dim=0 同一列
print(X.sum(dim=1, keepdim=True))


# 实现softmax运算，即softmax激活函数，加装在神经元的后面
def softmax(X):
    X_exp = X.exp()     # e为底的x次方
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  # 这里应用了广播机制？


# 定义模型
# torch.mm(a,b)矩阵a 和 b矩阵相乘；比如a的维度是(1, 2)，b的维度是(2, 3)，返回的就是(1, 3)的矩阵
# torch.mul()矩阵对应位相乘，a和b的维度必须相等，比如a的维度是(1, 2)，b的维度是(1, 2)，返回的仍是(1, 2)的矩阵
def net(X):
    return softmax(torch.mm(X.view((-1, num_inputs)), W) + b)


# 定义损失函数
def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))


def accuracy(y_hat, y):
    return (y_hat.argmax(dim=1) == y).float().mean().item()


# 本函数已保存在d2lzh_pytorch包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中描述
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        # argmax(): 在有多个最大值的情况下，只返回第一个出现的最大值的位置。
        # y_hat.argmax(dim=1)返回矩阵y_hat每行中最大元素的索引，且返回结果与变量y形状相同。
        n += y.shape[0]
    return acc_sum / n


# 模型训练
num_epochs, lr = 5, 0.1


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            if optimizer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                optimizer.step()  # “softmax回归的简洁实现”一节将用到

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size, [W, b], lr)


X, y = iter(test_iter).next()

true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

d2l.show_fashion_mnist(X[0:9], titles[0:9])

print("done!")