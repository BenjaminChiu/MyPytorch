import torch
from torch import nn


# 二维卷积 互相关运算，未引入padding

def corr2d(X, k):
    h, w = k.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))  # shape[0] 获取张量的行数，确定输出Y的形状

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # print(X[i:i + h, j: j+w])
            print()
            Y[i, j] = (X[i:i + h, j: j + w] * k).sum()

    return Y


X = torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
K = torch.tensor([[0, 1], [2, 3]])
Y = corr2d(X, K)

print()
print(X[2, 1])

print(Y)
