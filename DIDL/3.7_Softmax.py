import torch
from torch import nn
from torch.nn import init
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import utils.d2lzh_pytorch as d2l
import numpy as np
import sys
import time
from collections import OrderedDict

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs = 784
num_outputs = 10


# 转换 样本中(batch_size, 28, 28)数据为 (batch_size,784)

class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)


# 定义模型
class LinearNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)

    def forward(self, x):
        y = self.linear(x.view(x.shape[0], -1))  # x torch.size([256, 1, 28, 28])    256个展开，     压平 卷积 到全连接 =》一维
        return y


net = LinearNet(num_inputs, num_outputs).cuda()

net = nn.Sequential(
    OrderedDict([
        ('flatten', FlattenLayer()),
        ('linear', nn.Linear(num_inputs, num_outputs))
    ])
)
# 初始化模型权重
init.normal_(net.linear.weight, mean=0, std=0.01)
init.constant_(net.linear.bias, val=0)

# 定义损失函数 、 优化器
loss = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

start = time.time()

num_epochs = 5
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None, None, optimizer)

# 封装一个 通用的 cuda 训练函数 取代d2l.train_ch3


print('%.2f sec' % (time.time() - start))
print("done!")
