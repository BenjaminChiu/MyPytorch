# 网易云课堂 实验楼
# 可以理解为CNN的一个初级实现
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        # 1个输入图片通道，6个输出通道，3 * 3的卷积核
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        # an affine operation: y = Wx + b
        # 6*6 from image dimension
        # 16是最后的输出形状，6*6 是图片的维度形状
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 最大池化 over 一个（2， 2）的窗口
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 如果池化层是正方形， 只需要一个数字
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = x.view(-1, self.num_flat_features(x))       # 调用了下面的函数，作用：输出维度总和
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        # 除batch维度外的 所有维度？
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    # 模型中必须要定义 forward 函数，backward 函数（用来计算梯度）会被 autograd 自动创建。
    # 可以在 forward 函数中使用任何针对 Tensor 的操作。

net = Net()

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight


input = torch.randn(1, 1, 32, 32)
out = net(input)

net.zero_grad()
out.backward(torch.randn(1, 10))

output = net(input)
target = torch.randn(10)  # 随机值作为样例
target = target.view(1, -1)  # 使 target 和 output 的 shape 相同
criterion = nn.MSELoss()

loss = criterion(output, target)


print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

net.zero_grad()  # 清除梯度

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

import torch.optim as optim

# 创建优化器
optimizer = optim.SGD(net.parameters(), lr=0.01)

# 执行一次训练迭代过程
optimizer.zero_grad()  # 梯度置零
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()  # 更新