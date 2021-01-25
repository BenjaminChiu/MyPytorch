"""
@Desc   : 文件注释
@Time   : 2020-12-07 14:11
@Author : tank boy
@File   : pip_update.py
@coding : utf-8
"""
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



# 使用GPU训练
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
PATH = 'D:\DeepLearning_Data\MyModel\cifar_net.pth'

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
# 测试数据加载器
testset = torchvision.datasets.CIFAR10(root="D:\DeepLearning_Data\CIFAR", train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# 图像类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 定义一个卷积神经网络
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution kernel
        # 1个输入图片通道，6个输出通道，3 * 3的卷积核
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # an affine operation: y = Wx + b
        # 6*6 from image dimension
        # 16是最后的输出形状，6*6 是图片的维度形状
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 最大池化 over 一个（2， 2）的窗口
        # x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # # 如果池化层是正方形， 只需要一个数字
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = F.max_pool2d(F.relu(self.conv3(x)), 4)
        x = x.view(-1, 16 * 5 * 5)  # -1 是指 优先匹配后面的列，行自动生成
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 载入已有模型
net = Net()
net.load_state_dict(torch.load(PATH))
net.to(DEVICE)

if __name__ == '__main__':
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(10):
        print('Accuracy of %5s : %2d %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))
