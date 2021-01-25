# pytorch官方教程：图像分类器

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

# 读取、归一化CIFAR10
# 图像预处理步骤


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 训练数据加载器
trainset = torchvision.datasets.CIFAR10(root="D:\DeepLearning_Data\CIFAR", train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

# 测试数据加载器
testset = torchvision.datasets.CIFAR10(root="D:\DeepLearning_Data\CIFAR", train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

# 图像类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# 可视化训练图像
def imshow(img):
    # 展示图像的函数
    img = img / 2 + 0.5  # 反向归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # 1 2 0 ???
    plt.show()


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


net = Net().to(DEVICE)

# 损失函数（分类问题，交叉熵）
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # momentum 动量

if __name__ == '__main__':

    #
    # # 获取随机数据
    # dataiter = iter(trainloader)
    # images, lables = dataiter.next()
    #
    # # 展示图像
    # imshow(torchvision.utils.make_grid(images))
    #
    # # 显示图像标签
    # print(' '.join('%5s' % classes[lables[j]] for j in range(4)))

    # 开始训练

    # 迭代一次
    for epoch in range(1):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):  # 这个枚举什么意思???
            # 获取输入
            inputs = data[0].to(DEVICE)
            lables = data[1].to(DEVICE)
            # 梯度设为 0，让优化器来设置梯度为0？
            optimizer.zero_grad()

            # 正向传播，反向传播，优化
            outputs = net(inputs)
            loss = criterion(outputs, lables)
            loss.backward()
            optimizer.step()

            # 打印状态信息
            running_loss += loss.item()
            if i % 200 == 199:
                print('[%d, %5d] loss: %.5f' %
                      (epoch + 1, i + 1, running_loss / 200))  # 没200次，得出本批次平均loss
                running_loss = 0.0

    print('结束训练')
    # #
    # # 保存训练后的模型
    PATH = 'D:\DeepLearning_Data\MyModel\cifar_net.pth'
    torch.save(net.state_dict(), PATH)

    check = enumerate(trainloader, 0)

    # 测试网络
    dataiter = iter(testloader)
    test = dataiter.next()
    images, lable_2 = test
    #
    # # 显示图片
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % classes[lable_2[j]] for j in range(4)))

    # outputs = net(images)
    #
    # _, predicted = torch.max(outputs, 1)
    #
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

    # 在整个数据集上的表现
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in testloader:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #
    # print('Accuracy of the network on the 10000 test images: %d %%' %
    #       (100 * correct / total))
