# -*- coding: utf-8 -*-
# @Desc    : 借助pytorch，实现手写数字的识别
# @Time    : 2020-12-05 15:47
# @Author  : tank boy
# @File    : MNIST_My.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms

BATCH_SIZE = 512  # 大概需要2G的显存
EPOCHS = 20  # 总共训练批次
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 熟悉 理解Compose

# 载入数据集
train_datasets = torchvision.datasets.MNIST(root='D:\DeepLearning_Data', train=True,
                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081,))]))
test_datasets = torchvision.datasets.MNIST(root='D:\DeepLearning_Data', train=False,
                                           transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081,))]))

train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=BATCH_SIZE, shuffle=True)
# 测试集
test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=BATCH_SIZE, shuffle=False)


# 定义模型
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 1*1*28*28
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(1, 10, 5),
        #     nn.functional.relu(),
        #     nn.MaxPool2d(2, 2),
        #
        #     nn.Conv2d(10, 20, 3),
        #     nn.functional.relu(),
        #     nn.MaxPool2d(2, 2)
        # )

        # self.fc = nn.Sequential(
        #     nn.Linear(20 * 10 * 10, 500),
        #     nn.functional.relu(),
        #
        #     nn.Linear(500, 10),  # 最后一层没有激活函数
        #     F.log_softmax(dim=1)
        # )

    def forward(self, x):
        in_size = x.size(0)

        out = self.conv1(x)  # 1* 10 * 24 *24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 1* 10 * 12 * 12

        out = self.conv2(out)  # 1* 20 * 10 * 10
        out = F.relu(out)

        out = out.view(in_size, -1)  # 1 * 2000
        out = self.fc1(out)  # 1 * 500
        out = F.relu(out)

        out = self.fc2(out)  # 1 * 10
        out = F.log_softmax(out, dim=1)

        # 思考 x经过 卷积层后的形状？

        # feature = self.conv(x)
        # out = self.fc(feature.view(x.shape[0], -1))

        return out


model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())
accumulation_steps = 8  # 手动清零梯度的超参数


# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target)

        loss = loss / accumulation_steps  # 计算平均loss
        loss.backward()

        if (++batch_idx % accumulation_steps) == 0:
            optimizer.step()  # 根据累计的梯度更新模型参数
            optimizer.zero_grad()  # 每个批次梯度置0

        if (batch_idx + 1) % accumulation_steps == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum')  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print("\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)
    ))


if __name__ == '__main__':
    for epoch in range(EPOCHS):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)
