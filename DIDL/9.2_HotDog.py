# 微调 广泛运用在迁移学习中

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
import os
import sys
import utils.d2lzh_pytorch as d2l
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_dir = 'D:\DeepLearning_Data'
os.listdir(os.path.join(data_dir, "hotdog"))  # 包括两个数据集 train、test

# 图像通道归一化，由RGB三个通道的均值和方差来确定。深入理解这些值 怎么来的？
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_augs = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])

test_augs = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    normalize])

# 读取图像，使用实例 datasets


train_datasets = ImageFolder(os.path.join(data_dir, "hotdog/train"), transform=train_augs)
test_datasets = ImageFolder(os.path.join(data_dir, "hotdog/test"), transform=test_augs)

# 测试读取
# train_imgs = ImageFolder(os.path.join(data_dir, "hotdog/train"))
# test_imgs = ImageFolder(os.path.join(data_dir, "hotdog/test"))
#
# hotdogs = [train_imgs[i][0] for i in range(8)]
# not_hotdogs = [train_imgs[-i - 1][0] for i in range(8)]
# d2l.show_images(hotdogs + not_hotdogs, 2, 8, scale=2)
# plt.show()

# 定义、初始化模型

# 下载ResNet-18作为源模型（该模型使用ImageNet数据集预训练）
# pretrained=True 自动下载并加载预训练模型参数
pretrained_net = models.resnet18(pretrained=True)

print(pretrained_net.fc)

# 只获取fc层的参数，并且使用较大的学习率。其他层依然保存着预训练得到的参数
output_params = list(map(id, pretrained_net.fc.parameters()))
feature_params = filter(lambda p: id(p) not in output_params, pretrained_net.parameters())

lr = 0.01

# 其他层参数 由于是在有大量样本的预训练中得到的，保持低学习率调整
# 括号由小变大
optimizer = optim.SGD([{'params': feature_params},
                       {'params': pretrained_net.fc.parameters(), 'lr': lr * 10}],
                      lr=lr, weight_decay=0.001)


# 微调模型，工具函数
# num_epochs 训练次数
def train_fine_tuning(net, optimizer, batch_size=8, num_epochs=5):
    train_iter = DataLoader(train_datasets, batch_size, shuffle=True)
    test_iter = DataLoader(test_datasets, batch_size)
    loss = torch.nn.CrossEntropyLoss()
    d2l.train(train_iter, test_iter, net, loss, optimizer, DEVICE, num_epochs)


train_fine_tuning(pretrained_net, optimizer)
