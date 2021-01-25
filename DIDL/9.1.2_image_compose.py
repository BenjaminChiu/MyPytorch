"""
@Desc   : 图像增广
@Time   : 2020-12-11 22:23
@Author : tank boy
@File   : 9.1.2_image_compose.py
@coding : utf-8
"""
import time
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image

import sys
import utils.d2lzh_pytorch as d2l
import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# scale 表示图片是否缩放
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)  # 设置窗口尺寸，第1个参数为宽，第2个为高，单位英寸
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)
            axes[i][j].axes.get_yaxis().set_visible(False)
    return axes


# 使用图像增广训练模型
all_imges = torchvision.datasets.CIFAR10(train=True, root="D:\DeepLearning_Data\CIFAR", download=False)
# all_imges的每一个元素都是(image, label)
show_images([all_imges[i][0] for i in range(50)], 5, 10, scale=1)   # 5行 10列 1倍缩放

plt.show()

# 在这里我们只使用最简单的随机左右翻转。
# 此外，我们使用ToTensor将小批量图像转成PyTorch需要的格式，
# 即形状为(批量大小, 通道数, 高, 宽)、值域在0到1之间且类型为32位浮点数。
flip_aug = torchvision.transforms.Compose([
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()])

no_aug = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()])
#
# # 定义一个辅助函数来读取图像并应用图像增广
num_workers = 0 if sys.platform.startswith('win32') else 4  # 根据平台给线程数

def load_cifar10(is_train, augs, batch_size, root="D:\DeepLearning_Data\CIFAR"):
    dataset = torchvision.datasets.CIFAR10(root=root, train=is_train, transform=augs)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


# # 本函数已保存在d2lzh_pytorch包中方便以后使用
def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = d2l.evaluate_accuracy(test_iter, net)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))


def train_with_data_aug(train_augs, test_augs, lr=0.001):
    batch_size = 256
    train_iter = load_cifar10(True, train_augs, batch_size)
    test_iter = load_cifar10(False, test_augs, batch_size)

    net = d2l.resnet18(10)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    train(train_iter, test_iter, net, loss, optimizer, DEVICE, num_epochs=10)


train_with_data_aug(flip_aug, no_aug)
