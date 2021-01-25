# -*- coding: utf-8 -*-
# @Desc    : pytorch官方教程 迁移学习
# @Time    : 2020-11-09 16:07
# @Author  : tank boy
# @File    : 4_TransferLearning.py


# License: BSD
# Author: Sasank Chilamkurthy

from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()  # interactive mode

# ==============================加载数据======================================
# 在训练集数据扩充和归一化
# 在验证集上仅需要归一化

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),  # 随机裁剪一个area，然后resize
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

data_dir = 'D:\DeepLearning_Data\hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['train', 'val']}

dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=4)
               for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x])
                 for x in ['train', 'val']}

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# ==============================加载数据======================================



# ===========可视化部分图像数据=========================
def imshow(inp, title=None):
    '''Imshow for Tensor.'''
    inp = inp.numpy().transpose((1, 2, 0))  # 按照轴重新排列？
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean # ???
    inp = np.clip(inp, 0, 1)

    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.pause(0.001)        # 暂停一下，使plots 更新








# 4 训练模型
# 使用一个通用函数来训练模型。
# 参数scheduler是一个来自 torch.optim.lr_scheduler的学习速率调整类的对象(LR scheduler object)
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('_' * 10)

        # 每个epoch都有一个训练和验证阶段
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()       # set model to training mode
            else:
                model.eval()        # 设为验证状态

            running_loss = 0.0
            running_corrects = 0

            # 迭代数据
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)      # 都转入cuda

                # 设置 零参数梯度
                optimizer.zero_grad()

                # 前向传播
                # 只有训练的话，跟踪历史
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)     # 传入数据，拿到预测数据
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)       # 拿到损失

                    # 后向传播 + 仅在训练阶段进行优化
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计 评估模型
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            # 本批次
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # 深度复制mo
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since

    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    return model




# 5.可视化模型的预测结果
#一个通用的展示少量预测图片的函数
def visualize_model(model, num_images=6):
    was_training = model.training   # 标注模型状态？
    model.eval()        # 设为验证状态
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, lables) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            lables = lables.to(device)

            outputs = model(inputs)     # 传入数据，拿到预测数据
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')      # 关闭坐标轴
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return

        model.train(model=was_training)










# 主函数
if __name__ == '__main__':
    # 测试可视化图像数据
    # inputs, classes = next(iter(dataloaders['train']))
    # # 批量制作网格
    # out = torchvision.utils.make_grid(inputs)
    #
    # imshow(out, title=[class_names[x] for x in classes])
    # 测试可视化图像数据====END=====






    # 6.场景1：微调ConvNet
    # 加载预训练模型 并 重置最终完全连接的图层
    # model_ft = models.resnet18(pretrained=True)
    # num_ftrs = model_ft.fc.in_features
    # model_ft.fc = nn.Linear(num_ftrs, 2)    # 设置最终的全连接层
    #
    # model_ft = model_ft.to(device)
    #
    # criterion = nn.CrossEntropyLoss()   # 确定损失函数
    #
    # # 观察所有参数都正在优化
    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    #
    # # 每7个epochs衰减LR通过设置gamma=0.1
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    #
    #
    # # 训练和评估模型
    # # （1）训练模型 该过程在CPU上需要大约15-25分钟，但是在GPU上，它只需不到一分钟。
    # model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=6)
    # # （2）模型评估效果可视化
    # visualize_model(model_ft)
    # plt.show()  # 打印图片
    # 6.场景1：微调ConvNet ==========END=====







    # 场景2：ConvNet作为固定特征提取器

    # 在这里需要冻结除最后一层之外的所有网络。通过设置requires_grad == False来冻结参数，
    # 冻结目的：在反向传播backward()的时候他们的梯度就不会被计算。
    model_conv = models.resnet18(pretrained=True)
    for param in model_conv.parameters():
        param.requires_grad = False         # 遍历 冻结所有参数


    # 新构造的modules（模块、层） 参数默认不冻结（requires_grad=True）
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)    # 设置最终的全连接层
    model_conv = model_conv.to(device)  # 网络定义完成、转入cuda运行

    criterion = nn.CrossEntropyLoss()   # 确定损失函数

    # 显然，只有最后一层会被优化；之前层的参数不会动
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    # 每7个周期，学习率将会衰减一次
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)


    # 训练和评估
    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=25)
    # 模型评估，效果可视化
    visualize_model(model_conv)
    plt.ioff()
    plt.show()


