"""
@Desc   : 单发多框检测
@Time   : 2020-12-16 23:01
@Author : tank boy
@File   : 9.7_SSD.py
@coding : utf-8
"""

import time

import torch
import torch.nn as nn
import torch.optim as optim

import utils.d2lzh_pytorch as d2l

COUNT = 0
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 类别预测层，输出通道：预测q+1个类别；因框架原因实现动态 输入通道数
def cls_predictor(num_anchors, num_classes, i):
    if i == 0:
        in_channel = 64
    else:
        in_channel = 128

    return nn.Conv2d(in_channel, num_anchors * (num_classes + 1), kernel_size=3, padding=1)


# 边界框预测层，输出通道：预测每个锚框的4个偏移量
def bbox_predictor(num_anchors, i):
    if i == 0:
        in_channel = 64
    else:
        in_channel = 128
    return nn.Conv2d(in_channel, num_anchors * 4, kernel_size=3, padding=1)


# 前向传播层
def forward(x, fun):
    return fun(x)


# 例子
# Y1 = torch.zeros((2, 8, 20, 20)), cls_predictor(5, 10))
# Y2 = forward(torch.zeros((2, 16, 10, 10)), cls_predictor(3, 10))
# 传入参数shape：（批量大小，通道数，img_H, img_W ）

# X1 = torch.zeros((2, 8, 20, 20))
# X2 = torch.zeros((2, 16, 10, 10))
#
# out1 = forward(X1, cls_predictor(5, 10, X1.shape[1]))
# out2 = forward(X2, cls_predictor(3, 10, X2.shape[1]))
#
# print(out1.size())
# print(out2.size())


# (批量大小, 通道数, 高, 宽) -> (批量大小, 高*宽*通道数)  .flatten(1) 表示保留第0位，其他维度展开合并
def flatten_pred(pred):
    return pred.transpose(1, 3).transpose(1, 2).flatten(start_dim=1)


# 连接多尺度预测框，使用torch.cat()
def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


# out3 = concat_preds([out1, out2])
# print(out3.size())


OutC = 0


# 高和宽 减半块，目的：减半代表着更高维度的卷积，更高维度的特征表达，生成的锚框自然是较大的
def down_sample(in_channels, num_channels, flag):
    """
    在全局角度下，因为除了第一次被调用外，其他情况需要
    标注是否是全局第一次调用、上次输出通道数（本次输入通道数）
    :param in_channels:
    :param num_channels:
    :param flag: 本函数全局 被调用标志
    :return:
    """
    global OutC
    mymodel = nn.Sequential()
    for i in range(2):
        if i != 0 or flag != 0:  # 不是第一次传进来了
            real_in_channels = OutC
        else:
            real_in_channels = in_channels
        mymodel.add_module('conv_' + str(i), nn.Conv2d(real_in_channels, num_channels, kernel_size=3, padding=1))
        mymodel.add_module('BN_' + str(i), nn.BatchNorm2d(num_channels))
        mymodel.add_module('relu_' + str(i), nn.ReLU())
        OutC = num_channels
    mymodel.add_module('pool', nn.MaxPool2d(2))
    return mymodel


# X3 = torch.zeros((2, 3, 20, 20))
# out4 = forward(X3, down_sample(X3.shape[1], 10))
# print(out4.size())


# 基础网络块
def base_net(in_channels):
    mymodel = nn.Sequential()
    i = 0
    for num_filters in [16, 32, 64]:
        mymodel.add_module('base_net_' + str(i), down_sample(in_channels, num_filters, i))
        i = i + 1
    return mymodel


# X8 = torch.zeros((2, 3, 256, 256))
# out5 = forward(X8, base_net(X8.shape[1]))
# print(out5.size())


# 完整模型
# 每个模块输出的特征图既用来生成锚框，又用来预测这些锚框的类别和偏移量。
def get_blk(i):
    if i == 0:
        blk = base_net(3)  # 第1模块为基础网络块
    elif i == 1:
        blk = down_sample(64, 128, 0)  # 2-4 都是宽高减半模块 （多尺度特征块）缺两个参数
    elif i == 4:
        blk = nn.AdaptiveAvgPool2d(1)  # 使用全局最大池化层 将高、宽降为1（多尺度特征块）
    else:
        blk = down_sample(128, 128, 0)
    return blk


# ======================开始构造模型===========================


# 前向传播
# 返回值：卷积计算输出的特征图Y，根据Y生成的当前尺度的锚框，基于Y预测的锚框类别 和 偏移量
def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    global COUNT
    print('当前count=', str(COUNT))
    Y = blk(X)
    anchors = d2l.MultiBoxPrior(Y, sizes=size, ratios=ratio)  # 根据每个像素点生成原始锚框
    cls_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    COUNT = COUNT + 1
    return Y, anchors, cls_preds, bbox_preds


sizes = [[0.2, 0.272], [0.37, 0.447], [0.54, 0.619], [0.71, 0.79], [0.88, 0.961]]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1


class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        for i in range(5):
            # 即赋值语句self.blk_i = get_blk(i)
            setattr(self, 'blk_%d' % i, get_blk(i))
            setattr(self, 'cls_%d' % i, cls_predictor(num_anchors, num_classes, i))
            setattr(self, 'bbox_%d' % i, bbox_predictor(num_anchors, i))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self, 'blk_%d' % i)即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X, getattr(self, 'blk_%d' % i), sizes[i], ratios[i],
                getattr(self, 'cls_%d' % i), getattr(self, 'bbox_%d' % i))
        # reshape函数中的0表示保持批量大小不变
        t = concat_preds(cls_preds)
        t1 = t.view(t.shape[0], -1, self.num_classes + 1)
        # t1=cls_preds shape[batch_size, 每张图片原始锚框数，待预测类别+背景类(1) ]
        return torch.cat(anchors, dim=1), t1, concat_preds(bbox_preds)


# ===========Test1===START============
# X = torch.zeros((32, 3, 256, 256))
# net = TinySSD(num_classes=1)
# # net.initialize()
# anchors, cls_preds, bbox_preds = net(X)
# ===========Test1===END============

# ===========Test2===START============
# def init_weights(m):
#     if type(m) == nn.Linear or type(m) == nn.Conv2d:
#         torch.nn.init.xavier_uniform_(m.weight)
#
#
# net = TinySSD(num_classes=1)
# net.apply(init_weights)
#
# X = torch.zeros((32, 3, 256, 256))
# anchors, cls_preds, bbox_preds = net(X)
# ===========Test2===END============

# print('output anchors:', anchors.shape)
# print('output class preds:', cls_preds.shape)
# print('output bbox preds:', bbox_preds.shape)

BATCH_SIZE = 2
edge_size = 256
# img_size, DATA_DIR 已经在其中写死
train_iter, val_iter = d2l.load_data_pikachu(BATCH_SIZE)

net = TinySSD(num_classes=1)
net.to(DEVICE)

# trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.2, 'wd': 5e-4})
optimizer = optim.SGD(net.parameters(), lr=0.2, momentum=0.9)  # momentum 动量
# 锚框类别损失函数（分类问题，交叉熵_softmax）
cls_loss = nn.CrossEntropyLoss()
# 锚框偏移量损失函数（回归问题 L1范数损失：预测值和真实值之差的绝对值）
bbox_loss = nn.L1Loss()


# 掩码变量bbox_masks令负类锚框、填充锚框不参与损失计算
def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    cls = cls_loss(cls_preds, cls_labels)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks)
    return cls + bbox


def cls_eval(cls_preds, cls_labels):
    # 由于类别预测结果放在最后一维，argmax需要指定最后一维     item()转为标量
    return (cls_preds.argmax(axis=-1) == cls_labels).float().sum().item()


# 因为使用了L1范数损失，用平均绝对误差 来评价边界框的预测结果
def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return ((bbox_labels - bbox_preds) * bbox_masks).abs().float().sum().item()


if __name__ == '__main__':
    # 训练模型， 没有评价测试数据集

    for epoch in range(20):
        acc_sum, mae_sum, n, m = 0.0, 0.0, 0, 0
        # train_iter.reset()  # 从头读取数据
        start = time.time()
        with torch.no_grad():
            for sample in train_iter:
                X = sample['image'].to(DEVICE)
                Y = sample['label'].to(DEVICE)
                print(Y.size())

                # 生成多尺度的锚框，为每个锚框预测类别和偏移量
                anchors, cls_preds, bbox_preds = net(X)
                # 为每个锚框标注类别和偏移量
                anchors = anchors.to(DEVICE)
                # 偏移量，框掩码、真实类别
                bbox_labels, bbox_masks, cls_labels = d2l.MultiBoxTarget(anchors, Y)
                # 根据类别和偏移量的预测和标注值计算损失函数
                loss = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks)

                loss.backward()
                optimizer.step()
                acc_sum += cls_eval(cls_preds, cls_labels)
                n += cls_labels.size()
                mae_sum += bbox_eval(bbox_preds, bbox_labels, bbox_masks)
                m += bbox_labels.size()

        if (epoch + 1) % 5 == 0:
            print('epoch %2d, class err %.2e, bbox mae %.2e, time %.1f sec' % (
                epoch + 1, 1 - acc_sum / n, mae_sum / m, time.time() - start))
