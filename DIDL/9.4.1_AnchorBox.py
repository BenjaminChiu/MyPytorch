"""
@Desc   : 自动化，批量生成锚框
@Time   : 2020-12-12 17:50
@Author : tank boy
@coding : utf-8
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import utils.d2lzh_pytorch as d2l

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d2l.set_figsize()
img = Image.open('../img/catdog.png')
w, h = img.size

print("w = %d, h = %d" % (w, h))


# 生成锚框，指定输入、一组大小(0~1) 数组length：n个   ； 一组宽高比 数组length m个；该函数将返回输入的所有锚框
# 相同像素为中心的锚框的数量为n+m−1，当前状态为产生5个锚框
def MultiBoxPrior(feature_map, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5]):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成(xmin, ymin, xmax, ymax).
    Args:
        feature_map: torch tensor, Shape: [N, C, H, W].
        sizes: List of sizes (0~1) of generated MultiBoxPriores.
        ratios: List of aspect ratios (non-negative) of generated MultiBoxPriores.
    Returns:
        anchors of shape (1, num_anchors, 4). 由于batch里每个都一样, 所以第一维为1
    """
    pairs = []  # pairs of (size, sqrt(ration))
    for r in ratios:
        pairs.append([sizes[0], math.sqrt(r)])  # 获取一组大小比为size[0]的，带很多长宽比的数组 即教材中的m值  ;  math.sqrt() 对参数开方，return -> double
    for s in sizes[1:]:
        pairs.append([s, math.sqrt(ratios[0])])  # n-1 个 长度，因为是从第1个开始取的，而非第0个；避免与上式子(s1,r1)重复

    pairs = np.array(pairs)  # 将pairs(list:5) 转为 pairs(nd array: 5行 2列)

    ss1 = pairs[:, 0] * pairs[:, 1]  # 字面意思第一列*第二列  size * sqrt(ration)
    ss2 = pairs[:, 0] / pairs[:, 1]  # 字面意思第一列 除 第二列 size / sqrt(ration)
    base_anchors = np.stack([-ss1, -ss2, ss1, ss2], axis=1) / 2  # ndarray(5, 4) 为什么除2？以-ss1为例，根据锚框边长WS√r，它的一半恰好是坐标的x值
    # 返回了5组 锚框

    # 目的：输入数据（像素点）的归一化
    # 根据传入图像获得长和宽
    h, w = feature_map.shape[-2:]  # -2 表示什么意思？ 表示符号表示倒数 取倒数2个 即-2 -1 0，0不取 并且顺序是数组的正序=
    shifts_x = np.arange(0, w) / w  # 除自己有什么意义？归一化，目前结果得到的都是小于1的数。会在画图时乘w 回来。 arange 在给定间隔内返回均匀间隔的值
    shifts_y = np.arange(0, h) / h  # (nd array(361))
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)  # 将两个数组中数字 进行组合，得到这张图片上的像素点矩阵
    # 返回结果: [array([ [1,2,3] [1,2,3] ]), array([ [7,7,7] [8,8,8] ])]

    shift_x = shift_x.reshape(-1)  # 将二维数组 压缩为1行，不分行列，改成一串[1 2 3 4 5]
    shift_y = shift_y.reshape(-1)
    shifts = np.stack((shift_x, shift_y, shift_x, shift_y), axis=1)  # ndarray 168587, 4

    # shifts.reshape: ndarray 168587, 1, 4
    # base_anchors.reshape: ndarray 1, 5, 4
    # ？？？ 相加的目的：可能是将5组大小、宽高比 组合到 t1中   -1 应该表示一个模糊数，表示不管 如不管多少个的1行4列个数组
    anchors = shifts.reshape((-1, 1, 4)) + base_anchors.reshape((1, -1, 4))  # ndarray(168587, 5, 4) 像素点，5组比，4个坐标？

    # tensor(1, 842935, 4)
    return torch.tensor(anchors, dtype=torch.float32).view(1, -1, 4)


X = torch.Tensor(1, 3, h, w)  # 构造输入数据
Y = MultiBoxPrior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
Y.shape  # torch.Size([1, 842935, 4])   锚框变量y的形状为（1，锚框个数，4）

boxes = Y.reshape((h, w, 5, 4))  # 5 相同像素为中心的锚框个数
t4 = boxes[250, 250, 0, :]  # 图片中 250 250这个像素点，组合，4列坐标  * torch.tensor([w, h, w, h], dtype=torch.float32)

print(t4)


# 函数_make_list目的：将obj转为list或tuple
def _make_list(obj, default_values=None):
    if obj is None:
        obj = default_values
    elif not isinstance(obj, (list, tuple)):
        obj = [obj]
    return obj


# 描绘图像中以某个像素为中心的所有锚框
# 本函数已保存在d2lzh_pytorch包中方便以后使用
def show_bboxes(axes, bboxes, labels=None, colors=None):
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = d2l.bbox_to_rect(bbox.detach().cpu().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=6, color=text_color,
                      bbox=dict(facecolor=color, lw=0))


# bbox_scale 锚框的原始坐标值
d2l.set_figsize()
fig = plt.imshow(img)
bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
            ['s=0.75, r=1', 's=0.75, r=2', 's=0.55, r=0.5', 's=0.5, r=1', 's=0.25, r=1'])

plt.show()
