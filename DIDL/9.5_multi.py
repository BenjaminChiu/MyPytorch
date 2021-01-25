# 多尺度目标检测，
# 本质上，我们用输入图像在某个感受野区域内的信息来预测输入图像上与该区域位置相近的锚框的类别和偏移量。

# 个人理解 相对于 在每个像素点上生成 多组(s r)（大量）的原始锚框；
# 更需要一种均匀，少量稀疏的生成方式。这就是多目标检测
import matplotlib.pyplot as plt
import torch
from PIL import Image

import utils.d2lzh_pytorch as d2l

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d2l.set_figsize()
img = Image.open('../img/catdog.png')
w, h = img.size

print("w = %d, h = %d" % (w, h))

d2l.set_figsize()


# 归一化，这也是后来 乘 图像的宽和高的原因
# 锚框anchors中xx和yy轴的坐标值分别已除以特征图fmap的宽和高，这些值域在0和1之间的值
# 坐标值 除 宽、高后 表达了锚框在特征图中的相对位置


def display_anchors(fmap_h, fmap_w, s):
    # 前两维的取值不影响输出结果(原书这里是(1, 10, fmap_w, fmap_h), 我认为错了) 约定先传img_H, img_W
    fmap = torch.zeros((1, 10, fmap_h, fmap_w), dtype=torch.float32)  # 假装这是卷积网络 输出的特征图fmap
    # 这是因为卷积核(感受野) 卷积原始图片时，一个感受野 扫描的是原始图片的一个区域，而这个区域的信息被感受野提取后，输出到feature map上的一点
    # feature map是原始图片卷积得来的，故 feature map上的每一个像素 在原始图片上都是像感受野一样 是均匀分布的

    # 正是因为是在feature map上进行锚框生成，那么feature map上的每一个像素，都会生成多组锚框
    # 而在feature map上生成的这些锚框，那么在原始图片上，这些锚框也都是像感受野一样 均匀分布的

    # 平移所有锚框使均匀分布在图片上
    offset_x, offset_y = 1.0 / fmap_w, 1.0 / fmap_h  # 偏移量 归一化
    anchors = d2l.MultiBoxPrior(fmap, sizes=s, ratios=[1, 2, 0.5]) + torch.tensor([offset_x / 2, offset_y / 2, offset_x / 2, offset_y / 2])

    bbox_scale = torch.tensor([[w, h, w, h]], dtype=torch.float32)
    d2l.show_bboxes(plt.imshow(img).axes, anchors[0] * bbox_scale)  # 乘bbox_scale的原因：之前归一化除了，这会儿乘回来，获得真实长宽（还是坐标值？）


display_anchors(fmap_h=2, fmap_w=4, s=[0.15])
# display_anchors(fmap_w=2, fmap_h=1, s=[0.4])  # 特征图的高、宽减半，用更大的锚框检测更大的目标

plt.show()
