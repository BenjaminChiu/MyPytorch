"""
@Desc   : 基于锚框精度的 交并比
@Time   : 2020-12-12 17:50
@Author : tank boy
@File   : 9.4.2_IoU.py
@coding : utf-8
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

import utils.d2lzh_pytorch as d2l

d2l.set_figsize()
img = Image.open('../img/catdog.png')
w, h = img.size


def compute_intersection(set_1, set_2):
    """
    计算anchor之间的交集
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax)
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        intersection of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # PyTorch auto-broadcasts singleton dimensions
    # 通过比较左上角、右下角坐标，获得交集左上角（两者中较大）、右下角坐标（两者中较小）
    upper_bounds = torch.max(set_1[:, :2].unsqueeze(1), set_2[:, :2].unsqueeze(0))  # 比较左上角坐标 # set1: [N,2]->[N,1,2]->[N,M,2] ; set2: [M,2]->[1,M,2]->[N,M,2]
    lower_bounds = torch.min(set_1[:, 2:].unsqueeze(1), set_2[:, 2:].unsqueeze(0))  # 比较右下角坐标
    intersection_dims = torch.clamp(upper_bounds - lower_bounds, min=0)  # (n1, n2, 2) 限制范围下限min=0
    return intersection_dims[:, :, 0] * intersection_dims[:, :, 1]  # (n1, n2) 相交锚框的长 * 宽


# 在原始锚框 被 分配真实锚框函数 中被调用
def compute_jaccard(set_1, set_2):
    """
    计算anchor之间的Jaccard系数(IoU 交集与并集之比)
    Args:
        set_1: a tensor of dimensions (n1, 4), anchor表示成(xmin, ymin, xmax, ymax) 常将set1设为原始锚框
        set_2: a tensor of dimensions (n2, 4), anchor表示成(xmin, ymin, xmax, ymax)
    Returns:
        Jaccard Overlap of each of the boxes in set 1 with respect to each of the boxes in set 2, shape: (n1, n2)
    """
    # Find intersections
    intersection = compute_intersection(set_1, set_2)  # (n1, n2)

    # Find areas of each box in both sets
    # (x_max - x_min)框1长 * (y_max - y_min)框1高 得到面积
    areas_set_1 = (set_1[:, 2] - set_1[:, 0]) * (set_1[:, 3] - set_1[:, 1])  # (n1)
    areas_set_2 = (set_2[:, 2] - set_2[:, 0]) * (set_2[:, 3] - set_2[:, 1])  # (n2)

    # Find the union
    # PyTorch auto-broadcasts singleton dimensions
    union = areas_set_1.unsqueeze(1) + areas_set_2.unsqueeze(0) - intersection  # (n1, n2) 两个框面积相加 - 重复的一个交集 = 两框并集

    return intersection / union  # (n1, n2) 每一行，表示当前原始锚框 相对于 各个物体类真实锚框的交并比（设set1为原始锚框）；某一列 代表当前某个物体类的 关于所有原始锚框的交并比系数


bbox_scale = torch.tensor((w, h, w, h), dtype=torch.float32)
# 人工标注 真实的框
ground_truth = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                             [1, 0.55, 0.2, 0.9, 0.88]])  # shape：[2,5]
# 手动产生 锚框
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                        [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                        [0.57, 0.3, 0.92, 0.9]])

fig = plt.imshow(img)
d2l.show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
d2l.show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
plt.show()


# 以下函数已保存在d2lzh_pytorch包中方便以后使用
# 作用：根据原始anchor、真实bb的交并比表；为每个原始anchor 分配的 真实bb对应的索引
def assign_anchor(bb, anchor, jaccard_threshold=0.5):
    """
    # 按照「9.4.1. 生成多个锚框」图9.3所讲为每个anchor分配真实的bb, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        bb: 真实边界框(bounding box), shape:（nb, 4）
        anchor: 待分配的anchor, shape:（na, 4）
        jaccard_threshold: 预先设定的阈值
    Returns:
        assigned_idx: shape: (na, ), 每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
    """
    na = anchor.shape[0]  # 原始锚框个数
    nb = bb.shape[0]  # 真实锚框个数
    jaccard = compute_jaccard(anchor, bb).detach().cpu().numpy()  # 交并比表 shape: (na, nb)
    assigned_idx = np.ones(na) * -1  # 将每个原始锚框 默认设为背景类，初始全为-1（当前变量为：最合适的目标锚框容器，也是返回值）

    # 先为每个bb分配一个anchor(不要求满足jaccard_threshold)
    jaccard_cp = jaccard.copy()  # 复制一个交并比的系数

    # j列 ，在交并比表(jaccard)中 代表真实物体类的类别
    # 最终这个迭代器 会选出所有物体类的 最大的原始锚框，存放在assigned_idx中
    for j in range(nb):
        i = np.argmax(jaccard_cp[:, j])  # 工程意义：将当前物体类j的最大交并比的原始锚框 找出来；  将原始锚框交并比 中的 j列的最大元素的位置(索引) 赋值给 i
        assigned_idx[i] = j  # 利用上步中的索引（位置）i，找到在目标锚框容器(assigned_idx)中对应的原始锚框，赋值 真实物体类 类别编号
        jaccard_cp[i, :] = float("-inf")  # 因为原始锚框i，已经被录取为类别j的最合适锚框。剩下类别的交并比选拔不再参与；  将i行置空，赋值为负无穷, 相当于去掉这一行

    # 处理还未被分配的anchor, 要求满足jaccard_threshold。思考为什么还要选拔，一个物体类只要1个不就好了？
    for i in range(na):
        if assigned_idx[i] == -1:  # 未被选拔的原始锚框i
            j = np.argmax(jaccard[i, :])  # 将该原始锚框最有天赋的物体类别找出，得到物体类j
            if jaccard[i, j] >= jaccard_threshold:  # 该落榜原始锚框，最有天赋的交并比系数 >= 预设系数
                assigned_idx[i] = j  # 同样录取该原始锚框

    return torch.tensor(assigned_idx, dtype=torch.long)  # 转成tensor


def xy_to_cxcy(xy):
    """
    将(x_min, y_min, x_max, y_max)形式的anchor转换成(center_x, center_y, w, h)形式的.
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection/blob/master/utils.py
    Args:
        xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    Returns:
        bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    # (x_max + xmin) / 2
    return torch.cat([(xy[:, 2:] + xy[:, :2]) / 2,  # c_x, c_y
                      xy[:, 2:] - xy[:, :2]], 1)  # w, h   torch.cat((a, b), dim=1) 左右拼接(横着拼)；dim=0 上下拼


def MultiBoxTarget_one(anc, lab, eps=1e-6):
    """
    MultiBoxTarget函数的辅助函数, 处理batch中的一个
    Args:
        anc: shape of (原始锚框总数, 4)
        lab: shape of (真实锚框数, 5), 5代表[类别标签, 四个坐标值]
        eps: 一个极小值, 防止log0
    Returns:
        offset: (锚框总数*4, ) 原始锚框偏移量
        bbox_mask: (锚框总数, 4), 0代表背景, 1代表非背景
        cls_labels: (原始锚框总数), 0代表背景。原始锚框所属类别
    """
    an = anc.shape[0]  # 获得原始锚框个数
    assigned_idx = assign_anchor(lab[:, 1:], anc)  # lab(真实锚框数, 类别+4个坐标)   anc（原始锚框总数，4个坐标）
    bbox_mask = ((assigned_idx >= 0).float().unsqueeze(-1)).repeat(1, 4)  # (锚框总数, 4) 这4列中 后三列完全重复第一列。这么做的目的？

    cls_labels = torch.zeros(an, dtype=torch.long)  # 0表示背景 tensor([0, 0, 0, 0, 0])
    assigned_bb = torch.zeros((an, 4), dtype=torch.float32)  # 所有anchor对应的bb坐标 5行 4列个0
    for i in range(an):
        bb_idx = assigned_idx[i]  # 在原始锚框分配表中拿到，真实锚框 物体类 类别
        if bb_idx >= 0:  # 即非背景
            cls_labels[i] = lab[bb_idx, 0].long().item() + 1  # 注意要加一  [1, 2, 0, 0, 0]，why?+1：避免出现负数，且0为背景类，干脆物体类全部+1
            assigned_bb[i, :] = lab[bb_idx, 1:]  # 获得真实框的4个坐标

    center_anc = xy_to_cxcy(anc)  # (center_x, center_y, w, h)
    center_assigned_bb = xy_to_cxcy(assigned_bb)

    #                   10 * (真实c_x,c_y  -  原始c_x,c_y) / w,h
    offset_xy = 10.0 * (center_assigned_bb[:, :2] - center_anc[:, :2]) / center_anc[:, 2:]
    # 真实 w,h / 原始w,h
    offset_wh = 5.0 * torch.log(eps + center_assigned_bb[:, 2:] / center_anc[:, 2:])    # torch.log: e的x次方 = 输入的参数(tensor)，返回x -> tensor
    offset = torch.cat([offset_xy, offset_wh], dim=1) * bbox_mask  # (锚框总数, 4)

    return offset.view(-1), bbox_mask.view(-1), cls_labels      # view(-1) 不管行数多少，只有1列


# 该函数为锚框 标注类别、偏移量
# 该函数将背景类别设为0 （未被分配的锚框），并令从零开始的目标类别的整数索引自加1（1为狗，2为猫）
def MultiBoxTarget(anchor, label):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        anchor: torch tensor, 输入的锚框, 一般是通过MultiBoxPrior生成, shape:（1，锚框总数，4）
        label: 真实标签, shape为(bn, 每张图片最多的真实锚框数, 5) bn 插入的在0位加1维度  5代表[类别标签, 四个坐标值]
               第二维中，如果给定图片没有这么多锚框, 可以先用-1填充空白, 最后一维中的元素为[类别标签, 四个坐标值]
    Returns:
        列表, [bbox_offset, bbox_mask, cls_labels]，下面的bn 其实是插入生成的0位的1个维度
        bbox_offset: 每个锚框的标注4个偏移量，形状为(bn，锚框总数*4)。负类锚框的偏移量标注为0
        bbox_mask: 形状同bbox_offset,形状(批量大小, 锚框个数的四倍) 每个锚框的掩码变量中的元素, 一一对应上面每个锚框的4个偏移量, 负类锚框(背景)对应的掩码均为0（滤掉负类的偏移量）, 正类锚框的掩码均为1
        cls_labels: 每个锚框的标注类别, 其中0表示为背景, 形状为(bn，锚框总数)， [1, 2, 0, 0, 0]
    """
    assert len(anchor.shape) == 3 and len(label.shape) == 3  # 传入参数的合法性。 用于判断一个表达式，在表达式条件为 false 的时候触发异常。当前情况下：出错，什么也不做，直接弹出
    bn = label.shape[0]

    batch_offset = []
    batch_mask = []
    batch_cls_labels = []
    for b in range(bn):
        offset, bbox_mask, cls_labels = MultiBoxTarget_one(anchor[0, :, :], label[b, :, :])

        batch_offset.append(offset)
        batch_mask.append(bbox_mask)
        batch_cls_labels.append(cls_labels)

    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    cls_labels = torch.stack(batch_cls_labels)

    return [bbox_offset, bbox_mask, cls_labels]


# 入口
labels = MultiBoxTarget(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
print(labels)
