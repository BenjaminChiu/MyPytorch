"""
@Desc   : 非极大值抑制，得到都是一些局部最大值
@Time   : 2020-12-13 16:38
@Author : tank boy
@File   : 9.4.4_NMS.py
@coding : utf-8
"""
from collections import namedtuple

import matplotlib.pyplot as plt
import torch
from PIL import Image

import utils.d2lzh_pytorch as d2l

d2l.set_figsize()
img = Image.open('../img/catdog.png')
w, h = img.size

bbox_scale = torch.tensor((w, h, w, h), dtype=torch.float32)
# 原始锚框
anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
                        [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])

offset_preds = torch.tensor([0.0] * (4 * len(anchors)))
# 置信度，以物体类为行，包含每个框对当前物体类的置信度
# 以列为单位：每个框 对不同物体类的置信度
cls_probs = torch.tensor([[0., 0., 0., 0., ],  # 背景的预测概率 shape[3, 4]
                          [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
                          [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率

fig = plt.imshow(img)
# d2l.show_bboxes(fig.axes, anchors * bbox_scale, ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

# 具名元组 namedtuple('元组名称', '元组中元素名称')
Pred_BB_Info = namedtuple("Pred_BB_Info", ["index", "class_id", "confidence", "xyxy"])


def non_max_suppression(bb_info_list, nms_threshold=0.5):
    """
    非极大抑制处理预测的边界框
    Args:
        bb_info_list: Pred_BB_Info的列表, 包含预测类别、置信度等信息
        nms_threshold: 阈值，超参
    Returns:
        output: Pred_BB_Info的列表, 只保留过滤后（低于超参阈值）的边界框信息
    """
    output = []
    # 先根据置信度从高到低排序
    sorted_bb_info_list = sorted(bb_info_list, key=lambda x: x.confidence, reverse=True)

    while len(sorted_bb_info_list) != 0:
        best = sorted_bb_info_list.pop(0)  # 假设第一个元素是最合适的。将列表中第一个删除，并赋值给best
        output.append(best)

        if len(sorted_bb_info_list) == 0:
            break

        bb_xyxy = []  # 拿到 剩下未被选走锚框的所有坐标 list:总锚框数-1
        for bb in sorted_bb_info_list:
            bb_xyxy.append(bb.xyxy)
        # 比较最好框 与落榜框的IoU系数，tensor:1, shape:(len(sorted_bb_info_list))
        iou = d2l.compute_jaccard(torch.tensor([best.xyxy]), torch.tensor(bb_xyxy))[0]

        n = len(sorted_bb_info_list)
        # 找出这些落榜框中 低于 阈值的框。工程意义：大于阈值的框，因为和最佳框相似度高 被剔除。剩下的都是有特色的框
        sorted_bb_info_list = [sorted_bb_info_list[i] for i in range(n) if iou[i] <= nms_threshold]
    return output


def MultiBoxDetection_one(c_p, l_p, anc, nms_threshold=0.5):
    """
    MultiBoxDetection的辅助函数, 处理batch中的一个
    Args:
        c_p: 原始锚框置信度 shape(预测总类别数+1, 锚框个数)
        l_p: 偏移量，内容是4个坐标，压平成1维 (锚框个数*4, )
        anc: 原始锚框，内容是4个坐标，1行代表1个原始锚框(锚框个数, 4)
        nms_threshold: 非极大抑制中的阈值
    Return:
        output: (锚框个数, 6)
    """
    pred_bb_num = c_p.shape[1]  # 获得锚框个数
    anc = (anc + l_p.view(pred_bb_num, 4)).detach().cpu().numpy()  # 原始锚框 + 偏移量 = 真实框？ shape（锚框个数，4）4表示4个坐标

    confidence, class_id = torch.max(c_p, 0)  # 0表示dim=0, 按列检索 返回c_p每列最大值，返回值：1.改列最大值 2.该最大值 行号(行号代表物体类 类别)
    confidence = confidence.detach().cpu().numpy()  # 转成数组
    class_id = class_id.detach().cpu().numpy()
    # 组合 物体类别、置信度、经偏移修正后的锚框坐标，  list:原始锚框总数
    pred_bb_info = [Pred_BB_Info(
        index=i,
        class_id=class_id[i] - 1,  # 正类label从0开始
        confidence=confidence[i],
        xyxy=[*anc[i]])  # xyxy是个列表
        for i in range(pred_bb_num)]

    # 正类的index
    obj_bb_idx = [bb.index for bb in non_max_suppression(pred_bb_info, nms_threshold)]

    output = []
    # 组合这些特色框，物体类 类别、置信度、4个坐标  共6列      非特色框 类别重新设为-1（背景类）
    for bb in pred_bb_info:
        output.append([
            (bb.class_id if bb.index in obj_bb_idx else -1.0),
            bb.confidence,
            *bb.xyxy
        ])

    return torch.tensor(output)  # shape: (锚框个数, 6)


# 在多个合适的框中，选出最优的一个
def MultiBoxDetection(cls_prob, loc_pred, anchor, nms_threshold=0.5):
    """
    # 按照「9.4.1. 生成多个锚框」所讲的实现, anchor表示成归一化(xmin, ymin, xmax, ymax).
    https://zh.d2l.ai/chapter_computer-vision/anchor.html
    Args:
        cls_prob: 经过softmax后得到的各个锚框的预测概率, shape:(bn, 预测总类别数+1, 锚框个数)  bn:?  总类别数+1:加一个背景类
        loc_pred: 预测的各个锚框的偏移量, shape:(bn, 锚框个数*4) 偏移量怎么来的？
        anchor: MultiBoxPrior（该函数生成每个像素点的多组锚框，原始锚框）输出的默认锚框, shape: (1, 锚框个数, 4)
        nms_threshold: 非极大抑制中的阈值
    Returns:
        所有锚框的信息, shape: (bn, 锚框个数, 6)
        每个锚框信息由[class_id, confidence, xmin, ymin, xmax, ymax]表示
        class_id=-1 表示背景或在非极大值抑制中被移除了
    """
    # 这个断言的作用是什么？ 校验参数是否正常，不正常直接退出；正常接着运行。其中一个不满足即触发，它总能触发
    assert len(cls_prob.shape) == 3 and len(loc_pred.shape) == 2 and len(anchor.shape) == 3
    bn = cls_prob.shape[0]  # bn 这里恒=1

    batch_output = []
    for b in range(bn):
        batch_output.append(MultiBoxDetection_one(cls_prob[b], loc_pred[b], anchor[0], nms_threshold))  # 传入的参数升维，这里写死第0位（降维），相当与此时是原始参数了

    return torch.stack(batch_output)  # shape:看函数说明 stack在0位+1维度 什么意思？    浅显说法：把多个2维的张量凑成一个3维的张量；多个3维的凑成一个4维的张量…以此类推，也就是在增加新的维度进行堆叠。


# 入口
output = MultiBoxDetection(
    cls_probs.unsqueeze(dim=0), offset_preds.unsqueeze(dim=0),
    anchors.unsqueeze(dim=0), nms_threshold=0.5)

fig = d2l.plt.imshow(img)
# out转数组， shape(原始锚框个数，6)       i表示某一行，即某个锚框
for i in output[0].detach().cpu().numpy():
    if i[0] == -1:  # 锚框被设为背景类，退出
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])  # 根据类别、写置信度     前面的[int(i[0])] 是配备哪个类别的
    d2l.show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)  # 拿到坐标画特色锚框

plt.show()
