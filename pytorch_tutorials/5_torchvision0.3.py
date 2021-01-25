# -*- coding: utf-8 -*-
# @Desc    : 微调基于torchvision 0.3的目标检测模型
# @Time    : 2020-11-29 15:04
# @Author  : tank boy
# @File    : 5_torchvision0.3.py
import os
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # 下载所有图像文件，为其排序
        # 确保它们对齐
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 载入图片 ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, 'PedMasks', self.masks[idx])
        img = Image.open(img_path).convert("RGB")  # 将img转换为RGB
        # 请注意我们还没有将mask转换为RGB
        # 因为每种颜色对应一个不同的实例
        # 0是背景
        mask = Image.open(mask_path)
        # 实例被编码为不同的颜色
        # 对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
        obj_ids = np.unique(mask)
        # 第一个id是背景，所以删除它，不再从默认的0，而是1开始
        obj_ids = obj_ids[1:]

        # 将颜色编码的mask分成一组
        # 二进制格式
        masks = mask == obj_ids[:, None, None]

        # 获取每个mask的边界框坐标
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            # 只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 (等价于numpy.nonzero)
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 将所有坐标(boxes) 转换为torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 这里仅有一个类
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])  # 根据坐标获得面积

        # 假设所有实例都不是人群
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        # 新建一个元组，塞满所需结果
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.imgs)


# ===========1 微调已经预训练的模型===================
# 在COCO上加载经过预训练的预训练模型
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# replace the classifier with a new one, that has
# 将分类器替换为具有用户定义的 num_classes的新分类器
num_classes = 2  # 1 class (person) + background
# 获取分类器的输入参数的数量
in_features = model.roi_heads.box_predictor.cls_score.in_features
# 用新的头部替换预先训练好的头部
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# ========2 修改模型以添加不同的主干============
# 加载预先训练的模型进行分类和返回
# 只有功能
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN需要知道骨干网中的输出通道数量。对于mobilenet_v2，它是1280，所以我们需要在这里添加它
backbone.out_channels = 1280

# 我们让RPN在每个空间位置生成5 x 3个锚点
# 具有5种不同的大小和3种不同的宽高比。
# 我们有一个元组[元组[int]]
# 因为每个特征映射可能具有不同的大小和宽高比
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),))

# 定义一下我们将用于执行感兴趣区域裁剪的特征映射，以及重新缩放后裁剪的大小。
# 如果您的主干返回Tensor，则featmap_names应为[0]。
# 更一般地，主干应该返回OrderedDict [Tensor]
# 并且在featmap_names中，您可以选择要使用的功能映射。
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0], output_size=7, sampling_ratio=2)

# 核心目标：将这些pieces放在FasterRCNN模型中
model = FasterRCNN(backbone, num_classes=2, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)


# 使用Mask R-CNN

def get_model_instance_segmentation(num_classes):
    # 加载在COCO上预训练的预训练的实例分割模型
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 获取分类器的输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 用新的头部替换预先训练好的头部
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 现在获取掩膜分类器的输入特征数
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 并用新的掩膜预测器替换掩膜预测器
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

# 将使模型准备好在您的自定义数据集上进行训练和评估。


import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)








