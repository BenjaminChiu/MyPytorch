"""
@Desc   : 利用皮卡丘数据集，实现目标检测
@Time   : 2020-12-16 18:26
@Author : tank boy
@File   : 9.6.2_pikachu_run.py
@coding : utf-8
"""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image

import utils.d2lzh_pytorch as d2l

DATA_DIR = 'D:/DeepLearning_Data/pikachu'

# 断言 校验是否存在pikachu/train路径
assert os.path.exists(os.path.join(DATA_DIR, "train"))


# 读取数据集 继承自Dataset
class PikachuDetDataset(torch.utils.data.Dataset):
    """皮卡丘检测数据集类"""

    def __init__(self, data_dir, part, image_size=(256, 256)):
        assert part in ["train", "val"]
        self.image_size = image_size
        self.image_dir = os.path.join(data_dir, part, "images")

        with open(os.path.join(data_dir, part, "label.json")) as f:
            self.label = json.load(f)

        # 将 PIL 图片转换成位于[0.0, 1.0]的floatTensor, shape (C x H x W)
        self.transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        image_path = str(index + 1) + ".png"

        cls = self.label[image_path]["class"]
        label = np.array([cls] + self.label[image_path]["loc"],
                         dtype="float32")[None, :]

        PIL_img = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB').resize(self.image_size)
        img = self.transform(PIL_img)

        sample = {"label": label,  # shape: (1, 5) [class, xmin, ymin, xmax, ymax] class=-1 为背景类
                  "image": img  # shape: (3, *image_size) *image_size:通道数， img_H， img_W
                  }
        return sample


# DataLoader
def load_data_pikachu(batch_size, edge_size=256, data_dir=DATA_DIR):
    """edge_size：输出图像的宽和高"""
    image_size = (edge_size, edge_size)
    train_dataset = PikachuDetDataset(data_dir, 'train', image_size)
    val_dataset = PikachuDetDataset(data_dir, 'val', image_size)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_iter = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_iter, val_iter


if __name__ == '__main__':
    batch_size = 32
    edge_size = 256
    train_iter, val_iter = load_data_pikachu(batch_size, edge_size, DATA_DIR)
    batch = iter(train_iter).next()
    print(batch["image"].shape, batch["label"].shape)

    imgs = batch["image"][0:10].permute(0, 2, 3, 1)
    bboxes = batch["label"][0:10, 0, 1:]

    axes = d2l.show_images(imgs, 2, 5).flatten()
    for ax, bb in zip(axes, bboxes):
        d2l.show_bboxes(ax, [bb * edge_size], colors=['w'])

    plt.show()
