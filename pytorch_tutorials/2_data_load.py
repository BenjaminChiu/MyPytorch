# 数据加载与处理

from __future__ import print_function, division
import os
import torch
import pandas as pd              #用于更容易地进行csv解析
from skimage import io, transform    #用于图像的IO和变换
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
# from torch.utils.data.dataset import T_co
from torchvision import transforms, utils

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode 交互模式？

# ------------------------------封装前----------------------------------------

# data_path = 'D:/Develop_Data/DeepLearning_Data/faces/face_landmarks.csv'
#
# landmarks_frame = pd.read_csv(data_path)
#
# n = 60
# img_name = landmarks_frame.iloc[n, 0]       # 第一列 拿名字
# landmarks = landmarks_frame.iloc[n, 1:].values
# landmarks = landmarks.astype('float').reshape(-1, 2)      # 不管多少行，2列（img name，特征值）
#
# print('Image name: {}'.format(img_name))
# print('Landmarks shape: {}'.format(landmarks.shape))
# print('First 4 Landmarks: {}'.format(landmarks[:4]))


def show_landmarks(image, landmarks):
    # 显示带地标的图片
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker='.', c='r')      # landmarks[:, 0] 第一列
    # plt.pause(0.001) # pause a bit so that plots are updated  让plt休眠 后果为 打印多张图片

# plt.figure()
# show_landmarks(io.imread(os.path.join('D:/Develop_Data/DeepLearning_Data/faces', img_name)), landmarks)
# plt.show()


# -----------------------------封装后-------------------------------------

# 继承Dataset，dataset是数据集的抽象类

class FaceLandmarksDataset(Dataset):
    """面部标记数据集"""
    def __init__(self, csv_file, root_dir, transform=None):
        """
        :param csv_file: string 带注释的csv文件的路径
        :param root_dir: string 包含所有图像的目录
        :param transform: （callable， optional）：一个样本的可用的可选变换
        """
        self.landmarks_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    # idx 图片角标
    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
        image = io.imread(img_name)
        landmarks = self.landmarks_frame.iloc[idx, 1:]  # 没要第一列
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        return sample

# 数据可视化

# 获取数据集
face_dataset = FaceLandmarksDataset(csv_file='D:/Develop_Data/DeepLearning_Data/faces/face_landmarks.csv',
                     root_dir='D:/Develop_Data/DeepLearning_Data/faces')

fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i, sample['image'].shape, sample['landmarks'].shape)

    ax = plt.subplot(1, 5, i + 1)
    plt.tight_layout()
    ax.set_title('Sample #{}'.format(i))
    # ax.axis('off')
    show_landmarks(**sample)

    if i == 4:
        plt.show()
        print("执行")
        break


# 数据变换，因为图片大多数时候并不是标准统一的尺寸，我们需要将他们统一尺寸

# call函数 使类变为一个函数（让类new出来的对象，可以像函数一样被调用）
class Rescale(object):
    """将样本中的图像 重新缩放到指定大小

    Args:
        output_size（tuple或int）：所需的输出大小。 如果是元组，则输出为
         与output_size匹配。 如果是int，则匹配较小的图像边缘到output_size保持宽高比相同。
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]  # 只取前3个
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)       # 强转一下 数据类型
        img = transform.resize(image, (new_h, new_w))

        # h and w are 交换 for landmarks because for images,
        # x轴和y轴分别是轴1和0
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}

class RandomCrop(object):
    """随机裁剪图像
    :arg
        output_size（tuple或int）：所需的输出大小。 如果是int，方形裁剪是。
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h, left: left + new_w]
        landmarks = landmarks - [left, top]
        return {'image': image, 'landmarks': landmarks}



class ToTensor(object):
    """将ndarrays转换为Tensor"""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # 交换颜色轴因为
        # numpy包的图片是: H * W * C
        # torch包的图片是: C * H * W
        image = image.transpose((2, 0, 1))  # 转置矩阵
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}

# 使用重新缩放，随机裁剪
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256), RandomCrop(224)]) # 将两个效果叠加

# 在样本上应用上述的每个变换
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i+1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()