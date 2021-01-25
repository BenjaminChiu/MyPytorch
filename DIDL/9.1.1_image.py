import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import utils.d2lzh_pytorch as d2l
import matplotlib.pyplot as plt

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# sys.path.append("../..")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

d2l.set_figsize()  # 设置图尺寸
img = Image.open('../img/dog_01.jpg')

plt.imshow(img)


# scale 表示图片是否缩放
def show_images(imgs, num_rows, num_cols, scale=2):
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    for i in range(num_rows):
        for j in range(num_cols):
            axes[i][j].imshow(imgs[i * num_cols + j])
            axes[i][j].axes.get_xaxis().set_visible(False)  # 让x、y坐标轴 不出现
            axes[i][j].axes.get_yaxis().set_visible(False)
            # plt.axis('off')
    return axes


# 图像增广方法aug
def apply(img, aug, num_rows=2, num_cols=2, scale=1.5):
    Y = [aug(img) for _ in range(num_rows * num_cols)]
    show_images(Y, num_rows, num_cols, scale)


# 左右翻转
apply(img, torchvision.transforms.RandomHorizontalFlip())

# 上下翻转
# apply(img, torchvision.transforms.RandomVerticalFlip())


# 随机裁剪，参数说明 第一个参数200，将该区域的宽和高分别缩放'到'200像素
# scale 随机裁剪一块面积为原面积10% ~ 100%的区域
# ratio 该区域的宽和高之比 随机取自0.5 ~ 2
# 若无特殊说明，本节中aa和bb之间的随机数指的是从区间[a,b][a,b]中随机均匀采样所得到的连续值。
shape_aug = torchvision.transforms.RandomResizedCrop(200, scale=(0.1, 1), ratio=(0.5, 2))
# apply(img, shape_aug)


# 将图像的亮度随机变化为原图亮度的50%（1−0.5）∼ 150%（1+0.5）。
# apply(img, torchvision.transforms.ColorJitter(brightness=0.5))

# 色调
# apply(img, torchvision.transforms.ColorJitter(hue=0.5))

# 对比度
# apply(img, torchvision.transforms.ColorJitter(contrast=0.5))

# 饱和度
# apply(img, torchvision.transforms.ColorJitter(contrast=0.5))

# 随机变化图像的亮度、对比度、饱和度、色调
color_aug = torchvision.transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
# apply(img, color_aug)

# 叠加多个图像增广方法
augs = torchvision.transforms.Compose([torchvision.transforms.RandomHorizontalFlip(), color_aug, shape_aug])
# apply(img, augs)


plt.show()
