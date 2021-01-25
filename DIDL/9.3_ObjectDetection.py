# 目标检测 和 边界框

import torch
from PIL import Image
import utils.d2lzh_pytorch as d2l
import matplotlib.pyplot as plt


d2l.set_figsize()
img = Image.open('../img/catdog.png')
# 加分号只显示图
plt.imshow(img)
# # 关闭坐标轴
plt.axis('off')
plt.show()

# bbox是bounding box的缩写
# 将边界框(左上x, 左上y, 右下x, 右下y)，4个数字表示两个坐标
dog_bbox = [50, 38, 278, 299]
cat_bbox = [289, 78, 460, 300]


def bbox_to_rect(bbox, color):  # 本函数已保存在d2lzh_pytorch中方便以后使用
    # 将边界框(左上x, 左上y, 右下x, 右下y)格式转换成matplotlib格式：
    # ((左上x, 左上y), 宽, 高)
    return d2l.plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2] - bbox[0], height=bbox[3] - bbox[1],
        fill=False, edgecolor=color, linewidth=2)


fig = d2l.plt.imshow(img)
fig.axes.add_patch(bbox_to_rect(dog_bbox, 'blue'))
fig.axes.add_patch(bbox_to_rect(cat_bbox, 'red'))
# plt.axis('off')     # 关闭坐标轴
plt.show()
