import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

num_inputs = 2
num_examples = 500
features = torch.randn(num_examples, num_inputs, dtype=torch.float32)

true_w = [2, -3.4]
true_b = 4.2

# print(type(features[0]))
# print(type(features[:, 0]))
# temp = list(range(0, 10))
# print(list(range(0, 10)))
# print("数量是："+str(torch.numel(features)))

# 样本是 1000行，2列 的type
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b  # tensor张量 features[:,0] 表示第1列的所有内容；features[:,1] 表示第2列的所有内容
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()), dtype=torch.float32)


# print(features[0], labels[0])


def use_svg_display():
    display.set_matplotlib_formats('svg')  # 用矢量图展示


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize  # 设置图的尺寸


set_figsize()
plt.scatter(features[:, 1].numpy(), labels.numpy(), 1)
# plt.show()


# 开始读取样本

def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))

    random.shuffle(indices)  # 样本的读取顺序是随机的

    for i in range(0, num_examples, batch_size):
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

# 初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)

# 深入理解为什么要 参数梯度
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)


# 定义模型
def linreg(X, w, b):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    return torch.mm(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    # 注意这里返回的是向量, 另外, pytorch里的MSELoss并没有除以 2
    return (y_hat - y.view(y_hat.size())) ** 2 / 2


# 定义优化算法
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


# if __name__ == '__main__':
lr = 0.03
num_epochs = 10
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        prediction = net(X, w, b)
        l = loss(prediction, y).sum()  # l是有关小批量X和y的损失
        temp = l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数


        # 不要忘了梯度清零，为什么要在这清零梯度？
        w.grad.data.zero_()
        b.grad.data.zero_()

    # 每个批次打一下
    big_prediction = net(features, w, b)
    # plt.plot(big_prediction.data.numpy(), 'r-', lw=0.5)
    plt.plot(torch.mm(X, w), big_prediction.data.numpy(), 'r-', lw=0.5)
    # plt.plot(labels, 'y-', lw=1)
    # plt.pause(0.1)


    train_l = loss(big_prediction, labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))



plt.show()
print(true_w, '\n', w)
print(true_b, '\n', b)
