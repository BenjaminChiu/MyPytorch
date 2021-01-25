import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable

# from utils.visualize import make_dot

# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

x = torch.unsqueeze(torch.linspace(-0.8, 0.1, 100), dim=1)  # dim=0列 dim=1行     （100，1）
y = -5 * x + 10 + torch.rand(x.size())


# 画图
# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()


class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入 输出的维度都是1

    def forward(self, x):
        out = self.linear(x)
        return out


# if torch.cuda.is_available():
model = LinearRegression().cuda()
# else:
#     model = LinearRegression()
# print("模型结构：" + "\n")
# make_dot(model).view()

# 定义损失函数（均方误差） 和 优化函数（梯度下降）
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

# 模型训练
num_epochs = 1000
for epoch in range(num_epochs):
    # if torch.cuda.is_available():
    inputs = Variable(x).cuda()
    target = Variable(y).cuda()
    # else:
    #     inputs = Variable(x)
    #     target = Variable(y)

    # 向前传播，获得损失函数
    out = model(inputs)
    loss = criterion(out, target)

    # 向后传播，计算梯度，缩小损失
    optimizer.zero_grad()  # 注意每次迭代都需要清零，不然梯度会累加到一起，无法收敛
    loss.backward()
    optimizer.step()

    # if (epoch + 1) % 20 == 0:
    #     print('Epoch[{}/{}], loss:{:.6f}'.format(epoch + 1, num_epochs, loss.item()))

# 将模型变为测试模式，将数据放入模型，进行预测
model.eval()
# if torch.cuda.is_available():
predict = model(Variable(x).cuda())

# else:
#     predict = model(Variable(x))
#     predict = predict.data.numpy()


plt.plot(x.numpy(), y.numpy(), 'o', label='Original Data')
plt.plot(x.numpy(), predict.data.cpu().numpy(), 'r', label='Fitting Line', lw=4)
plt.show()
