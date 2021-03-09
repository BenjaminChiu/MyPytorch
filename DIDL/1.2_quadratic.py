# 非线性回归
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import matplotlib.pylab as plt
from torchsummary import summary

# import os
#
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n = 2
x = torch.unsqueeze(torch.linspace(-2, 2, 100), dim=1)  # dim=0列 dim=1行   （100,1）
y = x.pow(3) + 0.2 * torch.rand(x.size())  # pow平方

x, y = Variable(x).to(DEVICE), Variable(y).to(DEVICE)


# plt.scatter(x.data.numpy(),y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义网络有哪些层
        self.hidden_1 = torch.nn.Linear(1, 100)
        self.hidden_2 = torch.nn.Linear(100, 80)
        self.predict = torch.nn.Linear(80, 1)

    # 定义层的具体形式
    def forward(self, x):
        out = self.hidden_1(x)
        out = F.relu(out)

        out = self.hidden_2(out)
        out = F.relu(out)

        y = self.predict(out)
        return y


net = Net().to(DEVICE)
# print(net)

# print("打印网络结构：")
# summary(net, (1, 10, 1))


# 可视化,实时打印
# plt.ioff()
# plt.show()

loss_func = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

for t in range(5000):
    prediction = net(x)

    loss = loss_func(prediction, y)

    # 优化步骤
    optimizer.zero_grad()  # 每次循环，梯度都先设为0
    loss.backward()
    optimizer.step()

    if t % 20 == 0:
        plt.cla()
        plt.scatter(x.data.cpu().numpy(), y.data.cpu().numpy())
        plt.plot(x.data.cpu().numpy(), prediction.data.cpu().numpy(), 'r-', lw=3)
        # plt.text(0.5, 0, 'loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        print('loss=%.5f' % loss.data.cpu().numpy())
        plt.pause(0.1)

plt.ioff()
plt.show()
