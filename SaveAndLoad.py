'''模型的保存、加载'''

import torch
from torch import nn

x = torch.randn(9).cuda()
torch.save(x, 'x.pt')

x2 = torch.load('x.pt')
print(x2)