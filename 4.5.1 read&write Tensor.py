#读写tensor

import torch
from torch import nn

#可以直接使用torch中的save和load函数分别存储和读取tensor

x = torch.ones(3)
print('x',x)
torch.save(x,'x.pt')


x2 = torch.load('x.pt')
print('x2',x2)

