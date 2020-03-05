#模拟实现Sequential类
import torch
from torch import nn
from collections import OrderedDict

class MySequential(nn.Module):
    def __init__(self,*args):
        super(MySequential,self).__init__()
        if len(args) == 1 and isinstance(args[0],OrderedDict):#如果传入的是一个OrderedDict
            for key,module in args[0].items():
                self.add_module(key,module) #add_module 方法会将module添加进self._modules（一个OrderedDict）
        else:#传入的是一些Module
            for idx,module in enumerate(args):
                self.add_module(str(idx),module)

    def forward(self, input):
        #self._module 返回一个OrderedDict，保证会按照成员添加时的顺序遍历成
        for module in self._modules.values():
            input = module(input)
        return input

net = MySequential(
    nn.Linear(784,256),
    nn.ReLU(),
    nn.Linear(256,10)
    )

x = torch.rand(2,784)
print(net)
print(net(x))



