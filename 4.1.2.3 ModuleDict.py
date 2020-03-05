#ModuleDict类
import torch
import torch.nn as nn

net = nn.ModuleDict({'linear':nn.Linear(784,256),'act':nn.ReLU(),})
net['output'] = nn.Linear(256,10)#添加
print(net['linear'])#访问方式1
print(net.linear)#访问方式2
print(net.output)
print(net)#观察出与ModuleList不同的是,moduleList是按模块添加顺序排列,而moduleDict则按照字典字母顺序排列模块

