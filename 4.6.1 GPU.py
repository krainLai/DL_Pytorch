import torch
from torch import nn

#查看GPU是否可用
print(torch.cuda.is_available())

#查看GPU数量
print(torch.cuda.device_count())

#查看当前GPU索引号
print(torch.cuda.current_device())

#根据索引号查看GPU名字
print(torch.cuda.get_device_name(0))

