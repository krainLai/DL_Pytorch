#modulelist

import torch
import torch.nn as nn
import numpy as np

net = nn.ModuleList([nn.Linear(784,256),nn.ReLU()])
net.append(nn.Linear(256,10))

print(net[-1])
print(net)