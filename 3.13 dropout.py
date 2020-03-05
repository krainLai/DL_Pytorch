#从0实现丢弃法

import torch
import torch.nn as nn
import numpy as np
import sys
import d2lzh_pytorch as d2l

def dropout(x,drop_prob):
    assert 0 <= drop_prob <= 1

    x = x.float()


    if drop_prob ==1:
        return torch.zeros_like(x)#这种情况下把全部元素都丢弃

    keep_prob = 1 - drop_prob
    mask = (torch.randn(x.shape)<keep_prob).float()

    return mask*x/keep_prob


x = torch.arange(1,11).view(2,5)
print(x)

y=dropout(x,0.1)
print(y)