#多层感知机的简单实现

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import sys
import d2lzh_pytorch as dl2

if __name__ == '__main__':
    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    net = nn.Sequential(
        dl2.FlattenLayer(),
        nn.Linear(num_inputs,num_hiddens),
        nn.ReLU(),
        nn.Linear(num_hiddens,num_outputs)
    )

    for params in net.parameters():
        init.normal_(params,mean=0,std=0.01)

    batch_size = 256
    train_iter ,test_iter = dl2.load_data_fashion_mnist(batch_size)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.5)

    num_epochs = 5
    dl2.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)