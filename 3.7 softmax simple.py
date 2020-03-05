#softmax 回归的简单实现
import torch
import sys
import numpy as np
from torch import nn
from torch.nn import init
import d2lzh_pytorch as d2l
from collections import OrderedDict

if __name__ == '__main__':

    batch_size = 256
    train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

    num_inputs = 28*28
    num_outputs = 10

    #构建网络
    net = nn.Sequential(
        OrderedDict([
            ("flatten",d2l.FlattenLayer()),
            ("linear",nn.Linear(num_inputs,num_outputs))
    ]))
    # print(net)

    #均值为0,标准差为0.01的正太分布随机初始化模型的权重参数
    init.normal_(net.linear.weight,mean=0,std=0.01)
    init.constant_(net.linear.bias,val=0)

    #定义损失函数以及梯度优化算法
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(),lr=0.1)

    num_epochs =5
    d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)

    # cud = torch.cuda.is_available()
    #
    # if cud:
    #     print("cuda")

