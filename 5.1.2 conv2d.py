#自定义二维卷积层
#二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。
#卷积层的模型参数包括了卷积核和标量偏差
#在训练模型的时候，通常我们先对卷积核随机初始化，然后不断迭代卷积核和偏差
import torch
import torch.nn as nn

def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in  range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum() #点乘再求和
    return Y


class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D,self).__init__()
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return corr2d(x,self.weight) + self.bias



