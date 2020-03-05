#通过数据学习核数组

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


X = torch.ones(6,8)
X[:,2:6] = 0
print(X)
K = torch.tensor([[1,-1]],dtype=torch.float)
Y = corr2d(X,K)
print(Y)

#我们来看一个例子，它使用物体边缘检测中的输入数据x和输出数据y来学习我们构造的核数组k
#我们首先构造一个卷积层，其卷积核将被初始化成随机数组
#接下来在每一次迭代中，我们使用平方误差来比较y和卷积层的输出，然后计算梯度来更新权重

#构造一个核数组形状为（1,2）的二维卷积层
conv2d = Conv2D(kernel_size=(1,2))

step = 40
lr = 0.01

for i in range(step):
    Y_hat = conv2d(X)
    l = ((Y_hat - Y)**2).sum()
    l.backward()
    #梯度下降
    conv2d.weight.data -= lr*conv2d.weight.grad
    conv2d.bias.data -=  lr*conv2d.bias.grad

    #梯度清零
    conv2d.weight.grad.fill_(0)
    conv2d.bias.grad.fill_(0)
    if (i+1)%5 == 0:
        print('step:',(i+1),',loss:',l.item())

print('weight:',conv2d.weight.data)
print('bias:',conv2d.bias.data)
