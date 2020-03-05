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



#下面的例子，我们来看一个卷积层的简单应用：检测图像中物体的边缘，即找到像素变化的位置
#首先，我们先构造一张6*8的图像（高和宽分别为6像素和8像素的图像），它中间4列为黑0，其余为白1

x = torch.ones(6,8)
x[:,2:6] = 0
print(x)

#然后我们构造一个高和宽分别为1和2的卷积核K
#当它的输入做互相关运算时，如果横向相邻元素相同，输出为0，否则输出为非0

K = torch.tensor([[1,-1]],dtype=torch.float)
#下面将输入x和我们设计的卷积核k做互相关运算。可以看出，我们将从白到黑和黑到白的边缘分别检测成了1和-1，其余部分全部输出为0
Y = corr2d(x,K)
print(Y)