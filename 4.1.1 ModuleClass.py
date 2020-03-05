#Module 类
import torch
from torch import nn

class MLP(nn.Module):
    #声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self,**kwargs):
        #调用MLP父类BLOCK的构造函数来进行必要的初始化
        super(MLP,self).__init__(**kwargs)
        self.hidden = nn.Linear(784,256)
        self.act = nn.ReLU()
        self.output = nn.Linear(256,10)

    #定义模型的前向计算，即如何根据输入x计算返回所需的模型输出
    def forward(self,x):
        a = self.act(self.hidden(x))
        return self.output(a)


x = torch.rand(2,784)
net = MLP()
print(net)
print(net(x))


