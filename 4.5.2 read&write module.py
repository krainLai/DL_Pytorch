#pytorch 中,module的可学习的参数(权重和偏差),模块模型包含在参数中(通过model.parameters()访问
#state_dict是一个从参数名称隐射到参数Tensor的字典对象
import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super(MLP,self).__init__()
        self.hidden = nn.Linear(3,2)
        self.act = nn.ReLU()
        self.output = nn.Linear(2,1)

    def forward(self, x):
        a = self.act(self.hidden(x))
        return self.output(a)

net = MLP()
print(net.state_dict())

#注意,只有具有可学习参数的层(卷积层,线性层等)才有state_dict中的条目.
# 优化器optim也有一个state_dict,其中包含关于优化器状态以及所使用的超参数的信息
optimizer = torch.optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
print(optimizer.state_dict())



#保存和加载模型
x = torch.randn(2,3)
y = net(x)

torch.save(net.state_dict(),'./net.py')
net2 = MLP()
net2.load_state_dict(torch.load('./net.py'))
y2 = net2(x)
print(y==y2)