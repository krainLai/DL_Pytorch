#fancyMLP 花式多成感知机
#虽然前面所介绍的类会使模型构造更加简单,且不需要定义forward函数,但直接继承Module类可以极大地扩展模型构造的灵活性.
#我们将构造一个稍微复杂一点的网络fancyMLP.在这个网络中,我们通过get_constant函数创建训练中不被迭代的参数,即常数参数.
#在前向计算中,除了使用创建的常数参数外,我们还使用Tensor的函数和Python的控制流,并调用相同的层
import torch
import torch.nn as nn

class FancyMLP(nn.Module):
    def __init__(self,**kwargs):
        super(FancyMLP,self).__init__(**kwargs)
        self.rand_weight = torch.rand((20,20),requires_grad=False)#不可训练参数(常数参数)
        self.linear = nn.Linear(20,20)

    def forward(self,x):
        x = self.linear(x)
        print('x1',x)
        #使用创建的常数参数,以及nn.functional中的relu函数和mm函数
        x = nn.functional.relu(torch.mm(x,self.rand_weight.data)+1)

        #复用全连接层.等价于两个全连接层共享参数
        x = self.linear(x)
        #控制流,这里我们需要调用item函数来返回标量进行比较
        while x.norm().item() >1 :
            x /= 2
        if x.norm().item()<0.8:
            x *=10
        return x.sum()

x = torch.rand(2,20)
# print(x)
net = FancyMLP()
print(net)
print(net(x))
print("--------------------------------------------")
class NestMLP(nn.Module):
    def __init__(self,**kwargs):
        super(NestMLP,self).__init__(**kwargs)
        self.net = nn.Sequential(nn.Linear(40,30),nn.ReLU())

    def forward(self,x):
        return self.net(x)

net = nn.Sequential(NestMLP(),nn.Linear(30,20),FancyMLP())
x = torch.rand(1,40)
print(net)
print(net(x))

# print(torch.cuda.get_device_name(0))
# print(torch.cuda.device_count())

