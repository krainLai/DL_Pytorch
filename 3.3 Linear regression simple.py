#本节代码描述了线性回归的简洁表示

import torch
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.nn import init
import torch.optim as optim


num_inputs = 2
num_examples = 1000
true_w = [2.0,-3.4]
true_b = 4.2

features = torch.tensor(np.random.normal(0,1,(num_examples,num_inputs)),dtype = torch.float)#随机创建两组正太分布的特征数据  1000*2
labels = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b#设置(模拟)真实标签 1000
labels +=torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype = torch.float)#生成一个含有随机噪声的标签 1000
print("特征:",features[0],"\n标签:",labels[0])

#使用pytorch中的data包来读取数据
batch_size = 10
dataset = Data.TensorDataset(features,labels)#将训练数据的特征和标签组合
data_iter = Data.DataLoader(dataset,batch_size,shuffle=True)#随机读取小批量数据

#定义线性网络
class LinearNet(nn.Module):
    def __init__(self,n_feature):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(n_feature,1)

    #forward 定义前向传播函数
    def forward(self, x):
        y = self.linear(x).view(-1,1)
        return y

# net =LinearNet(num_inputs)

net = nn.Sequential(nn.Linear(num_inputs,1)
                    # nn.ReLU()
                    )

#使用pytorch init 模块初始化模型参数
init.normal_(net[0].weight,mean=0,std=0.01)#将权重参数每个元素初始化为采样于均值为0,标准差为0.01的正太分布,
init.constant_(net[0].bias,val=0)#偏差会初始化为0
print("网络:",net)
#损失函数
loss = nn.MSELoss()

#梯度优化算法
opt = optim.SGD(net.parameters(),lr=0.03)

#训练模型
num_epochs = 3
for epoch in range(1,num_epochs+1):
    for X,y in data_iter:
        y_pred = net(X)
        l = loss(y_pred,y.view(10,1))
        opt.zero_grad()#梯度清零
        l.backward()
        opt.step()
    print('epoch:{},  loss:{:.6f}'.format(epoch,l.item()))

dense = net[0]
print(true_w,dense.weight)#权重
print(true_b,dense.bias)#偏移