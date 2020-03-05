#线性回归的从零开始实现

#%matplotlib inline
import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random
import d2lzh_pytorch as d2lzh

num_inputs = 2
num_examples = 1000

true_w = [2,-3.4]
true_b = 4.2

#随机创建两组正太分布的特征数据  1000*2
features = torch.from_numpy(np.random.normal(0,1,(num_examples,num_inputs)))
#设置(模拟)真实标签 1000
labels = true_w[0]*features[:,0] + true_w[1]*features[:,1] + true_b
#生成一个含有随机噪声的标签 1000
labels +=torch.from_numpy(np.random.normal(0,0.01,size=labels.size()))

# print(features[0],labels[0])

#显示散点图
# plt.scatter(features[:,1].numpy(),labels.numpy(),1)
# plt.show()


#训练模型
batch_size = 10
lr = 0.03
num_epochs = 3
net = d2lzh.linereg
loss = d2lzh.squared_loss
sgd = d2lzh.sgd

#初始化模型参数
w = torch.tensor(np.random.normal(0,0.01,(num_inputs,1))) #2*1
b = torch.zeros(1,dtype=torch.float64)

# 后面训练时需要对参数梯度进行迭代,所以需要设置requires_grad = True
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

#训练模型一共需要num_epochs个迭代周期,
# 在每一个迭代周期中,会使用训练数据集中所有样本一次(假设样本数据能够被批量大小整除)
for epoch in range(num_epochs):

    #x和y分别是小批量样本的特征和标签
    for x,y in d2lzh.data_iter(batch_size,features,labels):
        # print(x,y)
        # break
        #l 是 小批量x和y的总损失
        l = loss(net(x,w,b),y).sum()
        l.backward() #小批量的损失对模型参数求梯度
        sgd([w,b],lr,batch_size)#使用小批量随机梯度下降迭代模型参数,梯度更新

        #不要忘了梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()

    train_1 = loss(net(features,w,b),labels)

    print("epoch{}: loss {:.6f}".format(epoch+1,train_1.mean().item()) )
print(true_w,"\n",w )
print(true_b,'\n',b)



#
# y = torch.Tensor([3.14,0.98,1.32])
# yp = torch.Tensor([2.33,1.07,1.23])
#
# loss = loss(yp,y)
# lossmean = loss.mean()
# print(loss,',',lossmean)









