#使用L2 范数正则化(权重衰减法)
import torch
import torch.nn as nn
import numpy as np
import sys
import d2lzh_pytorch as d2l

#为了容易观察过拟合问题,我们考虑使用高维线性回归设计问题,设维度为200,同时将样本数量降低,如20
n_train ,n_test,num_inputs = 20,100,200

true_w ,true_b = torch.ones(num_inputs,1)*0.01,0.05
features = torch.randn((n_train+n_test,num_inputs))
labels = torch.matmul(features,true_w) + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),dtype=torch.float)

train_features,test_features = features[:n_train,:],features[n_train:,:]
train_labels ,test_labels = labels[:n_train],labels[n_train:]

#初始化模型参数
def init_params():
    w = torch.randn((num_inputs,1),requires_grad=True)
    b = torch.zeros(1,requires_grad=True)
    return [w,b]

#定义L2范数惩罚项
#下面定义L2范数惩罚项,这里只惩罚模型的权重参数
def l2_penalty(w):
    return (w**2).sum()/2

#定义训练和测试
batch_size,num_epochs,lr = 1,100,0.003
net,loss = d2l.linereg,d2l.squared_loss

dataset = torch.utils.data.TensorDataset(train_features,train_labels)
train_iter = torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)

def fit_and_plot(lambd):
    w,b = init_params()
    train_ls,test_ls = [],[]
    for _ in range(num_epochs):
        for x,y in train_iter:

            y_pred = net(x,w,b)
            # y_pred = net(x)
            # print('w:',w.shape,'x:',x.shape,'y:',y.shape,'pre:',y_pred.shape)
            l = loss(y_pred,y) + lambd * l2_penalty(w)
            l = l.sum()

            if w.grad is not None:
                w.grad.data.zero_()
                b.grad.data.zero_()

            l.backward()
            d2l.sgd([w,b],lr,batch_size)
        train_ls.append(loss(net(train_features,w,b),train_labels).mean().item())
        test_ls.append(loss(net(test_features,w,b),test_labels).mean().item())
    d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','loss',
                 range(1,num_epochs+1),test_ls,['train','test']
                 )
    print('L2 norm of w',w.norm().item())


fit_and_plot(42)