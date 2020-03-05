import torch
import torchvision
import torchvision.transforms as transfroms
import matplotlib.pyplot as plt
import time
import sys
# sys.path.append("..")
import d2lzh_pytorch as d2l
import torch.utils.data as Data
import numpy as np
import torch.optim as optimizer


#y_hat 样本在类别中的预测概率 ,y 样本的标签类别,gather函数预测标签的概率
def cross_entropy(y_hat,y):
    return - torch.log(y_hat.gather(1,y.view(-1,1)))

#准确率函数
#给定一个类别的预测概率分布y_hat,我们把预测概率最大的类别作为输出类别.
#如果它与真实类别y一致,说明这次的预测是正确的.分布准确率即正确预测数量与总预测数量之比
#其中,y_hat.argmax(din=1)返回矩阵y_hat每行中最大的元素的索引,且返回结果与变量y形状相同
def accuracy(y_hat,y):
    return(y_hat.argmax(din=1) == y).float().mean().item()

def net(x):
    return d2l.softmax(torch.mm(x.view((-1, num_inputs)), w) + b)

if __name__ == '__main__': #此处不加,DataLoader中的 num_workers>0 会报错,因为运行缺少主程序,不能开启多线程
    batch_size = 256
    train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

    #已知样本输入是28*28的图像,我们使用向量来表示每个样本,则模型输入向量的长度是28*28=784,由于图像一个有10个类别,单层神经网络的输出层输出个数为10
    num_inputs = 28*28
    num_outputs = 10

    #定义随机均值为0,方差为0.01的正太分布参数,以及偏置b=0
    w = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_outputs)),dtype=torch.float)#不加dtype=torch.float会在后向传播中参数跟新中报错
    b = torch.zeros(num_outputs,dtype=torch.float)

    #模型参数梯度设置
    w.requires_grad_(True)
    b.requires_grad_(True)

    num_epochs,lr = 5,0.05

    d2l.train_ch3(net,train_iter,test_iter,cross_entropy,num_epochs,batch_size,[w,b],lr=lr)

    x,y = iter(test_iter).next()

    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(x).argmax(dim=1).numpy())
    titles = [true +"\n" +pred for true ,pred in zip (true_labels,pred_labels) ]

    d2l.show_fashion_mnist(x[0:9],titles[0:9])





