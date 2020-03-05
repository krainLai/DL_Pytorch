import random
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import sys
import time
import torch.utils.data as Data
import torchvision.transforms as transfroms
from torch import nn

#此函数每次返回batch_size(批量大小)个随机样本的特征和标签
def data_iter(batch_size: object, feature: object, labels: object) -> object:
    num_examples = len(feature)
    indices = list(range(num_examples))
    random.shuffle(indices)#样本的读取顺序是随机的
    for i in range(0,num_examples,batch_size):
        j = torch.LongTensor(indices[i:min(i+batch_size,num_examples)])
        yield feature.index_select(0,j),labels.index_select(0,j)


#此函数是线性回归矢量计算表达式的实现,使用mm函数做矩阵乘法
def linereg(x,w,b):
    # print('b',b.shape)
    return torch.mm(x,w) + b

#此函数为损失函数
def squared_loss(y_hat,y):
    #注意,这里返回的是向量,另外,pytorch里的MSEloss并没有除以2
    # print("(y_hat - y.view(-1,1)) ** 2 / 2")
    # return (y_hat - y.view(-1,1)) ** 2 / 2
    # print("(y_hat.view(-1) - y) ** 2 / 2")
    # return (y_hat.view(-1) - y) ** 2 / 2

    # print("(y_hat - y.view(-1, 1)) ** 2 / 2")
    # return (y_hat - y.view(-1, 1)) ** 2 / 2


    return (y_hat - y.view(y_hat.size()))**2/2

#自定义sgd梯度优化算法
#特点: 这里的自动求梯度模块计算得来的梯度是一个批量样本的和.我们将它除以批量大小来得到平均值
def sgd(params,lr,batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size


#fashionmninst集 将数值标签转为相应的文本标签
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt','trouser','pullover','dress','coat','sandal','shirt','sneaker','bag','ankle','boot']
    return [text_labels[int(i)] for i in labels]

#画多张图像和对应标签的函数
def show_fashion_mnist(images,labels):

    #_代表忽略不用的变量
    _,figs = plt.subplots(1,len(images),figsize=(12,2))
    # _, figs = plt.subplots(1, len(images) )
    for f,img,lbl in zip(figs,images,labels):
        f.imshow(img.view((28,28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

def load_data_fashion_mnist(batch_size):
    mnist_train = torchvision.datasets.FashionMNIST(root="./Dataset/FashionMnist", train=True, download=True,
                                                    transform=transfroms.ToTensor())
    mnist_test = torchvision.datasets.FashionMNIST(root="./Dataset/FashionMnist", train=False, download=True,
                                                   transform=transfroms.ToTensor())

    # 读取小批量数据,使用多进程来加速数据读取
    if sys.platform.startswith('Win'):
        num_workers = 0  # 0表示不需要额外的进程来加速读取数据
    else:
        num_workers = 2  # 四个进程加速 超过3会出现页面文件太小,无法操作

    train_iter = Data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_iter = Data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_iter,test_iter

def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(dim=1,keepdim = True)
    return x_exp/partition #这里使用了广播机制

#评价模型net在数据集上的准确率计算
def evaluate_accuracy(data_iter,net):
    acc_sum,n = 0.0,0
    for x,y in data_iter:
        acc_sum +=(net(x).argmax(dim=1)==y).float().sum().item()
        n += y.shape[0]
    return acc_sum/n

def train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params=None,lr=None,optimizer =None):
    for epoch in range(num_epochs):
        train_l_sum,train_acc_sum,n = 0.0,0.0,0
        for x,y in train_iter:
            y_hat = net(x)
            l = loss(y_hat,y).sum()

            #梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()
            #梯度优化
            if optimizer is None:
                sgd(params,lr,batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
            n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter,net)
        print('epoch:{} ,loss:{:.4f}, train_acc:{:.3f}, test_acc:{:.3f}'.format(epoch+1,train_l_sum/n , train_acc_sum/n,test_acc))


class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        # print(x.shape,x.shape[0])
        return x.view(x.shape[0],-1)

#绘制xy函数图像
def xyplot(x,y,name):
    #detach 截断反向传播的梯度流。
    plt.plot(x.detach().numpy(),y.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name+'(x)')
    plt.show()

#显示激活函数的曲线图
def showActivationFuntion(name):
    x = torch.arange(-11.0,11.0,0.1,requires_grad=True)
    if(name =='relu' ):
        # relu 激活函数
        y = x.relu()
        # d2l.xyplot(x,y,"relu")
        # y.sum().backward()#此函数主要作用是用来递归梯度信息,y.sum()无其他意义
        # d2l.xyplot(x,x.grad,"grad of relu")
    elif(name == 'sigmoid'):
        y = x.sigmoid()
        # d2l.xyplot(x, y, "sigmoid")
    elif(name == 'tanh'):
        y = x.tanh()
    else:
        print("funtion not exit!")
        return

    z = y.sum().backward()

    plt.subplot(1,2,1)
    plt.plot(x.detach().numpy(),y.detach().numpy())
    plt.xlabel('x')
    plt.ylabel(name+'(x)')
    plt.subplot(1,2,2)
    plt.plot(x.detach().numpy(), x.grad.detach().numpy())
    plt.xlabel('x')
    plt.ylabel('grad of '+name+'(x)')
    plt.show()

def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,legend = None,figsize=(3.5,3.5)):
    # dl2.set_figsize(figsize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # print(x_vals,y_vals)
    plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        plt.semilogy(x2_vals,y2_vals,linestyle=':')
        plt.legend(legend)
    plt.show()

