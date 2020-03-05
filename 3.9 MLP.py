#多层感知机的从0实现
import torch
import numpy as np
import d2lzh_pytorch as d2l

if __name__ =='__main__':
    #加载mnist数据集
    batch_size = 256
    train_iter,test_iter = d2l.load_data_fashion_mnist(batch_size)

    #fashion数据集的图像形状为28*28,类别数为10,本节依然使用28*28=784的向量表示每一张图像.
    # 因此输入个数为784.实验中我们设置超参数隐藏层单元数为256
    num_inputs,num_outputs,num_hiddens = 784,10,256

    w1 = torch.tensor(np.random.normal(0,0.01,(num_inputs,num_hiddens)),dtype=torch.float)
    b1 = torch.zeros(num_hiddens,dtype=torch.float)

    w2 = torch.tensor(np.random.normal(0,0.01,(num_hiddens,num_outputs)),dtype=torch.float)
    b2 = torch.zeros(num_outputs,dtype=torch.float)

    params = [w1,b1,w2,b2]
    for param in params:#添加梯度更新标识符
        param.requires_grad_(requires_grad=True)

    #自定义relu函数
    def relu(x):
        return torch.max(input=x,other=torch.tensor(0.0))

    def net(x):
        x = x.view((-1,num_inputs))
        h = relu(torch.matmul(x,w1)+b1)
        return torch.matmul(h,w2)+b2

    loss = torch.nn.CrossEntropyLoss()

    num_epochs ,lr= 5,100.0
    d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,params,lr)

    #增加显示
    x, y = iter(test_iter).next()

    true_labels = d2l.get_fashion_mnist_labels(y.numpy())
    pred_labels = d2l.get_fashion_mnist_labels(net(x).argmax(dim=1).numpy())
    titles = [true + "\n" + pred for true, pred in zip(true_labels, pred_labels)]

    d2l.show_fashion_mnist(x[0:9], titles[0:9])

