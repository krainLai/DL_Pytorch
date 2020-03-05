#拟合情况模拟
import torch
import numpy as np
import sys
import d2lzh_pytorch as dl2

#生成三阶多项式函数 y = 1.2x - 3.4x^2 +5.6X^3 +5+e e为服从均值为0,标准差为0.01的正态分布
#训练数据集和测试数据集都设为100
#生成数据集...
n_train,n_test,true_w,true_b = 100,100,[1.2,-3.4,5.6],5
features = torch.randn((n_train+n_test , 1),dtype=torch.float) #200*1
# print(features,features.shape)
poly_features = torch.cat((features,torch.pow(features,2),torch.pow(features,3)),1)#cat(..,1)按列拼接多项式特征项 200*3
# print(poly_features.shape)
# print(poly_features.dtype)
labels = true_w[0]*poly_features[:,0] + true_w[1]*poly_features[:,1] + true_w[2]*poly_features[:,2] + true_b
labels += torch.tensor(np.random.normal(0,0.01,size=labels.size()),dtype=torch.float)

def semilogy(x_vals,y_vals,x_label,y_label,x2_vals=None,y2_vals=None,legend = None,figsize=(3.5,3.5)):
    # dl2.set_figsize(figsize)
    dl2.plt.xlabel(x_label)
    dl2.plt.ylabel(y_label)
    # print(x_vals,y_vals)
    dl2.plt.semilogy(x_vals,y_vals)
    if x2_vals and y2_vals:
        dl2.plt.semilogy(x2_vals,y2_vals,linestyle=':')
        dl2.plt.legend(legend)
    dl2.plt.show()



num_epochs,loss = 1000,torch.nn.MSELoss()

def fit_and_plot(train_features,test_features,train_labels,test_lebels):
    # print('测试集形状',test_lebels.shape)
    net = torch.nn.Linear(train_features.shape[-1],1)
    # print('net', net)

    batch_size = min(10,train_labels.shape[0])
    # print('batch_size', batch_size)
    dataset = torch.utils.data.TensorDataset(train_features,train_labels)
    train_iter = torch.utils.data.DataLoader(dataset,batch_size,shuffle=True)

    optimizer = torch.optim.SGD(net.parameters(),lr=0.01)
    train_ls,test_ls = [],[]
    for i in range(num_epochs):
        # print("index:",i+1)
        for x,y in train_iter:
            # print('x',x)
            l = loss(net(x),y.view(-1,1))
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_labels = train_labels.view(-1,1)
        test_lebels = test_lebels.view(-1,1)
        train_ls.append(loss(net(train_features),train_labels).item())
        test_ls.append(loss(net(test_features), test_lebels).item())

        # testloss = loss(net(test_features),test_lebels).item()
        # trainloss = loss(net(train_features),train_labels).item()
        # print(testloss)
        # print(trainloss)

    print('final epoch: train loss:',train_ls[-1],', test loss:',test_ls[-1])
    semilogy(range(1,num_epochs+1),train_ls,'epochs','loss',
             range(1,num_epochs+1),test_ls,['train','test'])
    # print('final epoch: train loss', train_ls, '\ntest loss', test_ls)
    print('weight',net.weight.data,'\nbias:',net.bias.data)

#正常拟合
# fit_and_plot(poly_features[:n_train,:],poly_features[n_train:,:],labels[:n_train],labels[n_train:])

#欠拟合
# fit_and_plot(features[:n_train,:],features[n_train:,:],labels[:n_train],labels[n_train:])

#过拟合
fit_and_plot(poly_features[:3,:],poly_features[n_train:,:],labels[:3],labels[n_train:])
