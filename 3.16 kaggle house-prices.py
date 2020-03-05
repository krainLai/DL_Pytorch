#kaggle 房价预测
# 使用pandas库读入并处理数据

# pip install pandas
import torch
import torch.utils.data as data
import torch.nn as nn
import numpy as np
import d2lzh_pytorch as d2l
import pandas as pd
import sys


# print(torch.__version__)
torch.set_default_tensor_type(torch.FloatTensor)

train_data = pd.read_csv("Dataset/house-prices-dataset/train.csv")
test_data = pd.read_csv("Dataset/house-prices-dataset/test.csv")

# print(train_data.shape)#(1460,81)1460个样本,80个特征,1个标签
# print(test_data.shape)#(1459,80)1459个样本,80个特征
#
# print(train_data.iloc[0:4,[0,1,2,3,-3,-2,-1]])

# print(train_data.iloc[0:4,[1,-1]])
# print(test_data.iloc[0:4,[1,-1]])

#因为训练以及测试数据的第一项特征是ID,它能帮助模型记住每个训练样本,但是难以推广到测试样本,所以不使用它来进行训练
#我们将所有的训练数据和测试数据的79个特征按样本连结
all_features = pd.concat((train_data.iloc[:,1:-1],test_data.iloc[:,1:]))

#数据预处理 ,对连续数值的特征做标准化:设该特征在整个数据集上的均值为u,标准差为q.那么,我们可以将该特征的每个值先减去u,在除以q,得到标准化后的每个特征值;
# 对于缺失的特征值,我们将其替换成该特征的均值
# pandas里面的object是除了非数值之外的数据表示
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# print(numeric_features)
#每列数值数据进行 先减去u,在除以q,得到标准化后数值
all_features[numeric_features] = all_features[numeric_features].apply(lambda x:(x-x.mean())/(x.std()))
# print(all_features[numeric_features[:3]])
#因为标准化后,每个特征的均值变为0,所以可以用0来替换缺失值
all_features = all_features.fillna(0) #fillna 填充nan数据
# print(all_features['Fence'])
#dummy_na = True 将缺失值也当作合法的特征值并为其创建指示特征
#get_dummies 是利用pandas实现one hot encode(独热编码)的方式
all_features = pd.get_dummies(all_features,dummy_na=True)
print(all_features.shape) #get_dummies后,特征数从79,增加到了354
# print(all_features[:3])
#最后,通过values属性,得到numpy格式数据,并转成ndarray方便后面的训练
n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values,dtype=torch.float)
test_features = torch.tensor(all_features[n_train:].values,dtype=torch.float)
train_labels = torch.tensor(train_data.SalePrice.values,dtype=torch.float).view(-1,1)


#训练模型:使用一个基本的线性回归模型和平方损失函数来训练模型
loss = nn.MSELoss()

def get_net(feature_num):
    net = nn.Linear(feature_num,1)
    for param in net.parameters():
        nn.init.normal_(param,mean=0,std=0.01)
    return net
#定义比赛用来评价模型的对数均方根误差
def log_rmse(net,features,labels):
    with torch.no_grad():
        #将小于1的值设成1,使得取对数时的数值更稳定
        clipped_preds = torch.max(net(features),torch.tensor(1.0))
        rmse = torch.sqrt(2*loss(clipped_preds.log(),labels.log()).mean())
    return rmse.item()

def train(net,train_features,train_labels,test_features,test_labels,num_epochs,lr,weight_decay,batch_size):
    train_ls,test_ls = [],[]
    # print("train:",train_features.shape,train_labels.shape)
    dataset = data.TensorDataset(train_features,train_labels)
    train_iter = data.DataLoader(dataset,batch_size,shuffle=True)
    #adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay)#对权重参数进行衰减,预防过拟合
    net = net.float()
    for epoch in range(num_epochs):
        for x,y in train_iter:
            l = loss(net(x.float()),y.float())
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
        train_ls.append(log_rmse(net,train_features,train_labels))
        if test_labels is not None:
            test_ls.append(log_rmse(net,test_features,test_labels))

    return train_ls,test_ls

#k折交叉验证函数,返回k折交叉时所需的训练和验证数据
def get_k_fold_data(k,i,x,y):
    assert k>1
    fold_size = x.shape[0] // k
    x_train,y_train = None,None
    for j in range(k):
        idx = slice( j*fold_size,(j+1)*fold_size)
        x_part,y_part = x[idx,:],y[idx]
        # print('x_part:',j,'/',k,x_part.shape)
        if j == i:
            x_valid,y_valid = x_part,y_part
        elif x_train is None:
            x_train,y_train = x_part,y_part
        else:
            x_train = torch.cat((x_train,x_part),dim=0)
            y_train = torch.cat((y_train,y_part),dim=0)

    return x_train,y_train,x_valid,y_valid

#在k折交叉验证中,我们训练k次并返回训练和验证的平均误差
def k_fold(k,x_train,y_train,num_epochs,lr,weight_decay,batch_size):
    train_ls_sum,valid_ls_sum = 0,0
    # print('x_train shape',x_train.shape)
    for i in range(k):
        data = get_k_fold_data(k,i,x_train,y_train)
        net = get_net(x_train.shape[1])
        train_ls, valid_ls = train(net, *data, num_epochs, lr, weight_decay, batch_size)
        train_ls_sum += train_ls[-1]
        valid_ls_sum += valid_ls[-1]
        if i == k-1:
            d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','rmse',
                         range(1,num_epochs+1),valid_ls,['train','valid'])

        print('fold {},train rmse {:.4f},valid rmse {:.4f}'.format(i,train_ls[-1],valid_ls[-1]))
    return train_ls_sum/k ,valid_ls_sum/k

k,num_epochs,lr,weight_decay,batch_size= 5,120,5,0,64
train_l,valid_l = k_fold(k,train_features,train_labels,num_epochs,lr,weight_decay,batch_size)
print(k,'-fold validation,  avg train rmse:',train_l,', avg valid rmse:',valid_l)

def train_and_pred(train_features,test_features,trian_labels,test_data,num_epochs,lr,weight_decay,batch_size):
    net = get_net(train_features.shape[1])
    train_ls,_ = train(net,train_features,train_labels,None,None,num_epochs,lr,weight_decay,batch_size)
    d2l.semilogy(range(1,num_epochs+1),train_ls,'epochs','rmse')
    print('trian rmse:',train_ls[-1])
    preds = net(test_features).detach().numpy()
    test_data['SalePrice'] = pd.Series(preds.reshape(1,-1)[0])
    submission = pd.concat([test_data['Id'],test_data['SalePrice']],axis=1)
    # submission.to_csv('./sunmission.csv',index=False)
    submission.to_csv('Dataset/house-prices-dataset/sunmission.csv', index=False)


train_and_pred(train_features,test_features,train_labels,test_data,num_epochs,lr,weight_decay,batch_size)