#卷积层得名与卷积运算，但我们通常在卷积层中使用更加直观的互相关 cross-correlation运算

# 0 1 2
# 3 4 5    *   0 1  =  19 25
# 6 7 8        2 3     37 43

#用代码实现上述二维互相关运算
import torch
from torch import nn

def corr2d(X,K):
    h,w = K.shape
    Y = torch.zeros((X.shape[0]-h+1,X.shape[1]-w+1))
    for i in  range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j] = (X[i:i+h,j:j+w]*K).sum() #点乘再求和
    return Y

X = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
K = torch.tensor([[0,1],[2,3]])
Y = corr2d(X,K)
print(Y)
