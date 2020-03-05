#tensor的gpu计算

import torch

x = torch.tensor([1,2,3])
print(x)

#使用.cuda()可以将CPU上的Tensor转换(复制)到GPU上面.
#如果有多块GPU,我们用.cuda(i)来表示第i块GPU及相应的显存(i从0开始)且cuda(0)和cuda()等价
x = x.cuda(0)
print(x)

print(x.device)
print(x.dtype)


#可以直接在创建的时候就指定设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.tensor([1,2,3],device=device)
# 或者
x = torch.tensor([1,2,3]).to(device)
print(x)

#注意,对在GPU上的数据进行运算,那么结果还是存放在GPU上
y = x**2
print(y)

#注意:存储在不同位置上面的数据不可以直接进行计算,即存放在CPU上的数据不能和存放在GPU上面的数据直接进行运算

# z = y + x.cpu()


# a = torch.ones(2,3)
# print(a<3)
# print(torch.__version__)