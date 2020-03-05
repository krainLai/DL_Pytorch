import torch

#创建未初始化的tensor
x= torch.empty(5,3)
print(x)

#创建随机初始化的tensor
x = torch.rand(5,3)
print(x)

#创建一个5*3long 型全0的tensor
x = torch.zeros(5,3,dtype=torch.long)
print(x)

#根据数据创建tensor
x = torch.tensor([5.5,3])
print(x)

#通过现有的tensor创建tensor
#全1的tensor
x = x.new_ones(5,3,dtype=torch.float64) #返回的tensor默认具有相同的torch.dtype和torch.device
print(x)

x = torch.randn_like(x,dtype=torch.float)
print(x)

#通过 size() 和 shape来获取tensor的形状
print(x.size())
print(x.shape)

#对角线为1其他为0
x = torch.eye(3,3,dtype=int)
print(x)

#从 s 到 e,步长为 step
x = torch.arange(2,10,2)
print(x)

#从 s 到 e 均匀分布为step分
x = torch.linspace(0,10,5)
print(x)

