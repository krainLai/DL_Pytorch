import torch

x = torch.ones(2,2,requires_grad=True)
# print(x)
# print(x.grad_fn)#创建tensor x的对象,因为tensor是直接创建的,无创建对象,因此为None

y = x+2
# print(y)
# print(y.grad_fn) #y是由加法创建的,它有一个<AddBackward>的grad_fn

#再来点复杂度运算操作
z = y*y*3
out = z.mean()
print(z,out)

out.backward()
print(x.grad)