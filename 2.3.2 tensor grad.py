import torch

x = torch.ones(2,2,requires_grad=True)
print(x)
print(x.grad_fn)#创建tensor x的对象,因为tensor是直接创建的,无创建对象,因此为None

y = x+2
print(y)
print(y.grad_fn) #y是由加法创建的,它有一个<AddBackward>的grad_fn

#像上述x,是直接创建的,称为叶子节点,叶子节点对应的grad_fn是None
print(x.is_leaf,y.is_leaf)



#再来点复杂度运算操作
z = y*y*3
out = z.mean()
print(z,out)



#可以通过 in-palce的方式来改变requires_grad
a = torch.randn(2,2)#默认情况下 require_grad = False
a = ((a*3)/(a-1))
print(a.requires_grad)#False
a.requires_grad_(True)
print(a.requires_grad)#True
b = (a*a).sum()
print(b.grad_fn) #<SumBackward>

