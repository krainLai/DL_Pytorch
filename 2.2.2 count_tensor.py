import torch

#tensor的计算

#加法
print("加法1")
x = torch.zeros(5,3)
y = torch.rand(5,3)
print(x + y)

#加法2
print("加法2")
print(torch.add(x,y))

#还可以指定输出
result = torch.empty(5,3)
torch.add(x,y,out=result)
print(result)

#加法3
print("加法3")
y.add_(x)#将x加到y
print(y)


#索引
print("索引")
#索引结果与源数据共享内存,当索引内容被更改,内存内容也被更改
y = x[0,:]
y +=1
print(y)
print(x[0,:])


#指定维度上面选取
print("指定维度上面选取")
x = torch.eye(5,5)
z = torch.tensor([2])
y = torch.index_select(x,0,z)
print(x)
print(y)

#改变形状,也是内存共享,view只是改变了此张量的观察角度
print("改变形状")
y = x.view(25)
z = x.view(-1,5)#-1指所指的维度可根据其他维度的值推算出来
print(x.size(),y.size(),z.size())

