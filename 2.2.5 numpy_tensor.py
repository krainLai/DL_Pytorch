#numpy 和 tensor的转换

import torch

#numpy() he form_numpy() 将tensor和numpy中的数组互相转换,但是他们会进行内存共享

#tensor转numpy,以下例子中,a,b其实是共享内存的
print("tensor转numpy")
a = torch.ones(5)
b = a.numpy()
print(a,b)

a +=1
print(a,b)

b+=1
print(a,b)

#numpy转tensor
print("numpy转tensor")

import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
print(a,b)

a+=1
print(a,b)
b+=1
print(a,b)


#不再共享内存的转换方式,该方法总是进行数据拷贝,不够前面的方法快
print("numpy转tensor")
c = torch.tensor(a)
a+=1
print(a,c)
