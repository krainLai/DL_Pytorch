#torch 的运算内存开销
import torch


#索引,view不会开辟新的内存,而像 y = x+y 这样的运算会开辟新的内存,然后将y指向新的内存;

x = torch.tensor([1,2])
y = torch.tensor([3,4])

id_before = id(y)#内置id指向对应的内存地址

y = y+x
print(id(y) == id_before)#false

#如果想指定结果到原来的y的内存,我们可以使用前面介绍的索引来进行替换操作.在下面的例子中,我们把x+y的结果通过[:]写进y对应的内存中
x = torch.tensor([1,2])
y = torch.tensor([3,4])
id_before = id(y)

y[:] = x+y
print(id(y)==id_before)#true

#或者使用指定输出的方式
x = torch.tensor([1,2])
y = torch.tensor([3,4])
id_before = id(y)
# torch.add(y,x,out=y)
# y+=x
y.add_(x)
print(id(y)==id_before)#true

