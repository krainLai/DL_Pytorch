import torch
#用方法to(),可以将tensor在cpu和gpu(需要硬件支持)之间相互移动

x = torch.tensor([1,2])
if torch.cuda.is_available():
    device = torch.device("cuda")
    y = torch.ones_like(x,device=device)#y复制x的维度以及device

    x = x.to(device)
    z = x +y
    print(z)
    print(z.to("cpu",torch.double))