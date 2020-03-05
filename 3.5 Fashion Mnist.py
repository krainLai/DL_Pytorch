import torch
import torchvision
import torchvision.transforms as transfroms
import matplotlib.pyplot as plt
import time
import sys
# sys.path.append("..")
import d2lzh_pytorch as d2l
import torch.utils.data as Data

if __name__ == '__main__': #此处不加,DataLoader中的 num_workers>0 会报错,因为运行缺少主程序,不能开启多线程
    #从网上下载数据集
    mnist_train = torchvision.datasets.FashionMNIST(root="./Dataset/FashionMnist",train=True, download=True,transform=transfroms.ToTensor())
    mnist_test  = torchvision.datasets.FashionMNIST(root="./Dataset/FashionMnist",train=False,download=True,transform=transfroms.ToTensor())

    # print(type(mnist_train))
    # print(len(mnist_train),len(mnist_test))

    #获取数据集
    x,y = [],[]
    for i in range(10):
        x.append(mnist_train[i][0])
        y.append(mnist_train[i][1])

    # d2l.show_fashion_mnist(x,d2l.get_fashion_mnist_labels(y))

    # 读取小批量数据,使用多进程来加速数据读取
    batch_size = 256
    if sys.platform.startswith('Win'):
        num_workers = 0#0表示不需要额外的进程来加速读取数据
    else:
        num_workers = 3#四个进程加速 超过3会出现页面文件太小,无法操作

    train_iter = Data.DataLoader(mnist_train,batch_size=batch_size,shuffle=True,num_workers=num_workers)
    test_iter = Data.DataLoader(mnist_test,batch_size=batch_size,shuffle=False,num_workers=num_workers)
    start = time.time()

    for i,data in enumerate(train_iter):
        # x,y = data
        # print(x,y)
        # break
        continue

    print("time:{:.2f} sec".format(time.time() - start))