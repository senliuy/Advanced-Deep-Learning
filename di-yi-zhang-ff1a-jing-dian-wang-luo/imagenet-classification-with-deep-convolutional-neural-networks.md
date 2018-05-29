# ImageNet Classification with Deep Convolutional Neural Network

## 1. 从LeNet-5开始

使用卷积网络解决图像分类的问题可以往前追溯到1998年LeCun发表的LeNet，解决手写数字识别一文。LeNet又名LeNet-5，是因为在LeNet中，使用的均是5\*5的卷积核。LeNet的结构如图1。

![](/assets/AlexNet_1.png)

AlexNet中使用的结构直接影响了其之后沿用至今，卷积+池化+全连接至今仍然是最主流的结构，下面我们简单分析一下LeNet的结构。

**INPUT\(32\*32\)**：32\*32的手写数字（共10类）的黑白图片

**C1:** C1层使用了6个卷积核，每个卷积核的大小均是5\*5，pad=0, stride=1, 所以一次卷积之后，图像的大小是\(32-5+1\)/1=28$$$$，该层共有28\*28\*6=4704个神经元。

**S2: **S2层是卷积网络常使用的降采样层，在LeNet中，降采样的步长是2，降采样核的大小也是2。经过S2层，图像的大小减小一半，变成14\*14。



## Reference

\[1\] LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition\[J\]. Proceedings of the IEEE, 1998, 86\(11\): 2278-2324.

