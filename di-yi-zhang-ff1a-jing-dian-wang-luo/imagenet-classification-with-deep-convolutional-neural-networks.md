# ImageNet Classification with Deep Convolutional Neural Network

## 1. 从LeNet-5开始

使用卷积网络解决图像分类的问题可以往前追溯到1998年LeCun发表的LeNet，解决手写数字识别一文。LeNet又名LeNet-5，是因为在LeNet中，使用的均是5\*5的卷积核。LeNet的结构如图1。

\[AlexNet\_1\]

AlexNet中使用的结构直接影响了其之后沿用至今，卷积+池化+全连接至今仍然是最主流的结构，下面我们简单分析一下LeNet的结构。

**INPUT\(32\*32\)**：32\*32的手写数字（共10类）的黑白图片[^1]

**C1：**C1层使用了6个卷积核，每个卷积核的大小均是5\*5，pad=0, stride=1，激活函数使用的是反正切tanh，所以一次卷积之后，Feature Map的大小是\(32-5+1\)/1=28，该层共有28\*28\*6=4704个神经元。加上偏置项，该层共有\(5\*5+1\)\*6=146个参数

**S2：**S2层是卷积网络常使用的降采样层，在LeNet中，使用的是Max Pooling，降采样的步长是2，降采样核的大小也是2。经过S2层，Feature Map的大小减小一半，变成14\*14。该层共有14\*14\*6=1176个神经元。

**C3：**C3层是16个大小为5\*5，深度为6的卷积核，pad=0，stride=1，激活函数=tanh，一次卷积后，Feature Map的大小是\(14-5+1\)/1=10，神经元个数为10\*10\*16 = 1600，参数个数为\(5\*5\*6+1\)\*16 = 2416个参数

**S4**：步长是2，大小是2的Max Pooling降采样，该层使Feature Map变成5\*5，共有5\*5\*16 = 400个神经元

**C5**：节点数为120的全连接，激活函数是tanh，参数个数是\(400+1\)\*120=48120

**F6**：节点数为84的全连接，激活函数是tanh，参数个数是\(120+1\)\*84=10164

**OUTPUT**：10分类的输出层，所以使用的是softmax激活函数，参数个数是\(84+1\)\*10=850

使用keras搭建LeNet5的代码如下

```py
# 构建LeNet-5网络
model = Sequential()
model.add(Conv2D(input_shape = (28,28,1), filters=6, kernel_size=(5,5), padding='valid', activation='tanh'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Conv2D(input_shape=(14,14,6), filters=16, kernel_size=(5,5), padding='valid', activation='tanh'))
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(120, activation='tanh'))
model.add(Dense(84, activation='tanh'))
model.add(Dense(10, activation='softmax'))
```

如图2所示，经过10个epoch后，LeNet5就基本已收敛。

\[AlexNet\_2\]

完整的LeNet5在MNIST上完整训练过程见链接\([https://github.com/senliuy/CNN-Structures/blob/master/LeNet5.ipynb](https://github.com/senliuy/CNN-Structures/blob/master/LeNet5.ipynb)\)。

## 2. AlexNet

LeNet之后，卷积网络沉寂了14年。直到2012年，AlexNet\[2\]在ILSVRC2010一举夺魁，直接把ImageNet的精度提升了10个百分点，它将卷积网络的深度和宽度都提升到了新的高度。从此开始，深度学习开始再计算机视觉的各个领域开始披荆斩棘，至今深度学习仍是最热的话题。AlexNet作为教科书式的网络，值得每个学习深度学习的人深入研究。

AlexNet名字取自该论文的第一作者Alex Krizhevsky。

## Reference

\[1\] LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition\[J\]. Proceedings of the IEEE, 1998, 86\(11\): 2278-2324.

\[2\] Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks\[C\]//Advances in neural information processing systems. 2012: 1097-1105.

