# ImageNet Classification with Deep Convolutional Neural Network

## 1. 从LeNet-5开始

使用卷积网络解决图像分类的问题可以往前追溯到1998年LeCun发表的LeNet，解决手写数字识别一文。LeNet又名LeNet-5，是因为在LeNet中，使用的均是5\*5的卷积核。LeNet的结构如图1。

\[AlexNet\_1\]

AlexNet中使用的结构直接影响了其之后沿用至今，卷积+池化+全连接至今仍然是最主流的结构，下面我们简单分析一下LeNet的结构。

**INPUT\(32\*32\)**：32\*32的手写数字（共10类）的黑白图片[^1]

**C1：**C1层使用了6个卷积核，每个卷积核的大小均是5\*5，pad=0, stride=1，激活函数使用的是反正切tanh，所以一次卷积之后，Feature Map的大小是\(32-5+1\)/1=28，该层共有28\*28\*6=4704个神经元。加上偏置项，该层共有\(5\*5+1\)\*6=146个参数

```
f(x) = tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
```

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

AlexNet名字取自该论文的第一作者Alex Krizhevsky。在120万张图片的1000类分类任务上的top-1精度是37.5%，top-5则是15.3%，直接比第二的26.2%高出了近10个百分点。AlexNet取得如此成功的原因是其使网络的宽度和深度达到了前所有为的高度，而该模型也使网络的可学参数达到了58,322,314个。为了学习该网络，AlexNet并行使用了两块GTX 580，大幅提升了训练速度。这些共同促成了AlexNet的形成。

下面，我们来详细分析一下AlexNet，AlexNet的结构如下图

\[AlexNet\_3.png\]

keras实现的AlexNet代码如下

```
# 构建AlexNet-5网络
model = Sequential()
model.add(Conv2D(input_shape = (227,227,3), strides = 4, filters=96, kernel_size=(11,11), padding='valid', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3,3), strides=2))
model.add(Conv2D(filters=256, kernel_size=(5,5), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(3,3), strides=2))
model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=384, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(filters=256, kernel_size=(3,3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2), strides=2))
model.add(Flatten())
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.summary()
```

根据keras提供的summary\(\)工具，可以得到图4的AlexNet统计图

### 2.1 多GPU训练

首先对比图1和图3，我们发现AlexNet将网络分成了两个部分，这幅图这么画的原因是为了提升训练速度，作者使用了两块GPU\(叫做GPU1和GPU2\)并行训练模型，例如第二个卷积每个GPU只使用自身显存中的feature map，而第三个卷积是需要使用另外一个GPU显存中的feature map。不过得益于TensorFlow等开源框架对多机多卡的支持和显卡显存的提升，我们已经不太关心网络的底层实现了，所以这一部分就不再赘述。

\[AlexNet\_4.png\]

### 2.2 整流线性单元ReLU

在LeNet5中，论文使用了tanh作为激活函数，反正切的函数曲线如图5

\[AlexNet\_5.png\]

在BP的反向过程中，局部梯度会与整个损失函数关于该单元输出的梯度相乘。因为，当tanh\(x\)中的x的绝对值比较大的时候，此时该单元的梯度便非常接近于0，这样就会杀死梯度，导致没有更新值传过该梯度了。在深度学习中，这个现象叫做“饱和”。同样sigmoid激活函数也存在饱和的现象。

为了解决这个问题，AlexNet引入了ReLU激活函数

```
f(x) = max(0,x)
```

ReLU的函数曲线如下图

\[AlexNet\_6.jpg\]

在ReLU中，无论x的取值是多大，f\(x\)导数都是1，也就不存在导数过小导致的饱和的发生了。图7是我在MNIST数据集，根据LeNet

5使用不同的激活函数中得到的不同模型的收敛速度。

\[AlexNet\_7.jpg\]

### 2.3 LRN

局部响应归一化是一个已近被淘汰的算法，有VGG\[3\]的论文中已经指出，LRN并没有什么效果[^1]。在现在的网络中，LRN已经被其它归一化方法所替代，例如我在上面代码中给出的Batch Normalization。LRN是使用同一位置临近的feature map来归一化当前feature map的值的一种方法，其表达式为

```
b_{x,y}^i = \frac{a^i_{x,y}}{(k+\alpha\sum^{min(N-1,i+n/2}_{j=max(0,i-n/2)}(a^j_{x,y})^2)^\beta}
```

其中N表示feature map的数量，n=5, k=2, \alpha=0.5, \beta=0.75，这些值均由验证集得出。

### 2.4 Overlap pooling

当进行pooling的时候，如果步长stride小于pooling核的尺寸，相邻之间的pooling核会有相互覆盖的地方，这种方式便叫做overlap pooling。论文中指出这种方式可以减轻过拟合，至今未想通原因。

### 2.5 Dropout



Reference

\[1\] LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition\[J\]. Proceedings of the IEEE, 1998, 86\(11\): 2278-2324.

\[2\] Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks\[C\]//Advances in neural information processing systems. 2012: 1097-1105.

\[3\] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition\[J\]. arXiv preprint arXiv:1409.1556, 2014.

[^1]: Such normalisation does not improve the performance on the ILSVRC dataset, but leads to increased memory con- sumption and computation time. Where applicable, the parameters for the LRN layer are those of \(Krizhevsky et al., 2012\).

