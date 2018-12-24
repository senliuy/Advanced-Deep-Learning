# ImageNet Classification with Deep Convolutional Neural Network

## 1. 从LeNet-5开始

使用卷积网络解决图像分类的问题可以往前追溯到1998年LeCun发表的LeNet，解决手写数字识别一文。LeNet又名LeNet-5，是因为在LeNet中，使用的均是$$5\times5$$的卷积核。LeNet的结构如图1。

图1：LeNet-5结构

![](/assets/AlexNet_1.png)

AlexNet中使用的结构直接影响了其之后沿用至今，卷积+池化+全连接至今仍然是最主流的结构。卷积操作使网络可以响应和卷积核形状类似的特征，而池化则使网络拥有了一定程度的不变性。下面我们简单分析一下LeNet的结构。

**INPUT\(**$$32\times32$$**\)**：$$32\times32$$的手写数字（共10类）的黑白图片

**C1：**C1层使用了6个卷积核，每个卷积核的大小均是$$5\times5$$，pad=0, stride=1，激活函数使用的是反正切$$tanh$$，所以一次卷积之后，Feature Map的大小是$$(32-5+1)/1=28$$，该层共有$$28\times28\times6=4704$$个神经元。加上偏置项，该层共有$$(5\times5+1)\times6=146$$个参数


$$
f(x) = tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}
$$


**S2：**S2层是卷积网络常使用的降采样层，在LeNet中，使用的是Max Pooling，降采样的步长是2，降采样核的大小也是2。经过S2层，Feature Map的大小减小一半，变成$$14\times14$$。该层共有$$14\times14\times6=1176$$个神经元；

**C3：**C3层是16个大小为$$5\times5$$，深度为6的卷积核，pad = 0，stride = 1，激活函数 = $$tanh$$，一次卷积后，Feature Map的大小是$$(14-5+1)/1=10$$，神经元个数为$$10\times10\times16 = 1600$$，参数个数为$$(5\times5\times6+1)\times16 = 2416$$个参数；

**S4**：步长是2，大小是2的Max Pooling降采样，该层使Feature Map变成$$5\times5$$，共有$$5\times5\times16 = 400$$个神经元。注意池化并不是一种具体的运算，而是代表着一种对统计信息的提取，所以不含有参数。例如在衡量学生考试成绩时，我们不考虑学生试卷的每一道题，而是用总分（平均分）作为评判标准。另外一种常见的池化方式是平均池化（Average Pooling）。

**C5**：节点数为120的全连接，激活函数是$$tanh$$，参数个数是$$(400+1)\times120=48120$$；

**F6**：节点数为84的全连接，激活函数是tanh，参数个数是$$(120+1)\times84=10164$$；

**OUTPUT**：10分类的输出层，所以使用的是softmax激活函数，参数个数是$$(84+1)\times10=850$$。softmax用于分类有如下优点：

1. $$e^x$$使所有样本的值均大于0，且指数的性质使样本的区分度尽量高；
2. softmax所有可能的和为1，反映出分类为该类别的概率，输出选择概率最高的即可。

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

###### 图2：LeNet5在MNIST数据集上的收敛曲线图

![](/assets/AlexNet_2.png)

完整的LeNet5在MNIST上完整训练过程见链接\([https://github.com/senliuy/CNN-Structures/blob/master/LeNet5.ipynb](https://github.com/senliuy/CNN-Structures/blob/master/LeNet5.ipynb)\)。

## 2. AlexNet

LeNet之后，卷积网络沉寂了14年。直到2012年，AlexNet\[2\]在ILSVRC2010一举夺魁，直接把ImageNet的精度提升了10个百分点，它将卷积网络的深度和宽度都提升到了新的高度。从此开始，深度学习开始再计算机视觉的各个领域开始披荆斩棘，至今深度学习仍是最热的话题。AlexNet作为教科书式的网络，值得每个学习深度学习的人深入研究。

AlexNet名字取自该论文的第一作者Alex Krizhevsky。在120万张图片的1000类分类任务上的top-1精度是37.5%，top-5则是15.3%，直接比第二的26.2%高出了近10个百分点。AlexNet取得如此成功的原因是其使网络的宽度和深度达到了前所有为的高度，而该模型也使网络的可学参数达到了58,322,314个。为了学习该网络，AlexNet并行使用了两块GTX 580，大幅提升了训练速度。这些共同促成了AlexNet的形成。

我们知道，当我们想要使用机器学习解决非常复杂的问题时，我们必须使用容量足够大的模型。在深度学习中，增加网络的宽度和网络的深度是提升网络的容量，但是提升容量的同时也会带来两个问题

1. 计算资源的消耗
2. 模型容易过拟合

计算资源是当时限制深度学习发展的最关键的瓶颈，2011年Ciresan\[5\]等人提出了使用GPU部署CNN的技术框架，由此深度学习有了解决其计算瓶颈的硬件支持。

下面，我们来详细分析一下AlexNet，AlexNet的结构如下图

###### 图3：AlexNet的网络结构![](/assets/AlexNet_3.png)

keras实现的AlexNet代码如下

```py
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

根据keras提供的summary\(\)工具，可以得到图4的AlexNet统计图，计算方法参照LeNet5，不再赘述。

###### 图4：根据keras的summary函数得到AlexNet网络参数统计图

![1](/assets/AlexNet_4.png)

### 2.1 多GPU训练

首先对比图1和图3，我们发现AlexNet将网络分成了两个部分，这幅图这么画的原因是为了提升训练速度，作者使用了两块GPU\(叫做GPU1和GPU2\)并行训练模型，例如第二个卷积每个GPU只使用自身显存中的feature map，而第三个卷积是需要使用另外一个GPU显存中的feature map。不过得益于TensorFlow等开源框架对多机多卡的支持和显卡显存的提升，我们已经不太关心网络的底层实现了，所以这一部分就不再赘述。

### 2.2 整流线性单元ReLU

在LeNet5中，论文使用了tanh作为激活函数，反正切的函数曲线如图5

###### 图5：反正切（tanh）函数曲线

![](/assets/AlexNet_5.png)

在BP的反向过程中，局部梯度会与整个损失函数关于该单元输出的梯度相乘。因为，当$$tanh(x)$$中的$$x$$的绝对值比较大的时候，此时该单元的梯度便非常接近于0，在深度学习中，激活函数这个现象叫做“饱和”。同样$$sigmoid$$激活函数也存在饱和的现象。

“饱和”带来的一个深度学习中一个非常致命的问题，那便是**梯度消失**。梯度消失是由于是由BP中链式法则的乘法特性导致的，反应在深度学习的训练过程中便是越接近损失函数的参数，梯度越大，成为了主要学习的函数，而远离损失函数的参数的梯度则非常接近0，导致没有梯度传到这一部分参数，从而使得这一部分的参数很难学习到。

为了解决这个问题，AlexNet引入了ReLU激活函数


$$
f(x) = max(0,x)
$$


ReLU的函数曲线如下图

###### 图6：整流线性单元（relu）函数曲线

![](/assets/AlexNet_6.png)

在ReLU中，无论$$x$$的取值是多大，$$f(x)$$导数都是1，也就不存在导数小于1导致的梯度消失的现象发生了。图7是我在MNIST数据集，根据LeNet5使用不同的激活函数中得到的不同模型的收敛速度。

此外，由于ReLU将小于0的部分全部置0，所以ReLU的另外一个特点就是稀疏性，不仅可以优化网络的性能，还一定程度上缓解了过拟合的问题。

###### 图7：LeNet5使用不同激活函数的收敛速度对比

![](/assets/AlexNet_7.png)

虽然使用ReLU的节点不会有饱和问题，但是会“死掉”————大部分甚至所有的值为负值，从而导致该层的梯度都为0。死神经元是由进入网络的负值引起的（例如在大规模的梯度更新之后可能发生），减少学习率能缓解该现象。

### 2.3 LRN

局部响应归一化，模拟的是动物神经中的横向抑制效应，是一个已近被淘汰的算法，有VGG\[3\]的论文中已经指出，LRN并没有什么效果[^1]。在现在的网络中，LRN已经被其它归一化方法所替代，例如我在上面代码中给出的Batch Normalization。LRN是使用同一位置临近的feature map来归一化当前feature map的值的一种方法，其表达式为


$$
b_{x,y}^i = \frac{a^i_{x,y}}{(k+\alpha\sum^{min(N-1,i+n/2}_{j=max(0,i-n/2)}(a^j_{x,y})^2)^\beta}
$$


其中$$N$$表示feature map的数量，$$n=5, k=2, \alpha=0.5, \beta=0.75$$，这些值均由验证集得出。

另外，AlexNet把LRN方刚在了池化层之前，这在计算上是非常不经济的，一种更好的尝试是把LRN放在池化之后。

### 2.4 Overlap pooling

当进行pooling的时候，如果步长stride小于pooling核的尺寸，相邻之间的pooling核会有相互覆盖的地方，这种方式便叫做overlap pooling。论文中指出这种方式可以减轻过拟合，至今未想通原因。

### 2.5 Dropout

在AlexNet的前两层，作者使用了Dropout\[4\] 来减轻容量高的模型容易发生过拟合的现象。Dropout的使用方法是在训练过程中随机将一定比例的隐层节点置0，Dropout能够缓解过拟合的原因是每次训练都会采样出一个不同的网络架构，但是这些结构是共享权值的。这种技术减轻了节点之间的耦合性，因为一个节点不能依赖网络的其它节点。因此，节点能够学习到更健壮的特征，因为只有这样，节点才能适应每次采样得到的不同的网络结构。在测试时，我们是不对节点进行Drop的。

显然Dropout会减慢收敛速度，但其对减轻过拟合的优异表现仍旧使其在当前的网络中得到广泛的使用。

下图是LeNet-5中加入Dropout之后模型的训练loss曲线图，从图中我们可以看出，加入Dropout之后，训练速度放缓了一些，20个epoch之后，训练集的损失函数也高于没有Dropout的，但是加入Dropout之后，虽然loss=0.0735远高于没有Dropout的0.0155，但是测试集的准确率从0.9826上升到0.9841，可见Dropout对于缓解过拟合还是非常有帮助的。实验代码见：[https://github.com/senliuy/CNN-Structures/blob/master/LeNet\_Dropout.ipynb](https://github.com/senliuy/CNN-Structures/blob/master/LeNet_Dropout.ipynb)

###### 图8：Dropout vs None Dropout

![](/assets/AlexNet_8.png)

## Reference

\[1\] LeCun Y, Bottou L, Bengio Y, et al. Gradient-based learning applied to document recognition\[J\]. Proceedings of the IEEE, 1998, 86\(11\): 2278-2324.

\[2\] Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks\[C\]//Advances in neural information processing systems. 2012: 1097-1105.

\[3\] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition\[J\]. arXiv preprint arXiv:1409.1556, 2014.

\[4\] Hinton G E, Srivastava N, Krizhevsky A, et al. Improving neural networks by preventing co-adaptation of feature detectors\[J\]. arXiv preprint arXiv:1207.0580, 2012.

\[5\] Ciresan D C, Meier U, Masci J, et al. Flexible, high performance convolutional neural networks for image classification\[C\]//IJCAI Proceedings-International Joint Conference on Artificial Intelligence. 2011, 22\(1\): 1237.

[^1]: Such normalisation does not improve the performance on the ILSVRC dataset, but leads to increased memory con- sumption and computation time. Where applicable, the parameters for the LRN layer are those of \(Krizhevsky et al., 2012\).

