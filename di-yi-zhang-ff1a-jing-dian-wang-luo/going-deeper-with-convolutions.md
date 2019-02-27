# Going Deeper With Convolutions

## 前言

2012年之后，卷积网络的研究分成了两大流派，并且两个流派都在2014年有重要的研究成果发表。第一个流派是增加卷积网络的深度和宽度，经典的网络有ImageNet 2013年冠军ZF-net{{"zeiler2014visualizing"|cite}}以及我们在上篇文章中介绍的VGG系列{{"simonyan2014very"|cite}}。另外一个流派是增加卷积核的拟合能力，或者说是增加网络的复杂度，典型的网络有可以拟合任意凸函数的Maxout Networks{{"goodfellow2013maxout"|cite}}，可以拟合任意函数的Network in Network \(NIN\){{"lin2013network"|cite}}，以及本文要解析的基于Inception的GoogLeNet{{"szegedy2015going"|cite}}。为了能更透彻的了解GoogLeNet的思想，我们首先需要了解Maxout和NIN两种结构。

## 1. 背景知识

### 1.1 Maxout Networks

在之前介绍的AlexNet中，引入了Dropout {{"hinton2012improving"|cite}}来减轻模型的过拟合的问题。Dropout可以看做是一种集成模型的思想，在每个step中，会将网络的隐层节点以概率$$p$$置0。Dropout和传统的bagging方法主要有以下两个方面不同：

1. Dropout的每个子模型的权值是共享的；
2. 在训练的每个step中，Dropout每次会使用不同的样本子集训练不同的子网络。

这样在每个step中都会有不同的节点参与训练，减轻了节点之间的耦合性。在测试时，使用的是整个网络的所有节点，只是节点的输出值要乘以Dropout的概率$$p$$。

作者认为，与其像Dropout这种毫无选择的平均，我们不如有条件的选择节点来生成网络。在传统的神经网络中，第i层的隐层的计算方式如下（暂时不考虑激活函数）：

$$h_{i}= Wx_{i}+b_i$$

假设第i-1层和第i层的节点数分别是d和m，那么W则是一个$$d\times m$$的二维矩阵。而在Maxout网络中，W是一个**三维**矩阵，矩阵的维度是$$d\times m\times k$$，其中$$k$$表示Maxout网络的通道数，是Maxout网络唯一的参数。Maxout网络的数学表达式为：

$$h_i = max_{j\in[1,k]}z_{i,j}$$

其中$$z_{i, j}=x^TW_{...i,j}+b_{i,j}$$。

下面我们通过一个简单的例子来说明Maxout网络的工作方式。对于一个传统的网络，假设第$$i$$层有两个节点，第$$i+1$$层有1个节点，那么MLP的计算公式就是：

$$out = g(W*X+b)$$

其中$$g(\cdot)$$是激活函数，如$$tanh$$，$$relu$$等，如图1。

###### 图1：传统神经网络

![](/assets/Maxout_1.png)

如果我们将Maxout的参数k设置为5，Maxout网络可以展开成图2的形式：

###### 图2：Maxout网络

![](/assets/Maxout_2.png)

其中$$z=max(z1, z3, z3, z4, z5)$$。

其中$$z1$$-$$z5$$为线性函数，所以$$z$$可以看做是分段线性的激活函数。Maxout Network的论文中给出了证明，当k足够大时，Maxout单元可以以任意小的精度逼近任何凸函数，如图3。

###### 图3：Maxout的凸函数无线逼近性

![](/assets/Maxout_3.png)

在keras2.0之前的版本中，我们可以找到Maxout网络的实现，其核心代码只有一行。

```
output = K.max(K.dot(X, self.W) + self.b, axis=1)
```

Maxout网络存在的最大的一个问题是，网络的参数是传统网络的k倍，k倍的参数数量并没有带来其等价的精度提升，现基本已被工业界淘汰。

### 1.2 Network in Network

Maxout节点可以逼近任何凸函数，而NIN的节点理论上可以逼近任何函数。在NIN中，作者也是采用整图滑窗的形式，只是将卷积网络的卷积核替换成了一个小型的MLP网络，如图4所示：

###### 图4：Network in Network网络结构![](/assets/NIN_1.png)

在卷积操作中，一次卷积操作仅相当于卷积核和滑窗的一次矩阵乘法，其拟合能力有限。而MLP替代卷积操作则增加了每次滑窗的拟合能力，下图是将LeNet5改造成NIN在MNIST上的训练过程收敛曲线，通过实验结果我们可以得到三个重要信息：

1. NIN的参数数量远大于同类型的卷积网络；
2. NIN的收敛速度快于经典网络；
3. NIN的训练速度慢于经典网络。

###### 图5：NIN vs LeNet5

![](/assets/NIN_2.png)

```py
NIN = Sequential()
NIN.add(Conv2D(input_shape=(28,28,1), filters= 8, kernel_size = (5,5),padding = 'same',activation = 'relu'))
NIN.add(Conv2D(input_shape=(28,28,1), filters= 8, kernel_size = (1,1),padding = 'same',activation = 'relu'))
NIN.add(Flatten())
NIN.add(Dense(196,activation = 'relu'))
NIN.add(Reshape((14,14,1),input_shape = (196,1)))
NIN.add(Conv2D(16,(5,5),padding = 'same',activation = 'relu'))
NIN.add(Conv2D(16,(1,1),padding = 'same',activation = 'relu'))
NIN.add(Flatten())
NIN.add(Dense(120,activation = 'relu'))
NIN.add(Dense(84,activation = 'relu'))
NIN.add(Dense(10))
NIN.add(Activation('softmax'))
NIN.summary()
```

实验内容见链接：[https://github.com/senliuy/CNN-Structures/blob/master/NIN.ipynb](https://github.com/senliuy/CNN-Structures/blob/master/NIN.ipynb)

处对比全连接，NIN中的$$1\times1$$卷积操作保存了网络隐层节点和输入图像的位置关系，NIN的思想反而在物体定位和语义分割上得到了更广泛的应用。除了保存Feature Map的图像位置关系，$$x = y$$卷积还有两个用途：

1. 实现Feature Map的升维和降维；
2. 实现跨Feature Map的交互。

另外，NIN提出了使用Global Average Pooling来减轻全连接层的过拟合问题，即在卷积的最后一层，直接将每个Feature Map求均值，然后再接softmax。

### 1.3 Inception v1

GoogLeNet的核心部件叫做Inception。根据感受野的递推公式，不同大小的卷积核对应着不同大小的感受野。例如在VGG的最后一层，$$1\times1$$，$$3\times3$$和$$5\times5$$卷积的感受野分别是196，228，260。我们根据感受野的计算公式也可以知道，网络的层数越深，不同大小的卷积对应在原图的感受野的大小差距越大，这也就是为什么Inception通常在越深的层次中效果越明显。在每个Inception模块中，作者并行使用了$$1\times1$$，$$3\times3$$和$$5\times5$$三个不同大小的卷积核。同时，考虑到池化一直在卷积网络中扮演着积极的作用，所以作者建议Inception中也要加入一个并行的max pooling。至此，一个naive版本的Inception便诞生了，见图6

###### 图6：Naive Inception

![](/assets/Inception_1.png)

但是这个naive版本的Inception会使网络的Feature Map的数量乘以4，随着Inception数量的增多，Feature Map的数量会呈指数形式的增长，这对应着大量计算资源的消耗。为了提升运算速度，Inception使用了NIN中介绍的$$1\times1$$卷积在卷机操作之前进行降采样，由此便诞生了Inception v1，见图7。

###### 图7：Inception结构

![](/assets/Inception_2.png)

Inception的代码也比较容易实现，建立4个并行的分支并在最后merge到一起即可。为了运行MNIST数据集，我使用了更窄的网络（Feature Map数量均为4），论文中feature map的数量已注释在代码中。

```py
def inception(x):
    inception_1x1 = Conv2D(4,(1,1), padding='same', activation='relu')(x) #64
    inception_3x3_reduce = Conv2D(4,(1,1), padding='same', activation='relu')(x) #96
    inception_3x3 = Conv2D(4,(3,3), padding='same', activation='relu')(inception_3x3_reduce) #128
    inception_5x5_reduce = Conv2D(4,(1,1), padding='same', activation='relu')(x) #16
    inception_5x5 = Conv2D(4,(5,5), padding='same', activation='relu')(inception_5x5_reduce) #32
    inception_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(x) #192
    inception_pool_proj = Conv2D(4,(1,1), padding='same', activation='relu')(inception_pool) #32
    inception_output = merge([inception_1x1, inception_3x3, inception_5x5, inception_pool_proj], 
                                mode='concat', concat_axis=3)
    return inception_output
```

图8是使用相同Feature Map总量的Inception替代卷积网络的一层卷积的在MNIST数据集上的收敛速度对比，从实验结果可以看出，对于比较小的数据集，Inception的提升非常有限。对比两个网络的容量，我们发现Inception和相同Feature Map的3\*3卷积拥有相同数量的参数个数，实验内容见链接：[https://github.com/senliuy/CNN-Structures/blob/master/Inception.ipynb](https://github.com/senliuy/CNN-Structures/blob/master/Inception.ipynb)。

###### 图8：Inception vs 卷积网络

![](/assets/Inception_3.png)

### 1.4 GoogLeNet

GoogLeNet取名是为了致敬第一代深度卷积网络LeNet5，作者通过堆叠Inception的形式构造了一个由9个Inception模块共22层的网络，并一举拿下了ILSVRC2014图像分类任务的冠军。GoogLeNet的网络结构如图9 \(高清图片参考论文\)。

###### 图9：GoogLeNet

![](/assets/GoogLeNet_1.png)

对比其他网络，GoogLeNet的一个最大的不同是在中间多了两个softmax分支作为辅助损失函数（auxiliary loss）。在训练时，这两个分类器的损失会以0.3的比例添加到损失函数上。根据论文的解释，该分支有两个作用：

1. 保证较低层提取的特征也有分类物体的能力；
2. 具有提供正则化并克服梯度消失问题的能力；

需要注意的是，在测试的时候，这两个softmax分支会被移除。

辅助损失函数的提出，是遵循信息论中的数据处理不等式（Data Processing Inequality, DPI），所谓数据处理不等式，是指数据处理的步骤越多，则丢失的信息也会越多，表达如下

$$X \rightarrow Y \rightarrow Z$$

$$I(X;Z) \leq I(X;Y)$$

上式也就是说，在数据传输的过程中，信息有可能消失，但绝对不会凭空增加。反应到BP中，也就是在计算梯度的时候，梯度包含的损失信息会逐层减少，所以GoogLeNet网络的中间层添加了两组损失函数以防止损失的过度丢失。

### 1.5 Inception V2

在VGG中，我们讲解过，一个5\*5的卷积核与两个3\*3的卷积核拥有相同大小的感受野，但是两个3\*3的卷积核拥有更强的拟合能力，所以在Inception V2 {{"ioffe2015batch"|cite}}的版本中，作者将5\*5的卷积替换为两个3\*3的卷积。其实本文的最大贡献是Batch normalization的提出，关于BN，我们会另开一个版块单独讲解。

```py
def inception_v2(x):
    inception_1x1 = Conv2D(4,(1,1), padding='same', activation='relu')(x)
    inception_3x3_reduce = Conv2D(4,(1,1), padding='same', activation='relu')(x)
    inception_3x3 = Conv2D(4,(3,3), padding='same', activation='relu')(inception_3x3_reduce)
    inception_5x5_reduce = Conv2D(4,(1,1), padding='same', activation='relu')(x)
    inception_5x5_1 = Conv2D(4,(3,3), padding='same', activation='relu')(inception_5x5_reduce)
    inception_5x5_2 = Conv2D(4,(3,3), padding='same', activation='relu')(inception_5x5_1)
    inception_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    inception_pool_proj = Conv2D(4,(1,1), padding='same', activation='relu')(inception_pool)
    inception_output = merge([inception_1x1, inception_3x3, inception_5x5_2, inception_pool_proj], 
                                mode='concat', concat_axis=3)
    return inception_output
```

###### 图10：Inception v2

![](/assets/Inception_V2.png)

### 1.6 Inception V3

Inception V3{{"szegedy2016rethinking"|cite}}是将Inception V1和V2中的$$n\times n$$卷积换成一个$$n\times1$$和一个$$1\times n$$的卷积，这样做带来的好处有以下几点：

1. 节约了大量参数，提升了训练速度，减轻了过拟合的问题；
2. 多层卷积增加了模型的拟合能力；
3. 非对称卷积核的使用增加了特征的多样性。

```py
def inception_v3(x):
    inception_1x1 = Conv2D(4,(1,1), padding='same', activation='relu')(x)
    inception_3x3_reduce = Conv2D(4,(1,1), padding='same', activation='relu')(x)
    inception_3x1 = Conv2D(4,(3,1), padding='same', activation='relu')(inception_3x3_reduce)
    inception_1x3 = Conv2D(4,(1,3), padding='same', activation='relu')(inception_3x1)
    inception_5x5_reduce = Conv2D(4,(1,1), padding='same', activation='relu')(x)
    inception_5x1 = Conv2D(4,(5,1), padding='same', activation='relu')(inception_5x5_reduce)
    inception_1x5 = Conv2D(4,(1,5), padding='same', activation='relu')(inception_5x1)
    inception_pool = MaxPool2D(pool_size=(3,3), strides=(1,1), padding='same')(x)
    inception_pool_proj = Conv2D(4,(1,1), padding='same', activation='relu')(inception_pool)
    inception_output = merge([inception_1x1, inception_1x3, inception_1x5, inception_pool_proj], 
                                mode='concat', concat_axis=3)
    return inception_output
```

###### 图11：Inception v3

![](/assets/Inception_V3.png)

### 1.7 Inception V4

Inception V4 {{"szegedy2017inception"|cite}}将残差网络{{""|cite}}融合到了Inception模型中，即相当于在Inception中加入了一条输入到输出的short cut。

