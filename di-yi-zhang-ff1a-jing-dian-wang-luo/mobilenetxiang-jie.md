# MobileNet v1 and MobileNet v2

## 前言

MobileNet\[1\]（这里叫做MobileNet v1，简称v1）中使用的Depthwise Separable Convolution是模型压缩的一个最为经典的策略，它是通过将跨通道的$$3\times3$$卷积换成单通道的$$3\times3$$卷积+跨通道的$$1\times1$$卷积来达到此目的的。

MobileNet v2 \[2\]是在v1的Depthwise Separable的基础上引入了[残差结构](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)\[3\]。并发现了ReLU的在通道数较少的Feature Map上有非常严重信息损失问题，由此引入了Linear Bottlenecks和Inverted Residual。

首先在这篇文章中我们会详细介绍两个版本的MobileNet，然后我们会介绍如何使用Keras实现这两个算法。

## 1. MobileNet v1

### 1.1 回顾：传统卷积的参数量和计算量

传统的卷积网络是跨通道的，对于一个通道数为$$M$$的输入Feature Map，我们要得到通道数为$$N$$的输出Feature Map。普通卷积会使用$$N$$个不同的$$D_K \times D_K \times M$$以滑窗的形式遍历输入Feature Map，因此对于一个尺寸为$$D_K\times D_K$$的卷积的参数个数为$$D_K \times D_K \times M \times N$$。一个普通的卷积可以表示为：


$$
G_{k,l,n} = \sum_{i,j,m} \mathbf{K}_{i,j,m,n} \cdot \mathbf{K}_{k+i-1, l+j-1, m}
$$


它的一层网络的计算代价约为：


$$
D_K \times D_K \times M \times N \times D_W \times D_H
$$


其中$$(D_W, D_H)$$为Feature Map的尺寸。普通卷积如图1所示。

![](/assets/MobileNet_1.png)

v1中介绍的Depthwise Separable Convolution就是解决了传统卷积的参数数量和计算代价过于高昂的问题。Depthwise Separable Convolution分成Depthwise Convolution和Pointwise Convolution。

### 1.2 Depthwise卷积

其中Depthwise卷积是指不跨通道的卷积，也就是说Feature Map的每个通道有一个独立的卷积核，并且这个卷积核作用且仅作用在这个通道之上，如图2所示。

![](/assets/MobileNet_2.png)

从图2和图1的对比中我们可以看出，因为放弃了卷积时的跨通道。Depthwise卷积的参数数量为$$D_K \times D_K \times M$$。Depthwise Convolution的数学表达式为：


$$
\hat{G}_{k,l,m} = \sum_{i,j} \hat{K}_{i,j,n} \cdot F_{k+i-1, l+j-1, m}
$$


它的计算代价也是传统卷积的$$\frac{1}{N}$$为:


$$
D_K \times D_K \times M \times D_W \times D_H
$$


在Keras中，我们可以使用[`DepthwiseConv2D`](https://github.com/titu1994/MobileNetworks/blob/master/depthwise_conv.py)实现Depthwise卷积操作，它有几个重要的参数：

* `kernel_size`：卷积核的尺寸，一般设为3。
* `strides`：卷积的步长
* `padding`：是否加边
* `activation`：激活函数

由于Depthwise卷积的每个通道Feature Map产生且仅产生一个与之对应的Feature Map，也就是说输出层的Feature Map的channel数量等于输入层的Feature map的数量。因此`DepthwiseConv2D`不需要控制输出层的Feature Map的数量，因此并没有`filters`这个参数。

### 1.3 Pointwise卷积

Depthwise卷积的操作虽然非常高效，但是它仅相当于对当前的Feature Map的一个通道施加了一个过滤器，并不会合并若干个特征从而生成新的特征，而且由于在Depthwise卷积中输出Feature Map的通道数等于输入Feature Map的通道数，因此它并没有升维或者降维的功能。

为了解决这些问题，v1中引入了Pointwise卷积用于特征合并以及升维或者降维。很自然的我们可以想到使用$$1\times1$$卷积来完成这个功能。Pointwise的参数数量为$$M\times N$$，计算量为：


$$
M\times N \times D_W \times D_H
$$


Pointwise的可视化如图3：

![](/assets/MobileNet_3.png)

### 1.4 Depthwise Separable卷积

合并1.2中的Depthwise卷积和1.3中的Pointwise卷积便是v1中介绍的Depthwise Separable卷积。它的一组操作（一次Depthwise卷积加一次Pointwise卷积）的参数数量为：$$D_K \times D_K \times M + M\times N$$是普通卷积的


$$
\frac{D_K \times D_K \times M + M\times N}{D_K \times D_K \times M \times N} = \frac{1}{N} + \frac{1}{D_K^2}
$$


计算量为：


$$
D_K \times D_K \times M \times D_W \times D_H + M\times N \times D_W \times D_H
$$


和普通卷积的比值为：


$$
\frac{D_K \times D_K \times M \times D_W \times D_H + M\times N \times D_W \times D_H
}{D_K \times D_K \times M \times N \times D_W \times D_H} = \frac{1}{N} + \frac{1}{D_K^2}
$$


对于一个$$3\times3$$的卷积而言，v1的参数量和计算代价均为普通卷积的$$\frac{1}{8}$$左右。

### 1.5 Mobile v1的Keras实现及实验结果分析

通过上面的分析，我们知道一个普通卷积的一组卷积操作可以拆分成了个Depthwise卷积核一个Pointwise卷积，由此而形成MobileNet v1的结构。在这个实验中我们首先会搭建一个普通卷积，然后再将其改造成v1，并在MNIST上给出实验结果，代码和实验结果见链接[TODO]()。

首先我们搭建的传统卷积的结构如下面代码片段：

```py
def Simple_NaiveConvNet(input_shape, k):
    inputs = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', name='n_conv_1')(inputs)
    x = Conv2D(filters=64, kernel_size=(3,3),padding='same', activation='relu', name='n_conv_2')(x)
    x = Conv2D(filters=128, kernel_size=(3,3),padding='same', activation='relu', name='n_conv_3')(x)
    x = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2),padding='same', activation='relu', name='n_conv_4')(x)
    x = GlobalAveragePooling2D(name='n_gap')(x)
    x = BatchNormalization(name='n_bn_1')(x)
    x = Dense(128, activation='relu', name='n_fc1')(x)
    x = BatchNormalization(name='n_bc_2')(x)
    x = Dense(k, activation='softmax', name='n_output')(x)
    model = Model(inputs, x)
    return model
```

通过将$$3\times3$$的`Conv2D()`换成$$3\times3$$的`DepthwiseConv2D`加上$$1\times1$$的`Conv2D()`（第一层保留传统卷积），我们将其改造成了MobileNet v1。

```py
def Simple_MobileNetV1(input_shape, k):
    inputs = Input(shape=input_shape)
    x = Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', name='m_conv_1')(inputs)
    x = DepthwiseConv2D(kernel_size=(3,3),padding='same', activation='relu', name = 'm_dc_2')(x)
    x = Conv2D(filters=64, kernel_size=(1,1),padding='same', activation='relu', name = 'm_pc_2')(x)
    x = DepthwiseConv2D(kernel_size=(3,3), padding='same', activation='relu', name = 'm_dc_3')(x)
    x = Conv2D(filters=128, kernel_size=(1,1),padding='same', activation='relu', name = 'm_pc_3')(x)
    x = DepthwiseConv2D(kernel_size=(3,3), strides=(2,2),padding='same', activation='relu', name = 'm_dc_4')(x)
    x = Conv2D(filters=128, kernel_size=(1,1),padding='same', activation='relu', name = 'm_pc_4')(x)
    x = GlobalAveragePooling2D(name='m_gap')(x)
    x = BatchNormalization(name='m_bn_1')(x)
    x = Dense(128, activation='relu', name='m_fc1')(x)
    x = BatchNormalization(name='m_bc_2')(x)
    x = Dense(k, activation='softmax', name='m_output')(x)
    model = Model(inputs, x)
    return model
```

通过`Summary()`函数我们可以得到每个网络的每层的参数数量，见图4，左侧是普通卷积，右侧是MobileNet v1。

![](/assets/MobileNet_4.png)

普通卷积的参数总量为259,082，去除未改造的部分剩余的参数数量为239,936。v1的参数总量为48,330去掉未改造的部分剩余参数29,184个。两个的比值为$$\frac{239,936}{29,184} \approx 8.22$$，符合我们之前的推算。

接着我们在MNIST上跑一下实验，我们在CPU（Intel i7）和GPU（Nvidia 1080Ti）两个环境下运行一下，得到的收敛曲线如图5。在都训练10个epoch的情况下，我们发现MobileNet v1的结果要略差于传统卷积，这点完全可以理解，毕竟MobileNet v1的参数更少。

![](/assets/MobileNet_5.png)

对比单个Epcoh的训练时间，我们发现了一个奇怪的现象，在CPU上，v1的训练时间约70秒，传统卷积训练时间为140秒，这和我们的直觉是相同的。但是在GPU环境下，传统卷积和v1的训练时间分别为40秒和50秒，v1在GPU上反而更慢了，这是什么原因呢？

问题在于cudnn对传统卷积的并行支持比较完善，而在cudnn7之前的版本并不支持depthwise卷积，现在虽然支持了，其并行性并没有做优化，依旧采用循环的形式遍历每个通道，因此在GPU环境下MobileNet v1反而要慢于传统卷积。所以说，是**开源工具慢，并不是MobileNet v1的算法慢**。

最后，论文中给出了两个超参数$$\alpha$$和$$\rho$$分别用于控制每层的Feature Map的数量以及输入图像的尺寸，由于并没有涉及很多特有知识，这里不过多介绍。

## 2. MobileNet v2 详解

在MobileNet v2中，作者将v1中加入了残差网络，同时分析了v1的几个缺点并针对性的做了改进。v2的改进策略非常简单，但是在编写论文时，缺点分析的时候涉及了流行学习等内容，将优化过程弄得非常难懂。我们在这里简单总结一下v2中给出的问题分析，希望能对论文的阅读有所帮助，对v2的motivation感兴趣的同学推荐阅读论文。

当我们单独去看Feature Map的每个通道的像素的值的时候，其实这些值代表的特征可以映射到一个低维子空间的一个流形区域上。在进行完卷积操作之后往往会接一层激活函数来增加特征的非线性性，一个最常见的激活函数便是ReLU。根据我们在[残差网络](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)中介绍的数据处理不等式\(DPI\)，ReLU一定会带来信息损耗，而且这种损耗是没有办法恢复的，ReLU的信息损耗是当通道数非常少的时候更为明显。为什么这么说呢？我们看图6中这个例子，其输入是一个表示流形数据的矩阵，和卷机操作类似，他会经过$$n$$个ReLU的操作得到$$n$$个通道的Feature Map，然后我们试图通过这$$n$$个Feature Map还原输入数据，还原的越像说明信息损耗的越少。从图6中我们可以看出，当$$n$$的值比较小时，ReLU的信息损耗非常严重，当时当$$n$$的值比较大的时候，输入流形就能还原的很好了。

![](/assets/MobileNet_6.png)

根据对上面提到的信息损耗问题分析，我们可以有两种解决方案：

1. 既然是ReLU导致的信息损耗，那么我们就将ReLU替换成线性激活函数；
2. 如果比较多的通道数能减少信息损耗，那么我们就使用更多的通道。

### 2.1 Linear Bottlenecks

我们当然不能把ReLU全部换成线性激活函数，不然网络将会退化为单层神经网络，一个折中方案是在输出Feature Map的通道数较少的时候也就是bottleneck部分使用线性激活函数，其它时候使用ReLU。代码片段如下：

```py
def _bottleneck(inputs, nb_filters, t):
    x = Conv2D(filters=nb_filters * t, kernel_size=(1,1), padding='same')(inputs)
    x = Activation(relu6)(x)
    x = DepthwiseConv2D(kernel_size=(3,3), padding='same')(x)
    x = Activation(relu6)(x)
    x = Conv2D(filters=nb_filters, kernel_size=(1,1), padding='same')(x)
    # do not use activation function
    if not K.get_variable_shape(inputs)[3] == nb_filters:
        inputs = Conv2D(filters=nb_filters, kernel_size=(1,1), padding='same')(inputs)
    outputs = add([x, inputs])
    return outputs
```

这里使用了MobileNet中介绍的ReLU6激活函数，它是对ReLU在6上的截断，数学形式为：


$$
ReLU(6) = min(max(0,x),6)
$$


图7便是结合了残差网络和线性激活函数的MobileNet v2的一个block，最右侧是v1。

![](/assets/MobileNet_7.png)

### 2.2 Inverted Residual

当激活函数使用ReLU时，我们可以通过增加通道数来减少信息的损耗，使用参数$$t$$来控制，该层的通道数是输入Feature Map的$$t$$倍。传统的残差块的$$t$$一般取小于1的小数，常见的取值为0.1，而在v2中这个值一般是介于$$5-10$$之间的数，在作者的实验中，$$t=6$$。考虑到残差网络和v2的$$t$$的不同取值范围，他们分别形成了锥子形（两头小中间大）和沙漏形（两头大中间小）的结构，如图8所示，其中斜线Feature Map表示使用的是线性激活函数。

![](/assets/MobileNet_8.png)

### 2.3 MobileNet v2

综上我们可以得到MobileNet v2的一个block的详细参数，如图9所示，其中$$s$$代表步长：

![](/assets/MobileNet_9.png)

MobileNet v2的实现可以通过堆叠bottleneck的形式实现，如下面代码片段

```py
def MobileNetV2_relu(input_shape, k):
    inputs = Input(shape = input_shape)
    x = Conv2D(filters=32, kernel_size=(3,3), padding='same')(inputs)
    x = _bottleneck_relu(x, 8, 6)
    x = MaxPooling2D((2,2))(x)
    x = _bottleneck_relu(x, 16, 6)
    x = _bottleneck_relu(x, 16, 6)
    x = MaxPooling2D((2,2))(x)
    x = _bottleneck_relu(x, 32, 6)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(k, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model
```

## 3. 总结

在这篇文章中，我们介绍了两个版本的MobileNet，它们和传统卷积的对比如图10。

![](/assets/MobileNet_10.png)

如图\(b\)所示，MobileNet v1最主要的贡献是使用了Depthwise Separable Convolution，它又可以拆分成Depthwise卷积和Pointwise卷积。MobileNet v2主要是将残差网络和Depthwise Separable卷积进行了结合。通过分析单通道的流形特征对残差块进行了改进，包括对中间层的扩展以及bottleneck层的线性激活。





## Reference

\[1\] Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications\[J\]. arXiv preprint arXiv:1704.04861, 2017.

\[2\] Sandler M, Howard A, Zhu M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4510-4520.

\[3\] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

