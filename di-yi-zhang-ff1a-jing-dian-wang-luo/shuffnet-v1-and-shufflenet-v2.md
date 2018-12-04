# ShuffNet v1 and ShuffleNet v2

tags: ShuffNet v1, ShuffleNet v2

## 前言

在[ResNeXt](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/aggregated-residual-transformations-for-deep-neural-networks.html)\[3\]的文章中，分组卷积作为传统卷积核深度可分离卷积的一种折中方案被采用。这时大量的对于整个Feature Map的Pointwise卷积成为了ResNeXt的性能瓶颈。一种更高效的策略是在组内进行Pointwise卷积，但是这种组内Pointwise卷积的形式不利于通道之间的信息流通，为了解决这个问题，ShuffleNet v1中提出了通道洗牌（channel shuffle）操作。

在ShuffleNet v2的文章中作者指出现在普遍采用的FLOPs评估模型性能是非常不合理的，因为一批样本的训练时间除了看FLOPs，还有很多过程需要消耗时间，例如文件IO，内存读取，GPU执行效率等等。作者从内存消耗成本，GPU并行性两个方向分析了模型可能带来的非FLOPs的行动损耗，进而设计了更加高效的ShuffleNet v2。ShuffleNet v2的架构和[DenseNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/densely-connected-convolutional-networks.html)\[4\]有异曲同工之妙，而且其速度和精度都要优于DenseNet。

## 1. ShuffleNet v1

### 1.1 Channel Shuffle

通道洗牌是介于整个通道的Pointwise卷积和组内Pointwise卷积的一种折中方案，传统策略是在整个Feature Map上执行$$1\times1$$卷积。假设一个传统的深度可分离卷积由一个$$3\times3$$的Depthwise卷积和一个$$1\times1$$的Pointwise卷积组成。其中输入Feature Map的尺寸为$$h\times w \times c_1$$，输出Feature Map的尺寸为$$h \times w \times c_2$$，$$1\times1$$处的FLOPs为


$$
B = 9 \cdot h \cdot w + h \cdot w \cdot c_1 \cdot c_2
$$


一般情况下$$c_1 \cdot c_2$$是远大于9的，也就是说深度可分离卷积的性能瓶颈主要在Pointwise卷积上。

为了解决这个问题，ShuffleNet v1中提出了仅在分组内进行Pointwise卷积，对于一个分成了$$g$$个组的分组卷积，其FLOPs  
为：


$$
B = 9 \cdot h \cdot w + \frac{h \cdot w \cdot c_1 \cdot c_2}{g}
$$


从上面式子中我们可以看出组内Pointwise卷积可以非常有效的缓解性能瓶颈问题。然而这个策略的一个非常严重的问题是卷积直接的信息沟通不畅，网络趋近于一个由多个结构类似的网络构成的模型集成，精度大打折扣，如图1.\(a\)所示。

![](/assets/ShuffleNet_1.png)

为了解决通道之间的沟通问题，ShuffleNet v1提出了其最核心的操作：通道洗牌（Channel Shuffle）。假设分组Feature Map的尺寸为$$w\times h \times c_1$$，把$$c_1 = g\times n$$，其中$$g$$表示分组的组数。Channel Shuffle的操作细节如下：

1. 将Feature Map展开成$$g\times n\times w\times h$$的四维矩阵（为了简单理解，我们把$$w\times h$$降到一维，表示为s）；
2. 沿着尺寸为$$g\times n\times s$$的矩阵的$$g$$轴和$$n$$轴进行转置；
3. 将$$g$$轴和$$n$$轴进行平铺后得到洗牌之后的Feature Map；
4. 进行组内$$1\times1$$卷积。

shuffle的结果如图1.\(c\)所示，具体操作细节示意图见图2，Keras实现见代码片段1。

![](/assets/ShuffleNet_2.png)

```py
def channel_shuffle(x, groups):
    """
    Parameters
        x: Input tensor of with `channels_last` data format
        groups: int number of groups per channel
    Returns
        channel shuffled output tensor
    Examples
        Example for a 1D Array with 3 groups
        >>> d = np.array([0,1,2,3,4,5,6,7,8])
        >>> x = np.reshape(d, (3,3))
        >>> x = np.transpose(x, [1,0])
        >>> x = np.reshape(x, (9,))
        '[0 1 2 3 4 5 6 7 8] --> [0 3 6 1 4 7 2 5 8]'
    """
    height, width, in_channels = x.shape.as_list()[1:]
    channels_per_group = in_channels // groups
    x = K.reshape(x, [-1, height, width, groups, channels_per_group])
    x = K.permute_dimensions(x, (0, 1, 2, 4, 3))  # transpose
    x = K.reshape(x, [-1, height, width, in_channels])
    return x
```

从代码中我们也可以看出，channel shuffle的操作是步步可微分的，因此可以嵌入到卷积网络中。

### 1.2 ShuffleNet v1 单元

图3.\(a\)是一个普通的带有残差结构的深度可分离卷积，例如，[MobileNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/mobilenetxiang-jie.html)[5], [Xception](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/xception-deep-learning-with-depthwise-separable-convolutions.html)[6]。ShuffleNet v1的结构如图3.(b)，3.(c)。其中3.(b)不需要降采样，3.(c)是需要降采样的情况。

![](/assets/ShuffleNet_3.png)

3.(b)和3.(c)已经介绍了ShuffleNet v1全部的实现细节，我们仔细分析之：

1. 上下两个红色部分的$$1\times1$$卷积替换为$$1\times1$$的分组卷积，分组$$g$$一般不会很大，论文中的几个值分别是1，2，3，4，8。当$$g=1$$时，ShuffleNet v1退化为Xception。$$g$$的值确保能够被通道数整除，保证reshape操作的有效执行。

2. 在第一个$$1\times1$$卷积之后添加一个1.1节介绍的Channel Shuffle操作。

3. 如图3.(c)中需要降采样的情况，左侧shortcut部分使用的是步长为2的$$3\times3$$平均池化，右侧使用的是步长为2的$$3\times3$$的Depthwise卷积。

4. 去掉了$$3\times3$$卷积之后的ReLU激活，目的是为了减少ReLU激活造成的信息损耗，具体原因见[MobileNet v2](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/mobilenetxiang-jie.html)[7]。

5. 如果进行了降采样，为了保证参数数量不骤减，往往需要加倍通道数量。所以在3.(c)中使用的是拼接（Concat）操作用于加倍通道数，而3.(b)中则是一个单位加。

最后基于ShuffleNet v1 单元，我们计算一下ResNet，ResNeXt，ShuffleNet v1的FLOPs，即执行一个单元需要的计算量，假设输入Feature Map的尺寸为$$w\times h\times c$$，bottleneck的通道数为$$m$$。

1. ResNet：$$hw(2cm + 9m^2)$$
2. ResNeXt：$$hw(2cm + \frac{9m^2}{g})$$
3. ShuffleNet v1：$$hw(2cm/g + 9)$$

## Reference

\[1\] Zhang, X., Zhou, X., Lin, M., Sun, J.: Shufflenet: An extremely efficient convolu-  
tional neural network for mobile devices. arXiv preprint arXiv:1707.01083 \(2017\)

\[2\] Ma N, Zhang X, Zheng H T, et al. Shufflenet v2: Practical guidelines for efficient cnn architecture design\[J\]. arXiv preprint arXiv:1807.11164, 2018.

\[3\] Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks\[C\]//Computer Vision and Pattern Recognition \(CVPR\), 2017 IEEE Conference on. IEEE, 2017: 5987-5995.

\[4\] Huang G, Liu Z, Weinberger K Q, et al. Densely connected convolutional networks\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017, 1\(2\): 3.

[5] Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications\[J\]. arXiv preprint arXiv:1704.04861, 2017.

[6] Chollet F. Xception: Deep learning with depthwise separable convolutions\[J\]. arXiv preprint, 2017: 1610.02357.

[7] Sandler M, Howard A, Zhu M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4510-4520.


