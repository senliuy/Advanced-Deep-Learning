# ShuffNet v1 and ShuffleNet v2

tags: ShuffNet v1, ShuffleNet v2

## 前言

在[ResNeXt](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/aggregated-residual-transformations-for-deep-neural-networks.html){{"xie2017aggregated"|cite}}的文章中，分组卷积作为传统卷积核深度可分离卷积的一种折中方案被采用。这时大量的对于整个Feature Map的Pointwise卷积成为了ResNeXt的性能瓶颈。一种更高效的策略是在组内进行Pointwise卷积，但是这种组内Pointwise卷积的形式不利于通道之间的信息流通，为了解决这个问题，ShuffleNet v1{{"zhang2018shufflenet"|cite}}中提出了通道洗牌（channel shuffle）操作。

在ShuffleNet v2的文章中作者指出现在普遍采用的FLOPs评估模型性能是非常不合理的，因为一批样本的训练时间除了看FLOPs，还有很多过程需要消耗时间，例如文件IO，内存读取，GPU执行效率等等。作者从内存消耗成本，GPU并行性两个方向分析了模型可能带来的非FLOPs的行动损耗，进而设计了更加高效的ShuffleNet v2{{"ma2018shufflenet"|cite}}。ShuffleNet v2的架构和[DenseNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/densely-connected-convolutional-networks.html){{"huang2017densely"|cite}}有异曲同工之妙，而且其速度和精度都要优于DenseNet。

## 1. ShuffleNet v1

### 1.1 Channel Shuffle

通道洗牌是介于整个通道的Pointwise卷积和组内Pointwise卷积的一种折中方案，传统策略是在整个Feature Map上执行$$1\times1$$卷积。假设一个传统的深度可分离卷积由一个$$3\times3$$的Depthwise卷积和一个$$1\times1$$的Pointwise卷积组成。其中输入Feature Map的尺寸为$$h\times w \times c_1$$，输出Feature Map的尺寸为$$h \times w \times c_2$$，$$1\times1$$处的FLOPs为


$$
F = 9 \cdot h \cdot w + h \cdot w \cdot c_1 \cdot c_2
$$


一般情况下$$c_1 \cdot c_2$$是远大于9的，也就是说深度可分离卷积的性能瓶颈主要在Pointwise卷积上。

为了解决这个问题，ShuffleNet v1中提出了仅在分组内进行Pointwise卷积，对于一个分成了$$g$$个组的分组卷积，其FLOPs  
为：


$$
F = 9 \cdot h \cdot w + \frac{h \cdot w \cdot c_1 \cdot c_2}{g}
$$


从上面式子中我们可以看出组内Pointwise卷积可以非常有效的缓解性能瓶颈问题。然而这个策略的一个非常严重的问题是卷积直接的信息沟通不畅，网络趋近于一个由多个结构类似的网络构成的模型集成，精度大打折扣，如图1.\(a\)所示。

<figure>
<img src="/assets/ShuffleNet_1.png" alt="图1：分组Pointwise卷积 vs 通道洗牌"/>
<figcaption>图1：分组Pointwise卷积 vs 通道洗牌</figcaption>
</figure>

为了解决通道之间的沟通问题，ShuffleNet v1提出了其最核心的操作：通道洗牌（Channel Shuffle）。假设分组Feature Map的尺寸为$$w\times h \times c_1$$，把$$c_1 = g\times n$$，其中$$g$$表示分组的组数。Channel Shuffle的操作细节如下：

1. 将Feature Map展开成$$g\times n\times w\times h$$的四维矩阵（为了简单理解，我们把$$w\times h$$降到一维，表示为s）；
2. 沿着尺寸为$$g\times n\times s$$的矩阵的$$g$$轴和$$n$$轴进行转置；
3. 将$$g$$轴和$$n$$轴进行平铺后得到洗牌之后的Feature Map；
4. 进行组内$$1\times1$$卷积。

shuffle的结果如图1.\(c\)所示，具体操作细节示意图见图2，Keras实现见代码片段1。

<figure>
<img src="/assets/ShuffleNet_2.png" alt="图2：通道洗牌过程详解"/>
<figcaption>图2：通道洗牌过程详解</figcaption>
</figure>


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

图3.\(a\)是一个普通的带有残差结构的深度可分离卷积，例如，[MobileNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/mobilenetxiang-jie.html){{"howard2017mobilenets"|cite}}, [Xception](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/xception-deep-learning-with-depthwise-separable-convolutions.html){{"chollet2017xception"|cite}}。ShuffleNet v1的结构如图3.(b)，3.(c)。其中3.(b)不需要降采样，3.(c)是需要降采样的情况。

<figure>
<img src="/assets/ShuffleNet_3.png" alt="图3：(a) MobileNet, (b) ShuffleNet v1，(c) ShuffleNet v1降采样情况"/>
<figcaption>图3：(a) MobileNet, (b) ShuffleNet v1，(c) ShuffleNet v1降采样情况</figcaption>
</figure>

3.(b)和3.(c)已经介绍了ShuffleNet v1全部的实现细节，我们仔细分析之：

1. 上下两个红色部分的$$1\times1$$卷积替换为$$1\times1$$的分组卷积，分组$$g$$一般不会很大，论文中的几个值分别是1，2，3，4，8。当$$g=1$$时，ShuffleNet v1退化为Xception。$$g$$的值确保能够被通道数整除，保证reshape操作的有效执行。

2. 在第一个$$1\times1$$卷积之后添加一个1.1节介绍的Channel Shuffle操作。

3. 如图3.(c)中需要降采样的情况，左侧shortcut部分使用的是步长为2的$$3\times3$$平均池化，右侧使用的是步长为2的$$3\times3$$的Depthwise卷积。

4. 去掉了$$3\times3$$卷积之后的ReLU激活，目的是为了减少ReLU激活造成的信息损耗，具体原因见[MobileNet v2](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/mobilenetxiang-jie.html){{"sandler2018mobilenetv2"|cite}}。

5. 如果进行了降采样，为了保证参数数量不骤减，往往需要加倍通道数量。所以在3.(c)中使用的是拼接（Concat）操作用于加倍通道数，而3.(b)中则是一个单位加。

最后基于ShuffleNet v1 单元，我们计算一下ResNet，ResNeXt，ShuffleNet v1的FLOPs，即执行一个单元需要的计算量。Channel Shuffle处的操作数非常少，这里可以忽略不计。假设输入Feature Map的尺寸为$$w\times h\times c$$，bottleneck的通道数为$$m$$。

1. ResNet：
$$
F_{\text{ResNet}} = hwcm +3\cdot3\cdot hwmm  + hwcm = hw(2cm + 9m^2)
$$
2. ResNeXt：
$$
F_{\text{ResNeXt}} = hwcm +3\cdot3\cdot hw\frac{m}{g}\frac{m}{g}\cdot g  + hwcm = hw(2cm + \frac{9m^2}{g})
$$
3. ShuffleNet v1：
$$
F_{\text{ShuffleNet v1}} = hw\frac{c}{g}\frac{m}{g}\cdot g + 3\cdot 3 h w m + hw\frac{c}{g}\frac{m}{g}\cdot g = hw(\frac{2cm}{g} + 9m)
$$

我们可以非常容易得到它们的FLOPs的关系：

$$
F_{\text{ResNet}} < F_{\text{ResNeXt}} < F_{\text{ShuffleNet v1}}
$$

### 1.3 ShuffleNet v1 网络

ShuffleNet v1完整网络的搭建可以通过堆叠ShuffleNet v1 单元的形式构成，这里不再赘述。具体细节请查看已经开源的ShuffleNet v1的源码。

## 2. ShuffleNet v2

### 2.1 模型性能的评估指标

在上面的文章中我们统一使用FLOPs作为评估一个模型的性能指标，但是在ShuffleNet v2的论文中作者指出这个指标是间接的，因为一个模型实际的运行时间除了要把计算操作算进去之外，还有例如内存读写，GPU并行性，文件IO等也应该考虑进去。最直接的方案还应该回归到最原始的策略，即直接在同一个硬件上观察每个模型的运行时间。如图4所示，在整个模型的计算周期中，FLOPs耗时仅占50%左右，如果我们能优化另外50%，我们就能够在不损失计算量的前提下进一步提高模型的效率。

在ShuffleNet v2中，作者从内存访问代价（Memory Access Cost，MAC）和GPU并行性的方向分析了网络应该怎么设计才能进一步减少运行时间，直接的提高模型的效率。

### 2.2 高效模型的设计准则

** G1）：当输入通道数和输出通道数相同时，MAC最小 **

假设一个卷积操作的输入Feature Map的尺寸是$$w\times h\times c_1$$，输出Feature Map的尺寸为$$w\times h\times c_2$$。卷积操作的FLOPs为$$F = hwc_1 c_2$$。在计算这个卷积的过程中，输入Feature Map占用的内存大小是$$hwc_1$$，输出Feature Map占用的内存是$$hwc_2$$，卷积核占用的内存是$$c_1 c_2$$。总计：

![](/assets/ShuffleNet_f1.png)

当$$c_1 = c_2$$时上式取等号。也就是说当FLOPs确定的时候，$$c_1 = c_2$$时模型的运行效率最高，因为此时的MAC最小。

** G2）：MAC与分组数量$$g$$成正比 **

在分组卷积中，FLOPs为$$F = hw\frac{c_1}{g} \frac{c_2}{g} g = \frac{hwc_1 c_2}{g} $$，其MAC的计算方式为：

![](/assets/ShuffleNet_f2.png)

根据G2，我们在设计网络时$$g$$的值不应过大。

** G3）：网络的分支数量降低并行能力 **

分支数量比较多的典型网络是Inception，NasNet等。作者证明这个一组准则是设计了一组对照试验：如图4所示，通过控制卷积的通道数来使5组对照试验的FLOPs相同，通过实验我们发现它们按效率从高到低排列依次是 (a) > (b) > (d) > (c) > (e)。

<figure>
<img src="/assets/ShuffleNet_4.png" alt="图4：网络分支对比试验样本示意图"/>
<figcaption>图4：网络分支对比试验样本示意图</figcaption>
</figure>


造成这种现象的原因是更多的分支需要更多的卷积核加载和同步操作。

** G4）：Element-wise操作是非常耗时的 **

我们在计算FLOPs时往往只考虑卷积中的乘法操作，但是一些Element-wise操作（例如ReLU激活，偏置，单位加等）往往被忽略掉。作者指出这些Element-wise操作看似数量很少，但它对模型的速度影响非常大。尤其是深度可分离卷积这种MAC/FLOPs比值较高的算法。图5中统计了ShuffleNet v1和MobileNet v2中各个操作在GPU和ARM上的消耗时间占比。

<figure>
<img src="/assets/ShuffleNet_5.png" alt="图5：模型训练时间拆分示意图"/>
<figcaption>图5：模型训练时间拆分示意图</figcaption>
</figure>


总结一下，在设计高性能网络时，我们要尽可能做到：

1. G1). 使用输入通道和输出通道相同的卷积操作；
2. G2). 谨慎使用分组卷积；
3. G3). 减少网络分支数；
4. G4). 减少element-wise操作。

例如在ShuffleNet v1中使用的分组卷积是违背G2的，而每个ShuffleNet v1单元使用了bottleneck结构是违背G1的。MobileNet v2中的大量分支是违背G3的，在Depthwise处使用ReLU6激活是违背G4的。

从它的对比实验中我们可以看到虽然ShuffleNet v2要比和它FLOPs数量近似的的模型的速度要快。

### 2.3 ShuffleNet v2结构

图6中，(a)，(b)是刚刚介绍的ShuffleNet v1，(c)，(d)是这里要介绍的ShuffleNet v2。

<figure>
<img src="/assets/ShuffleNet_6.png" alt="图6：(a) ShuffleNet v1 ，(b)ShuffleNet v1 降采样， (c)ShuffleNet v2，(d)ShuffleNet v2 降采样"/>
<figcaption>图6：(a) ShuffleNet v1 ，(b)ShuffleNet v1 降采样， (c)ShuffleNet v2，(d)ShuffleNet v2 降采样</figcaption>
</figure>

仔细观察(c)，(d)对网络的改进我们发现了以下几点：

1. 在(c)中ShuffleNet v2使用了一个通道分割（Channel Split）操作。这个操作非常简单，即将$$c$$个输入Feature分成$$c-c'$$和$$c'$$两组，一般情况下$$c' = \frac{c}{2}$$。这种设计是为了尽量控制分支数，为了满足G3。

2. 在分割之后的两个分支，左侧是一个直接映射，右侧是一个输入通道数和输出通道数均相同的深度可分离卷积，为了满足G1。

3. 在右侧的卷积中，$$1\times1$$卷积并没有使用分组卷积，为了满足G2。

4. 最后在合并的时候均是使用拼接操作，为了满足G4。

5. 在堆叠ShuffleNet v2的时候，通道拼接，通道洗牌和通道分割可以合并成1个element-wise操作，也是为了满足G4。

最后当需要降采样的时候我们通过不进行通道分割的方式达到通道数量的加倍，如图6.(d)，非常简单。

### 2.4 ShuffleNet v2和DenseNet

ShuffleNet v2能够得到非常高的精度是因为它和DenseNet有着思想上非常一致的结构：强壮的特征重用（Feature Reuse）。在DenseNet中，作者大量使用的拼接操作直接将上一层的Feature Map原汁原味的传到下一个乃至下几个模块。从6.(c)中我们也可以看处，左侧的直接映射和DenseNet的特征重用是非常相似的。

不同于DenseNet的整个Feature Map的直接映射，ShuffleNet v2只映射了一半。恰恰是这一点不同，是ShuffleNet v2有了和DenseNet的升级版CondenseNet{{"huang2018condensenet"|cite}}相同的思想。在CondenseNet中，作者通过可视化DenseNet的特征重用和Feature Map的距离关系发现**距离越近的Feature Map之间的特征重用越重要**。ShuffleNet v2中第$$i$$个和第$$i+j$$个Feature Map的重用特征的数量是$$(\frac{1}{2})^j c$$。也就是距离越远，重用的特征越少。

## 总结

截止本文截止，ShuffleNet算是将轻量级网络推上了新的巅峰，两个版本都有其独到的地方。

ShuffleNet v1中提出的通道洗牌（Channel Shuffle）操作非常具有创新点，其对于解决分组卷积中通道通信困难上非常简单高效。

ShuffleNet v2分析了模型性能更直接的指标：运行时间。根据对运行时间的拆分，通过数学证明或是实验证明或是理论分析等方法提出了设计高效模型的四条准则，并根据这四条准则设计了ShuffleNet v2。ShuffleNet v2中的通道分割也是创新点满满。通过仔细分析通道分割，我们发现了它和DenseNet有异曲同工之妙，在这里轻量模型和高精度模型交汇在了一起。

ShuffleNet v2的证明和实验以及最后网络结构非常精彩，整篇论文读完给人一种畅快淋漓的感觉，建议读者们读完本文后拿出论文通读一遍，你一定会收获很多。


