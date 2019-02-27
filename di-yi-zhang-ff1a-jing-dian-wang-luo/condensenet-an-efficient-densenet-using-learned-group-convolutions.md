# CondenseNet: An Efficient DenseNet using Learned Group Convolutions

tags: DenseNet, CondenseNet

# 前言

CondenseNet{{"huang2018condensenet"|cite}}是黄高团队对其[DenseNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/densely-connected-convolutional-networks.html){{"huang2017densely"|cite}}的升级。DenseNet的密集连接其实是存在冗余的，其最大的影响便是影响网络的效率。首先，为了降低DenseNet的冗余问题，CondenseNet提出了在训练的过程中对不重要的权值进行剪枝，即学习一个稀疏的网络。但是测试的整个过程就是一个简单的卷积，因为网络已经在训练的时候优化完毕。其次，为了进一步提升效率，CondenseNet在$$1\times1$$卷积的时候使用了分组卷积，分组卷积在AlexNet中首先应用于双GPU架构，并在[ResNeXt](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/aggregated-residual-transformations-for-deep-neural-networks.html){{"xie2017aggregated"|cite}}中作为性能提升的策略首次被提出。最后，CondenseNet中指出临近的特征重用更重要，因此采用了指数增长的增长率（Growth Rate），并在DenseNet的block之间也添加了short-cut。

DenseNet，CondenseNet的训练和测试阶段的示意图如图1。其中的细节我们会在后面的部分详细解析。

<figure>
<img src="/assets/CondenseNet_1.png" alt="图1：CondensetNet 概览，(左)：DenseNet；(中)：CondenseNet训练；(右)：CondenseNet测试"/>
<figcaption>图1：CondensetNet 概览，(左)：DenseNet；(中)：CondenseNet训练；(右)：CondenseNet测试</figcaption>
</figure>

## 1. CondenseNet详解

### 1.1 分组卷积的问题

在[ShuffleNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/shuffnet-v1-and-shufflenet-v2.html){{"zhang2018shufflenet"|cite}}中我们指出分组卷积存在通道之间的信息沟通不畅以及特征多样性不足的问题。CondenseNet提出的解决策略是在训练的过程中让模型选择更好的分组方式，理论上每个通道的Feature Map是可以和所有Feature Map沟通到的。传统的沟通不畅的分组卷积自然不可能被学习到。图2是普通卷积核分组卷积的示意图。

<figure>
<img src="/assets/CondenseNet_2.png" alt="图2：(左)：普通卷积；(右)：分组卷积"/>
<figcaption>图2：(左)：普通卷积；(右)：分组卷积</figcaption>
</figure>

我们换一个角度来看分组卷积，它也可以别看做普通卷积的稀疏表示，只不过指着稀疏方式是由认为生硬的指定的。这种稀疏连接虽然高效，但是人为的毫无根据的指定那些连接重要，哪些连接需要被删除无疑非常不合理。CondenseNet指出的解决方案是使用训练数据学习卷积网络的稀疏表示，让识别精度决定哪些权值该被保留，这个过程叫做_learning group convolution_，即图1中间红色的'L-Conv'。

### 1.2 自学习分组卷积

如图3所示，自学习分组卷积（Learned Group Convolution）可以分成两个阶段：浓缩（Condensing）阶段和优化（Optimizing）阶段。其中浓缩阶段用于剪枝没用的特征，优化阶段用于优化剪枝之后的网络。

<figure>
<img src="/assets/CondenseNet_3.png" alt="图3：C=3时的CondenseNet的训练情况"/>
<figcaption>图3：C=3时的CondenseNet的训练情况</figcaption>
</figure>

在图3中分组数$$G=3$$。浓缩率$$C=3$$，即只保留原来1/3的特征。

浓缩率为$$C$$的CondenseNet会有$$C-1$$个浓缩阶段，它的第一个浓缩阶段（图3的最左侧）的初始化是普通的卷积网络，在训练该网络时使用了分组lasso正则项，这样学到的特征会呈现结构化稀疏分布，好处是在后面剪枝部分不会过分的影响精度。在每次浓缩阶段训练完成之后会有$$\frac{1}{C}$$的特征被剪枝掉。也就是经过浓缩阶段后，仅有$$\frac{1}{C}$$的特征被保留下来。

浓缩阶段之后是优化阶段，它会针对剪枝之后的网络单独做权值优化。CondenseNet用于优化阶段的总Epoch数和浓缩阶段是相同的。也就是说假设网络的总训练Epoch数是$$M$$，压缩率是$$C$$。它会有$$C-1$$个浓缩阶段，每个阶段的Epoch数均是$$\frac{M}{2(C-1)}$$。以及一个优化阶段，Epoch数为$$\frac{2}{M}$$，损失值，学习率以及Epoch分布情况见图4。

<figure>
<img src="/assets/CondenseNet_4.png" alt="图4：C=4时的CondenseNet的训练Epoch分布情况以及cosine学习率"/>
<figcaption>图4：C=4时的CondenseNet的训练Epoch分布情况以及cosine学习率</figcaption>
</figure>

图4中是基于CIFAR-10数据集，CondenseNet的压缩率$$C=4$$，所以有3个浓缩阶段。学习率是采用的是_cosine shape learning rate_{{"loshchilov2016sgdr"|cite}}。每次浓缩之后loss会有个明显的震动，最后一次loss震动的比较剧烈是因为一半的特征被剪枝掉。

### 1.3 剪枝准则

在1.2节中我们指出每个浓缩阶段中会有$$\frac{1}{C}$$的特征被剪枝掉，这里我们来分析如何确定哪些卷积核应该被剪掉。

CondenseNet的浓缩阶段一般发生在$$1\times1$$卷积部分，假设输入Feature Map的通道数是$$R$$，输出Feature Map的通道数是$$O$$。Feature Maps分成了$$G$$个组，图3中$$G=3$$。为了计算效率，我们希望剪枝之后每组的连接具有相同的稀疏模式，它们记做$$\{\mathbf{F}_1, \mathbf{F}_2, ..., \mathbf{F}_G\}$$。初始阶段，$$\mathbf{F}_g$$的连接数是$$\frac{O}{G}\times F$$，浓缩之后的连接数是$$\frac{O}{G}\times \frac{F}{C}$$。$$j \in (1,R)$$表示第输入Feature Map的索引，$$i \in (1, \frac{O}{G})$$表示组内输出Feature Map的索引。第$$j$$个滤波器关于这个组的重要性可以用权值的l1范数的和来表示：


$$
\sum_{i=1}^{\frac{O}{G}}|\mathbf{F}_{i,j}^g|
$$


我们一般可以认为l1范数和越大，该特征越重要，因为经过分组lasso正则化之后得到的是稀疏的特征，不重要的特征的值往往更接近0，因此可以被剪枝掉。在CondenseNet中我们剪枝掉的是$$\frac{O}{G}\times \frac{F}{C}$$个最小的特征，如图3中浓缩阶段2训练的便是浓缩阶段1剪枝$$\frac{1}{C}$$之后剩下的网络。优化阶段优化的是浓缩到只有原来的尺寸的$$\frac{1}{C}$$的网络。

CondenseNet的剪枝并不是直接将这个特征删除，而是通过掩码的形式将被剪枝的特征置0，因此在训练的过程中CondenseNet的时间并没有减少，反而会需要更多的显存用来保存掩码。

### 1.4 索引层

经过训练过程的剪枝之后我们得到了一个系数结构，如图3右侧所示。目前这种形式是不能用传统的分组卷积的形式计算的，如果使用训练过程中的掩码的形式则剪枝的意义就不复存在。

为了解决这个问题，在测试的时候CondenseNet引入了索引层（Index Layer），索引层的作用是将输入Feature Map重新整理以方便分组卷积的高效运行。举例：图3中，组1使用的输入Feature Maps是\(3,7,9,12\)，组2使用的Feature Maps是\(1,5,10,12\)，组3使用的Feature Maps是\(5,6,8,11\)，索引层的作用就是将输入Feature Map排列成\(3,7,9,12,1,5,10,12,5,6,8,11\)的形式，之后便可以直接连接标准的分组卷积，如图5所示。

<figure>
<img src="/assets/CondenseNet_5.png" alt="图5：CondenseNet的测试和Index层示意图"/>
<figcaption>图5：CondenseNet的测试和Index层示意图</figcaption>
</figure>

### 1.5 架构设计

在CondenseNet中作者对DenseNet做了两点改进：

**Growth rate的指数级增长**：增长率（Growth Rate）$$k$$是在[DenseNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/densely-connected-convolutional-networks.html)中提出的一个超参数，反应的是每个Dense Block中Feature Map通道数的增长速度，例如一个block中第1层的Feature Map的通道数是$$k_0$$，那么第$$i$$层的通道数即为$$k_0 + k\cdot(i-1)$$。

通过可视化DenseNet中特征重用的热力图，作者发现临近的Feature Map之间的特征重用更为有效，因此作者想通过增强临近节点之间的连接来增强模型的表现能力。为了实现这个动机，CondenseNet使用了指数级增长的增长率。按照上段给出的定义，第$$i$$层的通道数是$$k = 2^{i-1}k_0$$。也就是说越接近block输出层的地方保留的Feature Map越多，

**全密集连接**：在DenseNet中，block之间是没有shortcut的，CondenseNet在block之间也增加了shortcut，结合平均池化用于实现不同尺寸的Feature Map之间的拼接用以实现更强的特征重用，如图6所示。

<figure>
<img src="/assets/CondenseNet_6.png" alt="图6：CondenseNet block之间的全密集连接"/>
<figcaption>图6：CondenseNet block之间的全密集连接</figcaption>
</figure>

## 2. 总结

文章最大的创新点是通过模型训练得到目前轻量级网络中常见的分组卷积的分组形式，比ShuffleNet解决分组卷积通信问题的方法更加有效，并结合Index层实现了测试时候的高效运行。最后，结合对DenseNet的dense block的改进完成了CondenseNet的完整结构。

