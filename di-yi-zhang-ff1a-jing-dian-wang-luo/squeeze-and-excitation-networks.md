# Squeeze-and-Excitation Networks

## 前言

SENet的提出动机非常简单，传统的方法是将网络的Feature Map等权重的传到下一层，SENet的核心思想在于**建模通道之间的相互依赖关系，通过网络的全局损失函数自适应的重新矫正通道之间的特征相应强度**。

SENet由一些列SE block组成，一个SE block的过程分为Squeeze（压缩）和Excitation（激发）两个步骤。其中Squeeze通过在Feature Map层上执行Global Average Pooling得到当前Feature Map的全局压缩特征向量，Excitation通过两层全连接得到Feature Map中每个通道的权值，并将加权后的Feature Map作为下一层网络的输入。从上面的分析中我们可以看出SE block只依赖与当前的一组Feature Map，因此可以非常容易的嵌入到几乎现在所有的卷积网络中。论文中给出了在当时state-of-the-art的Inception\[2\]和残差网络\[3\]插入SE block后的实验结果，效果提升显著。

SENet虽然引入了更多的操作，但是其带来的性能下降尚在可以接受的范围之内，从GFLOPs，参数数量以及运行时间的实验结果上来看，SENet的损失并不是非常显著。

## 1. SENet详解

### 1.1. SE Block

一个SE Block的结构如图1所示

![](/assets/SENet_1.png)

网络的左半部分是一个传统的卷积变换，忽略掉这一部分并不会影响我们的SENet的理解。我们直接看一下后半部分，其中$$U$$是一个$$W\times H\times C$$的Feature Map，$$(W,H)$$是图像的尺寸，$$C$$是图像的通道数。

经过$$F_{sq}(\cdot)$$（Squeeze操作）后，图像变成了一个$$1\times1\times C$$的特征向量，特征向量的值由$$U$$确定。经过$$F_{ex}(\cdot,\mathbf{W})$$后，特征向量的维度没有变，但是向量值变成了新的值。这些值会通过和$$U$$的$$F_{scale}(\cdot,\cdot)$$得到加权后的$$\tilde{X}$$。$$\tilde{X}$$和$$U$$的维度是相同的。

### 1.2. Squeeze

Squeeze部分的作用是获得Feature Map $$U$$的每个通道的全局信息嵌入（特征向量）。在SE block中，这一步通过VGG中引入的Global Average Pooling（GAP）实现的。也就是通过求每个通道$$c, c\in\{1,C\}$$的Feature Map的平均值：

$$
z_c = \mathbf{F}_{sq}(\mathbf{u}_c) = \frac{1}{W\times H} \sum_{i=1}^W\sum_{j=1}^H u_c(i,j)
$$

通过GAP得到的特征值是全局的（虽然比较粗糙）。另外，$$z_c$$也可以通过其它方法得到，要求只有一个，得到的特征向量具有全局性。

### 1.3. Excitation

Excitation部分的作用是通过$$z_c$$学习$$C$$中每个通道的特征权值，要求有两点：

1. 要足够灵活，这样能保证学习到的权值比较具有价值；
2. 要足够简单，这样不至于添加SE blocks之后网络的训练速度大幅降低；
3. 通道之间的关系是non-exclusive的，也就是说学习到的特征能过激励重要的特征，抑制不重要的特征。

根据上面的要求，SE blocks使用了两层全连接构成的门机制（gate mechanism）。门控单元$$\mathbf{s}$$（即图1中$$1\times1\times C$$的特征向量）的计算法方式表示为：

$$
\mathbf{s} = \mathbf{F}_{ex}(\mathbf{z}, \mathbf{W}) = \sigma(g(\mathbf{z}, \mathbf{W})) = \sigma(g(\mathbf{W}_2 \delta(\mathbf{W}_1 \mathbf{z})))
$$

其中$$\delta$$表示ReLU激活函数，$$\sigma$$表示sigmoid激活函数。$$\mathbf{W}_1 \in \mathbb{R}^{\frac{C}{r}\times C}$$, $$\mathbf{W}_2 \in \mathbb{R}^{C\times\frac{C}{r}}$$分别是两个全连接层的权值矩阵。$$r$$则是中间层的隐层节点数，论文中指出这个值是16。

得到门控单元$$\mathbf{s}$$后，最后的输出$$\tilde{X}$$表示为$$\mathbf{s}$$和$$\mathbf{U}$$的向量积，即图1中的$$\mathbf{F}_{scale}(\cdot,\cdot)$$操作：

$$
\tilde{x}_c = \mathbf{F}_{scale}(\mathbf{u}_c,s_c) = s_c \cdot \mathbf{u}_c
$$

其中$$\tilde{x}_c$$是$$\tilde{X}$$的一个特征通道的一个Feature Map，$$s_c$$是门控单元$$\mathbf{s}$$（是个向量）中的一个标量值。

以上就是SE blocks算法的全部内容，SE blocks可以从两个角度理解：

1. SE blocks学习了每个Feature Map的动态先验；
2. SE blocks可以看做在Feature Map方向的Attention，因为注意力机制的本质也是学习一组权值。

### 1.4. SE-Inception
## Reference

\[1\] Hu J, Shen L, Sun G. Squeeze-and-excitation networks\[J\]. arXiv preprint arXiv:1709.01507, 2017, 7.

\[2\] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

\[3\] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

