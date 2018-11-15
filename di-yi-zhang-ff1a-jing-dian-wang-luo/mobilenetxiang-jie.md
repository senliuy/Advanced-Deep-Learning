# MobileNet v1 and MobileNet v2

## 前言

MobileNet\[1\]（这里叫做MobileNet v1，简称v1）中使用的Depthwise Separable Convolution是模型压缩的一个最为经典的策略，它是通过将跨通道的$$3\times3$$卷积换成单通道的$$3\times3$$卷积+跨通道的$$1\times1$$卷积来达到此目的的。

MobileNet v2 \[2\]是在v1的Depthwise Separable的基础上引入了[残差结构]()。并发现了ReLU的在通道数较少的Feature Map上有非常严重信息损失问题，由此引入了Linear Bottlenecks和Inverted Residual。

首先在这篇文章中我们会详细介绍两个版本的MobileNet，然后我们会介绍如何使用Keras实现这两个算法。

## 1. MobileNet v1详解

### 1.1 Depthwise Separable Convolution

传统的卷积网络是跨通道的，对于一个通道数为$$M$$的输入Feature Map，我们要得到通道数为$$N$$的输出Feature Map。普通卷积会使用$$N$$个不同的$$D_K \times D_K \times M$$以滑窗的形式遍历输入Feature Map，因此对于一个尺寸为$$D_K\times D_K$$的卷积的参数个数为$$D_K \times D_K \times M \times N$$。一个普通的卷积可以表示为：


$$
G_{k,l,n} = \sum_{i,j,m} \mathbf{K}_{i,j,m,n} \cdot \mathbf{K}_{k+i-1, l+j-1, m}
$$


它的一层网络的计算代价约为：


$$
D_k \times D_K \times M \times N \times D_W \times D_H
$$


其中$$(D_W, D_H)$$为Feature Map的尺寸。普通卷积如图1所示。

![](/assets/MobileNet_1.png)

v1中介绍的Depthwise Separable Convolution就是解决了传统卷积的参数数量和计算代价过于高昂的问题。

Depthwise Separable Convolution分成Depthwise Convolution和Pointwise Convolution。

### 1.2 Depthwise卷积

其中Depthwise卷积是指不跨通道的卷积，也就是说Feature Map的每个通道有一个独立的卷积核，并且这个卷积核作用且仅作用在这个通道之上，如图2所示。

![](/assets/MobileNet2.png)

从图2和图1的对比中我们可以看出，因为放弃了卷积时的跨通道。Depthwise卷积的参数数量仅为传统卷积的$$\frac{1}{N}$$。Depthwise Convolution的数学表达式为：

$$
\hat{G}_{k,l,m} = \sum_{i,j} \hat{K}_{i,j,n} \cdot F_{k+i-1, l+j-1, m}
$$

它的计算代价也是传统卷积的$$\frac{1}{N}$$为:

$$
D_k \times D_K \times M \times D_W \times D_H
$$

## 2. MobileNet v2 详解

### 2.1 Linear Bottlenecks

### 2.2 Inverted Residual

## Reference

\[1\] Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications\[J\]. arXiv preprint arXiv:1704.04861, 2017.

\[2\] Sandler M, Howard A, Zhu M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4510-4520.

