# Xception: Deep Learning with Depthwise Separable Convolutions

## 前言

深度可分离卷积（Depthwise Separable Convolution）率先是由 Laurent Sifre载气博士论文《Rigid-Motion Scattering For Image Classification》\[2\]中提出。经典的[MobileNet]() \[3\]系列算法便是采用深度可分离卷积作为其核心结构。

这篇文章主要从[Inception]() \[4\]的角度出发，探讨了Inception和深度可分离卷积的关系，从一个全新的角度解释了深度可分离卷积。再结合stoa的[残差网络]()\[5\]，一个新的架构Xception应运而生。Xception取义自Extreme Inception，即Xception是一种极端的Inception，下面我们来看看它是怎样的一种极端法。

## 1. Inception回顾

Inception的核心思想是将channel分成若干个不同感受野大小的通道，除了能获得不同的感受野，Inception还能大幅的降低参数数量。我们看图1中一个简单版本的Inception模型

![](/assets/Xception_1.png)

对于一个输入的Feature Map，首先通过三组$$1\times1$$卷积得到三组Feature Map，它和先使用一组$$1\times1$$卷积得到Feature Map，再将这组Feature Map分成三组是完全等价的（图2）。假设图1中$$1\times1$$卷积核的个数都是$$k_1$$，$$3\times3$$的卷积核的个数都是$$k_2$$，输入Feature Map的通道数为$$m$$，那么这个简单版本的参数个数为


$$
m\times k_1 + 3\times 3\times 3 \times \frac{k_1}{3} \times \frac{k_2}{3} = m\times k_1+ 3\times k_1 \times k_2
$$
![](/assets/Xception_2.png)

对比相同通道数，但是没有分组的普通卷积，普通卷积的参数数量为：


$$
m\times k_1 + 3\times3\times k_1 \times k_2
$$


参数数量约为Inception的三倍。

## 2. Xception

如果Inception是将$$3\times3$$卷积分成3组，那么考虑一种极度的情况，我们如果将Inception的$$1\times1$$得到的$$k_1$$个通道的Feature Map完全分开呢？也就是使用$$k_1$$个不同的卷积分别在每个通道上进行卷积，它的参数数量是：


$$
m\times k_1 + k_1\times 3\times 3
$$


它的参数数量是普通卷积的$$\frac{1}{k_2}$$，论文中将这种极端（Extreme）的Inception命名为Xception，如图3所示。

![](/assets/Xception_3.png)

在搭建GoogLeNet网络时，我们一般采用堆叠Inception的形式，

## Reference

\[1\] Chollet F. Xception: Deep learning with depthwise separable convolutions\[J\]. arXiv preprint, 2017: 1610.02357.

\[2\] L. Sifre. Rigid-motion scattering for image classification. PhD thesis, Ph. D. thesis, 2014. 1, 3

\[3\] Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications\[J\]. arXiv preprint arXiv:1704.04861, 2017.

\[4\] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

\[5\] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

