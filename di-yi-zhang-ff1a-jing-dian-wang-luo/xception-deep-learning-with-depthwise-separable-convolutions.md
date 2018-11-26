# Xception: Deep Learning with Depthwise Separable Convolutions

tags: Inception, Xception

## 前言

深度可分离卷积（Depthwise Separable Convolution）率先是由 Laurent Sifre载气博士论文《Rigid-Motion Scattering For Image Classification》\[2\]中提出。经典的[MobileNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/mobilenetxiang-jie.html) \[3\]系列算法便是采用深度可分离卷积作为其核心结构。

这篇文章主要从[Inception](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/going-deeper-with-convolutions.html) \[4\]的角度出发，探讨了Inception和深度可分离卷积的关系，从一个全新的角度解释了深度可分离卷积。再结合stoa的[残差网络](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)\[5\]，一个新的架构Xception应运而生。Xception取义自Extreme Inception，即Xception是一种极端的Inception，下面我们来看看它是怎样的一种极端法。

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


更多时候我们希望两组卷积的输出Feature Map相同，这里我们将Inception的$$1\times1$$卷积的通道数设为$$k_2$$，即参数数量为


$$
m\times k_2 + k_2\times 3\times 3
$$


它的参数数量是普通卷积的$$\frac{1}{k_1}$$，我们把这种形式的Inception叫做Extreme Inception，如图3所示。

![](/assets/Xception_3.png)

在搭建GoogLeNet网络时，我们一般采用堆叠Inception的形式，同理在搭建由Extreme Inception构成的网络的时候也是采用堆叠的方式，论文中将这种形式的网络结构叫做Xception。

如果你看过深度可分离卷积的话你就会发现它和Xception几乎是等价的，区别之一就是先计算Pointwise卷积和先计算Depthwise的卷积的区别。

在[MobileNet v2](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/mobilenetxiang-jie.html)\[6\]中，我们指出bottleneck的最后一层$$1\times1$$卷积核为线性激活时能够更有助于减少信息损耗，这也就是Xception和深度可分离卷积（准确说是MobileNet v2）的第二个不同点。

结合残差结构，一个完整的模型见图4，其实现Keras官方已经[开源](https://github.com/keras-team/keras-applications/blob/master/keras_applications/xception.py)。

![](/assets/Xception_4.png)

上图中要注意的几点：

1. Keras的```SeparalbeConv```函数是由$$3\times3$$的depthwise卷积和$$1\times1$$的pointwise卷积组成，因此可用于升维和降维；
2. 图5中的$$\oplus$$是add操作，即两个Feature Map进行单位加。

## 3. 总结

Xception的结构和MobileNet非常像，两个算法的提出时间近似，不存在谁抄袭谁的问题。他们从不同的角度揭示了深度可分离卷积的强大作用，MobileNet的思路是通过将普通$$3\times3$$卷积拆分的形式来减少参数数量，而Xception是通过对Inception的充分解耦来完成的。

## Reference

\[1\] Chollet F. Xception: Deep learning with depthwise separable convolutions\[J\]. arXiv preprint, 2017: 1610.02357.

\[2\] L. Sifre. Rigid-motion scattering for image classification. PhD thesis, Ph. D. thesis, 2014. 1, 3

\[3\] Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications\[J\]. arXiv preprint arXiv:1704.04861, 2017.

\[4\] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

\[5\] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

\[6\] Sandler M, Howard A, Zhu M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4510-4520.

