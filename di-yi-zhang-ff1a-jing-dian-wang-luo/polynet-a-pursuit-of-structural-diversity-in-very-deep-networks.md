# PolyNet: A Pursuit of Structural Diversity in Very Deep Networks

## 前言

在[Inception v4](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/going-deeper-with-convolutions.html)\[2\]中，[Inception](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/going-deeper-with-convolutions.html)\[3\]和[残差网络](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)\[4\]首次得以共同使用，后面简称IR。这篇文章提出的PolyNet可以看做是IR的进一步扩展，它从多项式的角度推出了更加复杂且效果更好的混合模型，并通过实验得出了这些复杂模型的最优混合形式，命名为_Very Deep PolyNet_。

本文试图从**结构多样性**上说明PolyNet的提出动机，但还是没有摆脱通过堆积模型结构（Inception，ResNet）来得到更好效果的牢笼，模型创新性上有所欠缺。其主要贡献是虽然增加网络的深度和宽度能提升性能，但是其收益会很快变少，这时候如果从结构多样性的角度出发优化模型，带来的效益也许会由于增加深度带来的效益，为我们优化网络结构提供了一个新的方向。

## 2. PolyNet详解

### 2.1 结构多样性

当前增大网络表达能力的一个最常见的策略是通过增加网络深度，但是如图1所示，随着网络的深度增加，网络的收益变得越来越小。另一个模型优化的策略是增加网络的宽度，例如增加Feature Map的数量。但是增加网络的宽度是非常不经济的，因为每增加$$k$$个参数，其计算复杂度和显存占用都要增加$$k^2$$。

![](/assets/PolyNet_1.png)

因此作者效仿IR的思想，希望通过更复杂的block结构来获得比增加深度更大的效益，这种策略在真实场景中还是非常有用的，即如何在有限的硬件资源条件下最大化模型的精度。

### 2.2 多项式模型

本文是从多项式的角度推导block结构的。首先一个经典的残差block可以表示为：


$$
(I+F)\cdot \mathbf{x} = \mathbf{x} + F \cdot \mathbf{x}  := \mathbf{x} + F(\mathbf{x} )
$$


其中$$\mathbf{x}$$是输入，$$I$$是单位映射，在残差网络中$$F$$是两个连续的卷积操作。如果$$F$$是Inception的话，上式便是IR的表达式，如图2所示。

![](/assets/PolyNet_2.png)

## Reference

\[1\] Zhang X, Li Z, Loy C C, et al. Polynet: A pursuit of structural diversity in very deep networks\[C\]//Computer Vision and Pattern Recognition \(CVPR\), 2017 IEEE Conference on. IEEE, 2017: 3900-3908.

\[2\]Szegedy C, Ioffe S, Vanhoucke V, et al. Inception-v4, inception-resnet and the impact of residual connections on learning\[C\]//AAAI. 2017, 4: 12.

\[3\] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

\[4\] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

