# Xception: Deep Learning with Depthwise Separable Convolutions

## 前言

深度可分离卷积（Depthwise Separable Convolution）率先是由 Laurent Sifre载气博士论文《Rigid-Motion Scattering For Image Classification》[2]中提出。经典的[MobileNet]() [3]系列算法便是采用深度可分离卷积作为其核心结构。

这篇文章主要从[Inception]() [4]的角度出发，探讨了Inception和深度可分离卷积的关系，从一个全新的角度解释了深度可分离卷积。再结合stoa的[残差网络]()[5]，一个新的架构Xception应运而生。Xception取义自Extreme Inception，即Xception是一种极端的Inception，下面我们来看看它是怎样的一种极端法。

## Inception回顾




## Reference

[1] Chollet F. Xception: Deep learning with depthwise separable convolutions[J]. arXiv preprint, 2017: 1610.02357.

[2] L. Sifre. Rigid-motion scattering for image classification. PhD thesis, Ph. D. thesis, 2014. 1, 3

[3] Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications\[J\]. arXiv preprint arXiv:1704.04861, 2017.

[4] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

