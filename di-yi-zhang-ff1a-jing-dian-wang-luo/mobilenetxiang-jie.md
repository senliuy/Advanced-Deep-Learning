# MobileNet v1 and MobileNet v2

## 前言

MobileNet[1]（这里叫做MobileNet v1，简称v1）中使用的Depthwise Separable Convolution是模型压缩的一个最为经典的策略，它是通过将跨通道的$$3\times3$$卷积换成单通道的$$3\times3$$卷积+跨通道的$$1\times1$$卷积来达到此目的的。

MobileNet v2 [2]是在v1的Depthwise Separable的基础上引入了[残差结构]()。并发现了ReLU的在通道数较少的Feature Map上有非常严重信息损失问题，由此引入了Linear Bottlenecks和Inverted Residual。

首先在这篇文章中我们会详细介绍两个版本的MobileNet，然后我们会介绍如何使用Keras实现这两个算法。

## 1. MobileNet v1详解

### 1.1 Depthwise Separable Convolution

## 2. MobileNet v2 详解

### 2.1 Linear Bottlenecks

### 2.2 Inverted Residual

## Reference

[1] Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications[J]. arXiv preprint arXiv:1704.04861, 2017.

[2] Sandler M, Howard A, Zhu M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4510-4520.
