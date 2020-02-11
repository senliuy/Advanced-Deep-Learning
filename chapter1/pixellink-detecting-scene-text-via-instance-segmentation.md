# PixelLink: Detecting Scene Text via Instance Segmentation

## 前言

在前面的文章中，我们介绍了文本框回归的算法DeepText, CTPN以及RRPN；也介绍了以及实例分割的HMCP，在这里我们介绍一下另外一个基于实例分割的文字检测算法：PixelLink。根据PixelLink的算法名字我们也可以推测到，它有两个重点，一个是Pixel（像素），一个是Link（像素点之间的连接），这两个重点也是构成PixelLink的网络的输出层和损失函数优化的目标值。下面我们来看一下PixelLink的详细内容。

## 1. 网络架构

![](/assets/PixelLink_1.png)

