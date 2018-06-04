# Going Deeper With Convolutions

## 前言

2012年之后，卷积网络的研究分成了两大流派，并且两个流派都在2014年有重要的研究成果发表。第一个流派是增加卷积网络的深度，经典的网络有ImageNet 2013年冠军ZF-net\[1\]以及我们在上篇文章中介绍的VGG系列\[2\]。另外一个流派是增加网络的宽度，或者说是增加网络的复杂度，典型的网络有可以拟合任意凸函数的Maxout Networks \[3\]，可以拟合任意函数的Network in Network \(NIN\)\[4\]，以及本文要解析的基于Inception的GoogLeNet\[5\]。为了能更透彻的了解GoogLeNet的思想，我们首先需要了解Maxout和NIN两种结构。

## 1. 背景知识

### 1.1 Maxout Networks

在之前介绍的AlexNet中，引入了Dropout \[6\]来减轻模型的过拟合的问题。Dropout可以看做是一种集成模型的思想，在模型训练的每个step，Dropout都会以概率p将某些

## Reference

\[1\] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional neural networks. In ECCV, 2014

\[2\] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition\[J\]. arXiv preprint arXiv:1409.1556, 2014.

\[3\] Goodfellow I J, Warde-Farley D, Mirza M, et al. Maxout networks\[J\]. arXiv preprint arXiv:1302.4389, 2013.

\[4\] M. Lin, Q. Chen, and S. Yan. Network in network. CoRR, abs/1312.4400, 2013.

\[5\] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

\[6\] 

