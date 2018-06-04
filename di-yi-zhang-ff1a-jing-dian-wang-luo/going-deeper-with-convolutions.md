# Going Deeper With Convolutions

## 前言

2012年之后，卷积网络的研究分成了两大流派，并且两个流派都在2014年有重要的研究成果发表。第一个流派是增加卷积网络的深度，经典的网络有ImageNet 2013年冠军ZF-net\[1\]以及我们在上篇文章中介绍的VGG系列\[2\]。另外一个流派是增加网络的宽度，或者说是增加网络的复杂度，典型的网络有可以拟合任意凸函数的Maxout Networks \[3\]，可以拟合任意函数的Network in Network \(NIN\)\[4\]，以及本文要解析的基于Inception的GoogLeNet\[5\]。为了能更透彻的了解GoogLeNet的思想，我们首先需要了解Maxout和NIN两种结构。

## 1. 背景知识

### 1.1 Maxout Networks

在之前介绍的AlexNet中，引入了Dropout \[6\]来减轻模型的过拟合的问题。Dropout可以看做是一种集成模型的思想，在每个step中，会将网络的隐层节点以概率p置0。Dropout和传统的bagging方法主要有以下两个方面不同：

1. Dropout的每个子模型的权值是共享的；
2. 在训练的每个step中，Dropout每次会使用不同的样本子集训练不同的子网络。

这样在每个step中都会有不同的节点参与训练，减轻了节点之间的耦合性。在测试时，使用的是整个网络的所有节点，只是节点的输出值要乘以Dropout的概率p。

作者认为，与其像Dropout这种毫无选择的平均，我们不如有条件的选择更合适的节点来生成网络。

## Reference

\[1\] M. D. Zeiler and R. Fergus. Visualizing and understanding convolutional neural networks. In ECCV, 2014

\[2\] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition\[J\]. arXiv preprint arXiv:1409.1556, 2014.

\[3\] Goodfellow I J, Warde-Farley D, Mirza M, et al. Maxout networks\[J\]. arXiv preprint arXiv:1302.4389, 2013.

\[4\] M. Lin, Q. Chen, and S. Yan. Network in network. CoRR, abs/1312.4400, 2013.

\[5\] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

\[6\] Hinton G E, Srivastava N, Krizhevsky A, et al. Improving neural networks by preventing co-adaptation of feature detectors\[J\]. arXiv preprint arXiv:1207.0580, 2012.

