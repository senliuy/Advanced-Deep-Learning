# EAST: An Efficient and Accurate Scene Text Detector

## 前言

作为同一个团队出品的两个产品，EAST的设计和[HMCP](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/east-an-efficient-and-accurate-scene-text-detector.html)如出一辙，他们最重要的思想都是将文字检测任务转换成语义分割任务。该方法对比之前文字检测的算法的优点是将检测过程的多个阶段合成一个，不仅有利于端到端的进行调参，而且检测速度也更快。EAST的几个核心改进如下：

* 1. EAST提供了两种形式的损失函数，本别基于RBOX和基于QUAD；
* 1. 位置掩码不再是0或者1，而是包含了该点在文本区域中的位置信息；
* 1. 提出了效率跟高的Locality-Aware NMS。

EAST已经开源，所以我们根据[源码](https://github.com/argman/EAST)分析一下这篇文章。

## 1. EAST

### 1.1 骨干网络

论文中EAST使用了PVANet的结构，实际上你可以自由选择网络的类型，例如源码中使用的是残差网络，在这里我们遵循论文中使用的PAVNet来进行说明。EAST的网络结构如图1所示：

###### 图1：EAST算法流程图

![](/assets/EAST_1.png)

左边橙色部分是PVANet的主干网络，用于特征提取。该网络有5个block，每个block执行完会进行一次降采样，Feature Map的尺寸变成原来的$$1/2$$。对于一张$$224\times224$$的图片，Feature Map的边长依次是$$224\rightarrow 112\rightarrow 56 \rightarrow 28 \rightarrow 14 \rightarrow 7$$。

中间绿色部分是特征合并分支，该分支从最后面的$$7\times7$$的Feature Map开始逐层向上上采样及合并。如图中绿色虚线部分所示，$$f_1$$是一个尺寸为$$7\times7$$的Feature Map，经过双线性插值上采样之后尺寸变为$$14\times14$$。这和$$f_2$$的尺寸是相同的，通过concatnate操作合并到一起，经过1层$$1\times1$$卷积核1层$$3\times3$$的same卷积得到尺寸为$$14\times14$$的Feature Map $$h_2$$。其中$$1\times1$$卷积用于降维，目的是降低网络复杂度。



## Reference

\[1\] Zhou X, Yao C, Wen H, et al. EAST: an efficient and accurate scene text detector\[C\]//Proc. CVPR. 2017: 2642-2651.

\[2\] Yao C, Bai X, Sang N, et al. Scene text detection via holistic, multi-channel prediction\[J\]. arXiv preprint arXiv:1606.09002, 2016.

