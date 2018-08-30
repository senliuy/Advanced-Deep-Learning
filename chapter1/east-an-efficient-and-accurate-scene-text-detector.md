# EAST: An Efficient and Accurate Scene Text Detector

## 前言

作为同一个团队出品的两个产品，EAST的设计和[HMCP](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/east-an-efficient-and-accurate-scene-text-detector.html)如出一辙，他们最重要的思想都是将文字检测任务转换成语义分割任务。该方法对比之前文字检测的算法的优点是将检测过程的多个阶段合成一个，不仅有利于端到端的进行调参，而且检测速度也更快。EAST的几个核心改进如下：

* 1. EAST提供了两种形式的损失函数，本别基于RBOX和基于QUAD；
* 2. 位置掩码不再是0或者1，而是包含了该点在文本区域中的位置信息；
* 3. 提出了效率跟高的Locality-Aware NMS。

EAST已经开源，所以我们根据[源码](https://github.com/argman/EAST)分析一下这篇文章。

## 1. EAST骨干网络

论文中EAST使用了PVANet的结构，实际上你可以自由选择网络的类型，例如源码中使用的是残差网络，在这里我们遵循论文中使用的PAVNet来进行说明。EAST的网络结构如图1所示：

###### 图1：EAST算法流程图

![](/assets/EAST_1.png)

## Reference

\[1\] Zhou X, Yao C, Wen H, et al. EAST: an efficient and accurate scene text detector[C]//Proc. CVPR. 2017: 2642-2651.

\[2\] Yao C, Bai X, Sang N, et al. Scene text detection via holistic, multi-channel prediction\[J\]. arXiv preprint arXiv:1606.09002, 2016.



