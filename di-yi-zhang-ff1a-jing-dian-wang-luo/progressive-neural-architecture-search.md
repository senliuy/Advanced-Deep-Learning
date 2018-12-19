# Progressive Neural Architecture Search

tags: NAS, NASNet, PNASNet

## 前言

在[NAS]()[2]和[NASNet]()[3]中我们介绍了如何使用强化学习训练卷积网络的超参。NAS是该系列的第一篇，提出了使用强化学习训练一个控制器（RNN），该控制器的输出是卷积网络的超参，可以生成一个完整的卷积网络。NASNet提出学习网络的一个单元比直接整个网络效率更高且更容易迁移到其它数据集，并在ImageNet上取得了当时最优的效果。

本文是约翰霍普金斯在读博士刘晨曦在Google实习的一篇文章，基于NASNet提出了PNASNet，其训练时间降为NASNet的1/8并且取得了目前在ImageNet上最优的效果。其主要的优化策略为：

1. 更小的搜索空间；
2. Sequential model-based optimization(SMBO)：一种启发式搜索的策略，训练的模型从简单到复杂，从剪枝的空间中进行搜索；
4. 代理函数：使用代理函数预测模型的精度，省去了耗时的训练过程。

在阅读本文之前，确保你已经读懂了[NAS]()和[NASNet]()两篇文章。

## 1. PNASNet详解

### 1.1 更小的搜索空间

回顾NASNet的控制器策略，它是一个有$$2\times B \times 5$$个输出的LSTM，其中2表示分别学习Normal Cell和Reduction Cell。$$B$$表示每个网络单元有$$B$$个网络块。$$5$$表示网络块有5个需要学习的超参，记做$$(I_1, I_2, O_1, O_2, C)$$。$$I_1, I_2 \in \mathcal{I}_b$$用于预测网络块两个隐层状态的输入（Input），它会从之前一个，之前两个，或者已经计算的网络块中选择一个。$$O_1, O_2 \in \mathcal{O}$$用于预测对两个隐层状态的输入的操作（Operation，共有13个，具体见NASNet。$$C\in \mathcal{C}$$表示$$O_1, O_2$$的合并方式，有单位加和合并两种操作。因此它的搜索空间的大小为：

$$
(2^2\times13^2 \times 3^2\times13^2 \times4^2\times13^2 \times5^2\times13^2 \times6^2\times13^2 \times 2)^2 \approx 2.0\times 10^{34}
$$

PNASNet的控制器的运作方式和NASNet类似，但也有几点不同。

**只有Normal Cell**：PNASNet只学习了Normal Cell，是否进行降采样用户自己设置。当使用降采样时，它使用和Normal Cell完全相同的架构，只是要把Feature Map的数量乘2。这种操作使控制器的输出节点数变为$$B \times 5$$。

**更小的$$\mathcal{O}$$**：在观察NASNet的实验结果是，我们发现有5个操作是从未被使用过的，因此我们将它们从搜索空间中删去，保留的操作剩下了8个：

* 直接映射
* $$1\times1$$卷积；
* $$3\times3$$深度可分离卷积；
* $$3\times3$$空洞卷积；
* $$3\times3$$平均池化；
* $$3\times3$$最大池化；
* $$5\times5$$深度可分离卷积；
* $$7\times7$$深度可分离卷积；
* $$1\times7$$卷积 + $$7\times1$$卷积；

**合并$$\mathcal{C}$$**：通过观察NASNet的实验结果，作者发现拼接操作也从未被使用，因此我们也可以将这种情况从搜索空间中删掉。

因此PNASNet的搜索空间的大小是：

$$
2^2\times8^2 \times 3^2\times8^2 \times4^2\times8^2 \times5^2\times8^2 \times6^2\times8^2 \approx 5.6\times 10^{14}
$$

排除掉两个隐层状态的对称的情况后，NASNet的搜索空间的大小为$$10^{28}$$，PNASNet的搜索空间为$$10^{12}$$。这两个值的具体计算比较复杂，且和本文主要要讲解的内容关系不大，感兴趣的读者自行推算。

### 1.2 SMBO

尽管已经将优化搜索空间优化到了$$10^{12}$$的数量级，但是这个规模依然十分庞大，在其中进行搜索依旧非常耗时。

## Reference

[1] Liu C, Zoph B, Shlens J, et al. Progressive neural architecture search[J]. arXiv preprint arXiv:1712.00559, 2017.

[2] Zoph B, Le Q V. Neural architecture search with reinforcement learning[J]. arXiv preprint arXiv:1611.01578, 2016.

[3] Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition[J]. arXiv preprint arXiv:1707.07012, 2017, 2(6).