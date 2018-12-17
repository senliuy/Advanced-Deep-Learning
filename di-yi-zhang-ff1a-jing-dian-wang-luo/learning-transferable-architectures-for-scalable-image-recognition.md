# Learning Transferable Architectures for Scalable Image Recognition

## 前言

在[NAS](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/neural-architecture-search-with-reinforecement-learning.html)\[2\]一文中我们介绍了如何使用强化学习学习一个完整的CNN网络或是一个独立的RNN单元，这种dataset interest的网络的效果也是目前最优的。但是NAS提出的网络的计算代价是相当昂贵的，仅仅在CIFAR-10上学习一个网络就需要500台GPU运行28天才能找到最优结构。这使得NAS很难迁移到大数据集上，更不要提ImageNet这样几百G的数据规模了。而在目前的行内规则上，如果不能在ImageNet上取得令人信服的结果，你的网络结构很难令人信服的。

为了将NAS迁移到大数据集乃至ImageNet上，这篇文章提出了在小数据（CIFAR-10）上学习一个网络块（block），然后通过堆叠更多的这些网络块的形式将网络迁移到更复杂，尺寸更大的数据集上面。因此这篇文章的最大贡献便是介绍了如何使用强化学习学习这些网络块。作者将用于ImageNet的NAS简称为NASNet，文本依旧采用NASNet的简称来称呼这个算法。实验数据也证明了NASNet的有效性，其在ImageNet的top-1精度和top-5精度均取得了当时最优的效果。

阅读本文前，强烈建议移步到我的《[Neural Architecture Search with Reinforecement Learning](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/neural-architecture-search-with-reinforecement-learning.html)》介绍文章中，因为本文并不会涉及强化学习部分，只会介绍控制器是如何学习一个NASNet网络块的。

## 1. NASNet详解

在NASNet中，完整的网络的结构还是需要手动设计的，NASNet学习的是完整网络中被堆叠、被重复使用的网络块。为了便于将网络迁移到不同的数据集上，我们需要学习两种类型的网络块：（1）_Normal Cell_：输出Feature Map和输入Feature Map的尺寸相同；（2）_Reduction Cell_：输出Feature Map对输入Feature Map进行了一次降采样，在Reduction Cell中，对使用Input Feature作为输入的操作（卷积或者池化）会默认步长为2。

NASNet的控制器的结构如图1所示

![](/assets/NASNet_1.png)

## Reference

\[1\] Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition\[J\]. arXiv preprint arXiv:1707.07012, 2017, 2\(6\).

\[2\] Zoph B, Le Q V. Neural architecture search with reinforcement learning\[J\]. arXiv preprint arXiv:1611.01578, 2016.

