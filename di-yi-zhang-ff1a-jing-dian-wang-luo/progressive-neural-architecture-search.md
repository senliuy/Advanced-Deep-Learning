# Progressive Neural Architecture Search

tags: NAS, NASNet, PNAS

## 前言

在[NAS]()[2]和[NASNet]()[3]中我们介绍了如何使用强化学习训练卷积网络的超参。NAS是该系列的第一篇，提出了使用强化学习训练一个控制器（RNN），该控制器的输出是卷积网络的超参，可以生成一个完整的卷积网络。NASNet提出学习网络的一个单元比直接整个网络效率更高且更容易迁移到其它数据集，并在ImageNet上取得了当时最优的效果。

本文是约翰霍普金斯在读博士刘晨曦在Google实习的一篇文章，基于NASNet提出了PNAS，其训练时间降为NASNet的1/8并且取得了目前在ImageNet上最优的效果。其主要的优化策略为：

1. 更小的搜索空间；
2. Sequential model-based optimization(SMBO)：一种启发式搜索的策略，训练的模型从简单到复杂，不在精度低的模型上浪费时间；
3. Beam Search：在训练过程中剪枝掉精度差的搜索空间；
4. 代理函数：使用代理函数预测模型的精度，省去了耗时的训练过程。

## 1. PNAS详解

### 1.1 更小的搜索空间



## Reference

[1] Liu C, Zoph B, Shlens J, et al. Progressive neural architecture search[J]. arXiv preprint arXiv:1712.00559, 2017.

[2] Zoph B, Le Q V. Neural architecture search with reinforcement learning[J]. arXiv preprint arXiv:1611.01578, 2016.

[3] Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition[J]. arXiv preprint arXiv:1707.07012, 2017, 2(6).