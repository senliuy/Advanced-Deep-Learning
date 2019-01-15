# Weight Normalization

## 前言

之前介绍的BN和LN都是在数据的层面上做的归一化，而这篇文章介绍的Weight Normalization（WN)是在权值的维度上做的归一化。WN的做法是将权值矩阵$$W$$在其欧氏范数和其方向上解耦成了矩阵参数$$\mathbf{v}$$和向量参数$$g$$后使用SGD分别优化这两个参数。

WN也是和样本量无关的，所以可以应用在batchsize较小以及RNN等动态网络中；另外BN使用的基于mini-batch的归一化统计量代替全局统计量，相当于在梯度计算中引入了噪声。而WN则没有这个问题，所以在生成模型，强化学习等噪声敏感的环境中WN的效果也要优于BN。

WN没有一如额外参数，这样更节约显存。同时WN的计算效率也要优于要计算归一化统计量的BN。

## 1. WN详解

一层神经网络的计算可以表示为：

$$
y = \phi(\mathbf{w}\cdot\mathbf{x}+b)
$$


## Reference