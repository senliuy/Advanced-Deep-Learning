# Weight Normalization

## 前言

之前介绍的BN和LN都是在数据的层面上做的归一化，而这篇文章介绍的Weight Normalization（WN)是在权值的维度上做的归一化。WN的做法是将权值矩阵$$W$$在其欧氏范数和其方向上解耦成了矩阵参数$$\mathbf{v}$$和向量参数$$g$$，然后分别优化这两个向量。

## Reference