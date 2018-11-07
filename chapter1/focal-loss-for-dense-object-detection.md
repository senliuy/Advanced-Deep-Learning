# Focal Loss for Dense Object Detection

## 前言

目前主流的检测算法分为两个方向：（1）以R-CNN系列为代表的two-stage方向；（2）以YOLO系列为代表的one-stage方向。虽然one-stage方向的速度更快，但是其精度往往比较低。究其原因，有两个方面：

1. 正样本（Positive Example）和负样本（Negative Example）的不平衡；
2. 难样本（Hard Example）和易样本（Easy Example）的不平衡。

解决正负样本的不平衡的传统策略是使用平衡的交叉熵损失函数，对于难易样本的不平衡通常是使用Hard Negative Mining的策略。

而作者的解决方案是基于交叉熵提出了一个新的损失函数Focal Loss（FL）。

$$
\text{FL}(p_t) = - \alpha_t (1-p_t)^{\gamma}log(p_t)
$$
FL是一个尺度动态可调的交叉熵损失函数，在FL中有两个参数$$\alpha_t$$和$$\gamma$$，其中$$\alpha_t$$主要作用是解决正负样本的不平衡，$$\gamma$$主要是解决难易样本的不平衡。

