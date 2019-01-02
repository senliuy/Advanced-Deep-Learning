# Batch Normalization

## 前言

Batch Normalization(BN)是深度学习中非常好用的一个算法，加入BN层的网络往往更加稳定并且BN还起到了一定的正则化的作用。在这篇文章中，我们将详细介绍BN的技术细节[1]以及其能工作的原因[2]。

在提出BN的文章中[1]，作者BN能工作的原因是BN解决了普通网络的内部协变量偏移（Internel Covariate Shift, ICS）的问题，所谓ICS是指网络各层的分布不一致，网络需要适应这种不一致从而增加了学习的难度。而在[2]中，作者通过实验验证了BN其实和ICS的关系并不大，其能工作的原因是使损失平面更加平滑，并给出了其结论的数学证明。

## 1. BN详解

## Reference

[1] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[J]. arXiv preprint arXiv:1502.03167, 2015.

[2] Santurkar S, Tsipras D, Ilyas A, et al. How Does Batch Normalization Help Optimization?(No, It Is Not About Internal Covariate Shift)[J]. arXiv preprint arXiv:1805.11604, 2018.