# Batch Normalization

## 前言

Batch Normalization(BN)是深度学习中非常好用的一个算法，加入BN层的网络往往更加稳定并且BN还起到了一定的正则化的作用。在这篇文章中，我们将详细介绍BN的技术细节[1]以及其能work的原因[2]。

## Reference

[1] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[J]. arXiv preprint arXiv:1502.03167, 2015.

[2] Santurkar S, Tsipras D, Ilyas A, et al. How Does Batch Normalization Help Optimization?(No, It Is Not About Internal Covariate Shift)[J]. arXiv preprint arXiv:1805.11604, 2018.