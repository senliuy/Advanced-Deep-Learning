# Regularized Evolution for Image Classifier Architecture Search

## 前言

在我们之前介绍的NAS系列算法中，模型结构的搜索均是通过强化学习实现的。这篇要介绍的AmoebaNet是通过遗传算法的进化策略（Evolution）实现的模型结构的学习过程。该算法的主要特点是在进化过程中引入了年龄的概念，使进化时更倾向于选择更为年轻的性能好的结构，这样确保了进化过程中的多样性和优胜劣汰的特点，这个过程叫做年龄进化（Aging Evolution，AE）。作者为他的网络取名AmoebaNet，Amoeba中文名为变形体，是对形态不固定的生物体的统称，作者也是借这个词来表达AE拥有探索更广的搜索空间的能力。AmoebaNet取得了当时在ImageNet数据集上top-1和top-5的最高精度。

## 1. AmoebaNet算法详解

### 1.1 搜索空间

AmoebaNet使用的是和[NASNet]()[2]相同的搜索空间

## Reference

[1] Real E, Aggarwal A, Huang Y, et al. Regularized evolution for image classifier architecture search[J]. arXiv preprint arXiv:1802.01548, 2018.

[2] Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition\[J\]. arXiv preprint arXiv:1707.07012, 2017, 2\(6\).

 