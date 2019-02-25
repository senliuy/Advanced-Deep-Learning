# Switchable Normalization

## 前言

在之前的文章中，我们介绍了BN[2]，LN[3]，IN[4]以及GN[5]的算法细节及适用的任务。虽然这些归一化方法往往能提升模型的性能，但是当你接收一个任务时，具体选择哪个归一化方法仍然需要人工选择，这往往需要大量的对照实验或者开发者优秀的经验才能选出最合适的归一化方法。本文提出了Switchable Normalization（SN），它的算法核心在于提出了一个可微的归一化层，可以让模型根据数据来学习到每一层该使用的归一化方法。所以SN是一个任务无关的归一化方法，不管是LN适用

## Reference

[1] Luo P, Ren J, Peng Z. Differentiable Learning-to-Normalize via Switchable Normalization. arXiv:1806.10779, 2018.

[2] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[J]. arXiv preprint arXiv:1502.03167, 2015.

[3] Ba J L, Kiros J R, Hinton G E. Layer normalization[J]. arXiv preprint arXiv:1607.06450, 2016.

[4] Vedaldi V L D U A. Instance Normalization: The Missing Ingredient for Fast Stylization[J]. arXiv preprint arXiv:1607.08022, 2016.

[5] Wu Y, He K. Group normalization[J]. arXiv preprint arXiv:1803.08494, 2018.