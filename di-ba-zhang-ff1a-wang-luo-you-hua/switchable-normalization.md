# Switchable Normalization

## 前言

在之前的文章中，我们介绍了BN\[2\]，LN\[3\]，IN\[4\]以及GN\[5\]的算法细节及适用的任务。虽然这些归一化方法往往能提升模型的性能，但是当你接收一个任务时，具体选择哪个归一化方法仍然需要人工选择，这往往需要大量的对照实验或者开发者优秀的经验才能选出最合适的归一化方法。本文提出了Switchable Normalization（SN），它的算法核心在于提出了一个可微的归一化层，可以让模型根据数据来学习到每一层该选择的归一化方法，亦或是三个归一化方法的加权和，如图1所示。所以SN是一个任务无关的归一化方法，不管是LN适用的RNN还是IN适用的图像风格迁移（IST），SN均能用到该应用中。作者在实验中直接将SN用到了包括分类，检测，分割，IST，LSTM等各个方向的任务中，SN均取得了非常好的效果。

![](/assets/SN_1.png)

## 1. SN详解

### 1.1 回顾

SN实现了对BN，LN以及IN的统一。以CNN为例，假设一个4D Feature Map的尺寸为$$(N,C,W,H)$$，假设$$h_{ncij}$$和$$\hat{h}_{ncij}$$分别是归一化前后的像素点的值，其中$$n\in[1,N]$$，$$c\in[1,C]$$，$$i\in[1,H]$$，$$j\in[1,W]$$。假设$$\mu$$和$$\sigma$$分别是均值和方差，上面所介绍的所有归一化方法均可以表示为：


$$
\hat{h}_{ncij} = \gamma \frac{h_{ncij} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta
$$


其中$$\beta$$和$$\gamma$$分别是位移变量和缩放变量，$$\epsilon$$是一个非常小的数用以防止除0。上面式子概括了BN，LN，以及IN三种归一化的计算公式，唯一不同是计算$$\mu$$和$$\sigma$$统计的像素点不同。我们可以将$$\mu$$和$$\sigma$$表示为：


$$
\mu_k = \frac{1}{I_k} \sum_{(n,c,i,j)\in I_k}
 h_{ncij},
 \quad
 \sigma_k^2 = \frac{1}{I_k} \sum_{(n,c,i,j)\in I_k}
(h_{ncij} - \mu_k)^2
$$


其中$$k \in \{in,ln,bn\}$$。IN是统计的是单个批量，单个通道的所有像素点，如图1绿色部分。BN统计的是单个通道上所有像素点，如图1红色部分。LN统计的是单个批量上的所有像素点，如图1黄色部分。它们依次可以表示为$$I_{in} = \{(i,j)|i\in[1,H], j\in[1,W]\}$$，$$I_{bn} = \{(i,j)|n\in[1,N], i\in[1,H], j\in[1,W]\}$$，$$I_{ln} = \{(i,j)|c\in[1,C], i\in[1,H], j\in[1,W]\}$$。

### 1.2 SN算法介绍

SN算法是为三组不同的$$\mu_{k}$$以及$$\sigma_{k}$$分别学习三个总共6个标量值（$$w_k$$和$$w'_k$$），$$\hat{h}_{ncij}$$的计算使用的是它们的加权和：


$$
\hat{h}_{ncij} = \gamma \frac{h_{ncij} - \sum_{k\in\Omega}w_k \mu_k}{\sqrt{\sum_{k\in\Omega} w'_k \sigma_k^2 + \epsilon}} + \beta
$$


其中$$\Omega = \{in,ln,bn\}$$。在计算$$(\mu_{ln},\sigma_{ln})$$和$$(\mu_{bn},\sigma_{bn})$$时，我们可以使用$$(\mu_{in},\sigma_{in})$$作为中间变量以减少计算量。


$$
\mu_{in} = \frac{1}{HW} \sum_{i,j}^{H,W}h_{ncij}
\quad
\sigma_{in}^2 = \frac{1}{HW}\sum_{i,j}^{H,W}(h_{ncij}- \mu_{in})^2
$$



$$
\mu_{ln} = \frac{1}{C} \sum_{c=1}^{C}\mu_{in}
\quad
\sigma_{ln}^2 = \frac{1}{C}\sum_{c=1}^{C}(\sigma_{in}^2 + \mu_{in}^2) - \mu_{ln}^2
$$



$$
\mu_{bn} = \frac{1}{N} \sum_{n=1}^{N}\mu_{in}
\quad
\sigma_{bn}^2 = \frac{1}{N}\sum_{n=1}^{N}(\sigma_{in}^2 + \mu_{in}^2) - \mu_{bn}^2
$$


$$w_k$$是通过softmax计算得到的激活函数：


$$
w_k = \frac{e^{\lambda_k}}{\sum_{z\in\{in,ln,bn\}}e^{\lambda_z}}\quad \text{and} \quad k\in\{in,ln,bn\}
$$


其中$$\{\lambda_{in}, \lambda_{bn}, \lambda_{ln}\}$$是需要优化的3个参数，可以通过BP调整它们的值。同理我们也可以计算$$w'$$对应的参数值$$\{\lambda'_{in}, \lambda'_{bn}, \lambda'_{ln}\}$$。

从上面的分析中我们可以看出，SN只增加了6个参数$$\Phi = \{\lambda_{in}, \lambda_{bn}, \lambda_{ln}, \lambda'_{in}, \lambda'_{bn}, \lambda'_{ln}\}$$。假设原始网络的参数集为$$\Theta$$，带有SN的网络的损失函数可以表示为$$\mathcal{L}(\Theta, \Phi)$$，他可以通过BP联合优化$$\Theta$$和$$\Phi$$。对SN的反向推导感兴趣的同学参考论文附件H。

## 2 SN的原理

### 2.1 SN的普遍适用性

SN通过根据不同的任务调整不同归一化策略的权值使其可以直接应用到不同的任务中。图2可视化了在不同任务上不同归一化策略的权值比重：

![](/assets/SN_2.png)

从图2中我们可以看出LSTM以及IST都学到了最适合它们本身的归一化策略。

### 2.2 SN与BatchSize

  

## Reference

\[1\] Luo P, Ren J, Peng Z. Differentiable Learning-to-Normalize via Switchable Normalization. arXiv:1806.10779, 2018.

\[2\] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift\[J\]. arXiv preprint arXiv:1502.03167, 2015.

\[3\] Ba J L, Kiros J R, Hinton G E. Layer normalization\[J\]. arXiv preprint arXiv:1607.06450, 2016.

\[4\] Vedaldi V L D U A. Instance Normalization: The Missing Ingredient for Fast Stylization\[J\]. arXiv preprint arXiv:1607.08022, 2016.

\[5\] Wu Y, He K. Group normalization\[J\]. arXiv preprint arXiv:1803.08494, 2018.

\[6\] [https://htmlpreview.github.io/?https://github.com/switchablenorms/Switchable-Normalization/blob/master/blog\_cn/blog\_cn.html](https://htmlpreview.github.io/?https://github.com/switchablenorms/Switchable-Normalization/blob/master/blog_cn/blog_cn.html)

