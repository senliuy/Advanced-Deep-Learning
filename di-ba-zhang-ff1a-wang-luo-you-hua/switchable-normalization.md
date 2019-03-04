# Switchable Normalization

tags: Normalization

## 前言

在之前的文章中，我们介绍了[BN](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-ba-zhang-ff1a-wang-luo-you-hua/batch-normalization.html){{"ioffe2015batch"|cite}}，[LN](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-ba-zhang-ff1a-wang-luo-you-hua/layer-normalization.html){{"ba2016layer"|cite}}，[IN](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-ba-zhang-ff1a-wang-luo-you-hua/instance-normalization.html){{"ulyanov2016instance"|cite}}以及[GN](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-ba-zhang-ff1a-wang-luo-you-hua/group-normalization.html){{"wu2018group"|cite}}的算法细节及适用的任务。虽然这些归一化方法往往能提升模型的性能，但是当你接收一个任务时，具体选择哪个归一化方法仍然需要人工选择，这往往需要大量的对照实验或者开发者优秀的经验才能选出最合适的归一化方法。本文提出了Switchable Normalization（SN）{{"luo2018differentiable"|cite}}，它的算法核心在于提出了一个可微的归一化层，可以让模型根据数据来学习到每一层该选择的归一化方法，亦或是三个归一化方法的加权和，如图1所示。所以SN是一个任务无关的归一化方法，不管是LN适用的RNN还是IN适用的图像风格迁移（IST），SN均能用到该应用中。作者在实验中直接将SN用到了包括分类，检测，分割，IST，LSTM等各个方向的任务中，SN均取得了非常好的效果。

<figure>
<img src="/assets/SN_1.png" alt="图1：SN是LN，BN以及IN的加权和" />
<figcaption>图1：SN是LN，BN以及IN的加权和</figcaption>
</figure>

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

### 1.3 测试

在BN的测试过程中，为了计算其归一化统计量，传统的BN方法是从训练过程中利用滑动平均的方法得到的均值和方差。在SN的BN部分，它使用的是一种叫做**批平均**batch average的方法，它分成两步：1.固定网络中的SN层，从训练集中随机抽取若干个批量的样本，将输入输入到网络中；2.计算这些批量在特定SN层的$$\mu$$和$$\sigma$$的平均值，它们将会作为测试阶段的均值和方差。实验结果表明，在SN中批平均的效果略微优于滑动平均。

## 2. SN的优点

### 2.1 SN的普遍适用性

SN通过根据不同的任务调整不同归一化策略的权值使其可以直接应用到不同的任务中。图2可视化了在不同任务上不同归一化策略的权值比重：

<figure>
<img src="/assets/SN_2.png" alt="图2：SN在不同任务下的权值分布可视化图" />
<figcaption>图2：SN在不同任务下的权值分布可视化图</figcaption>
</figure>

从图2中我们可以看出LSTM以及IST都学到了最适合它们本身的归一化策略。

### 2.2 SN与BatchSize

SN也能根据batchsize的大小自动调整不同归一化策略的比重，如果batchsize的值比较小，SN学到的BN的权重就会很小，反之BN的权重就会很大，如图3所示：

<figure>
<img src="/assets/SN_3.png" alt="图3：SN在不同batchsize下的权值分布可视化图" />
<figcaption>图3：SN在不同batchsize下的权值分布可视化图</figcaption>
</figure>

图3中括号的意思是(#GPU, batchsize)。

## 3. 总结

这篇文章介绍了统一了BN，LN以及IN三种归一化策略的SN，SN具有以下三个有点：

1. 鲁棒性：无论batchsize的大小如何，SN均能取得非常好的效果；
2. 通用性：SN可以直接应用到各种类型的应用中，减去了人工选择归一化策略的繁琐；
3. 多样性：由于网络的不同层在网络中起着不同的作用，SN能够为每层学到不同的归一化策略，这种自适应的归一化策略往往要优于单一方案人工设定的归一化策略。

