# Batch Normalization

## 前言

Batch Normalization\(BN\)是深度学习中非常好用的一个算法，加入BN层的网络往往更加稳定并且BN还起到了一定的正则化的作用。在这篇文章中，我们将详细介绍BN的技术细节\[1\]以及其能工作的原因\[2\]。

在提出BN的文章中\[1\]，作者BN能工作的原因是BN解决了普通网络的内部协变量偏移（Internel Covariate Shift, ICS）的问题，所谓ICS是指网络各层的分布不一致，网络需要适应这种不一致从而增加了学习的难度。而在\[2\]中，作者通过实验验证了BN其实和ICS的关系并不大，其能工作的原因是使损失平面更加平滑，并给出了其结论的数学证明。

## 1. BN详解

### 1.1 内部协变量偏移

BN的提出是基于小批量随机梯度下降（mini-batch SGD）的，mini-batch SGD是介于one-example SGD和full-batch SGD的一个折中方案，其优点是比full-batch SGD有更小的硬件需求，比one-example SGD有更好的收敛速度和并行能力。随机梯度下降的缺点是对参数比较敏感，较大的学习率和不合适的初始化值均有可能导致训练过程中发生梯度消失或者梯度爆炸的现象的出现。BN的出现则有效的解决了这个问题。

在Sergey Ioffe的文章中，他们认为BN的主要贡献是减弱了内部协变量偏移（ICS）的问题，论文中对ICS的定义是：as the change in the distribution of network activations due to the change in network parameters during training。作者认为ICS是导致网络收敛的慢的罪魁祸首，因为模型需要学习在训练过程中会不断变化的隐层输入分布。作者提出BN的动机是企图在训练过程中将每一层的隐层节点的输入固定下来，这样就可以避免ICS的问题了。

在深度学习训练中，白化（Whiten）是加速收敛的一个小Trick，所谓白化是指将图像像素点变化到均值为0，方差为1的正态分布。我们知道在深度学习中，第$$i$$层的输出会直接作为第$$i+1$$层的输入，所以我们能不能对神经网络的每一层的输入都做一次白化呢？其实BN就是这么做的。

### 1.2 梯度饱和

我们知道sigmoid激活函数和tanh激活函数存在梯度饱和的区域，其原因是激活函数的输入值过大或者过小，其得到的激活函数的梯度值会非常接近于0，使得网络的收敛速度减慢。传统的方法是使用不存在梯度饱和区域的激活函数，例如ReLU等。BN也可以缓解梯度饱和的问题，它的策略是在调用激活函数之前将$$WX+b$$的值归一化到梯度值比较大的区域。

### 1.3 BN过程

如果按照传统白化的方法，整个数据集都会参与归一化的计算，但是这种过程无疑是非常耗时的。BN的归一化过程是以批量为单位的。如图1所示，假设一个批量有$$n$$个样本 $$\mathcal{B} = \{x_1, x_2, ..., x_m\}$$，每个样本有$$d$$个特征，那么这个批量的每个样本第$$k$$个特征的归一化后的值为


$$
\hat{x}^{(k)} = \frac{x^{(k)} - E[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}]}}
$$


其中$$E$$和 $$\text{Var}$$分别表示在第$$k$$个特征上这个批量中所有的样本的均值和方差。

![](/assets/BN_1.png)

这种表示会对模型的收敛有帮助，但是也可能破坏已经学习到的特征。为了解决这个问题，BN添加了两个可以学习的变量$$\beta$$和$$\gamma$$用于控制网络能够表达直接映射，也就是能够还原BN之前学习到的特征。


$$
y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}
$$


当$$\gamma^{(k)} = \sqrt{\text{Var}[x^{(k)}]}$$并且$$\beta^{(k)} = E[x^{(k)}]$$时，$$y^{(k)} = x^{(k)}$$，也就是说经过BN操作之后的网络容量是不小于没有BN操作的网络容量的。

综上所述，BN可以看做一个以$$\gamma$$和$$\beta$$为参数的，从$$x_{1...m}$$到$$y_{1...m}$$的一个映射，表示为

$$
BN_{\gamma,\beta}:x_{1...m}\rightarrow y_{1...m}
$$

BN的伪代码如算法1所示



## Reference

\[1\] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift\[J\]. arXiv preprint arXiv:1502.03167, 2015.

\[2\] Santurkar S, Tsipras D, Ilyas A, et al. How Does Batch Normalization Help Optimization?\(No, It Is Not About Internal Covariate Shift\)\[J\]. arXiv preprint arXiv:1805.11604, 2018.

