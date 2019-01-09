# Layer Normalization

## 前言

在上一篇的文章中我们介绍了BN\[2\]的计算方法并且讲解了BN如何应用在MLP以及CNN中如何使用BN。在文章的最后，我们指出BN并不适用于RNN等动态网络和batchsize较小的时候效果不好。Layer Normalization（LN）\[1\]的提出有效的解决BN的这两个问题。LN和BN不同点是归一化的维度是互相垂直的，如图1所示。在图1中$$N$$表示样本轴，C表示通道轴，F是每个通道的特征数量。BN如右侧所示，它是取不同样本的同一个通道的特征做归一化；LN则是如左侧所示，它取的是同一个样本的不同通道做归一化。

![](/assets/LN_1.png)

## 1. BN的问题

以图1为例我们仔细分析BN为什么有不适用于RNN和不适用于batchsize较小的场景的原因。图2是一个简单的RNN网络，其中一个含有5个样本（$$N$$）组成的一个batch，在这个例子中每个样本都包含4个特征（$$C$$），但是每个样本的长度不同（$$T$$）。在BN中是按照特征进行归一化的，也就是沿着平行于$$y$$轴的方向进行归一化。

![](/assets/LN_2.png)

### 1.1 BN与Batch Size

如图1右侧部分，BN是按照样本数计算归一化统计量的，当样本数很少时，比如说只有4个。这四个样本的均值和方差便不能反映全局的统计信息，所以基于少量样本的BN的效果会变得很差。在一些场景中，比如说硬件资源受限，在线学习等场景，BN是非常不适用的。

### 1.2 BN与RNN

RNN可以展开成一个隐藏层共享参数的MLP，随着时间片的增多，展开后的MLP的层数也在增多，最终层数由输入数据的时间片的数量决定，所以RNN是一个动态的网络。

图1

## Reference

\[1\] Ba J L, Kiros J R, Hinton G E. Layer normalization\[J\]. arXiv preprint arXiv:1607.06450, 2016.

\[2\] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift\[J\]. arXiv preprint arXiv:1502.03167, 2015.

