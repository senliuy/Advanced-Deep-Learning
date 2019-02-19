# Hierarchical Attention Networks for Document Classification

tags: Attention

## 前言

本文提出了一个Hierarchical Attention Network（HAN）模型用来做文章分类的任务，该算法提出的动机是考虑到在一个句子中，不同的单词对于决定这个句子的含义起着不同的作用；然后在一篇文章中，不同的句子又对于该文档的分类起着不同的作用。所以这篇层次Attention模型分别在单词层次和句子层次添加了一个Attention机制。实验结果表明这种机制可以提升文章分类的效果，同时通过Attention的权值向量的权值我们可以看出究竟哪些句子以及哪些单词对文档分类起着更重要的作用。

## 1. HAN算法详解

### 1.1 网络结构

HAN的网络结构如图1所示，它的核心结构由两个部分组成，下面是一个单词编码器加上基于单词编码的Attention层，上面是一个句子编码器和一个基于句子编码的Attention层。

![](/assets/HAN_1.png)

在详细介绍网络结构之前我们先给出几个重要参数的定义。假设一篇文章由$$L$$个句子组成，第$$s_i(i\in[1,L])$$个句子包含$$T_i$$个单词，$$w_{it}$$是第$$i$$个句子中的第$$t(t\in[1,T_i])$$个单词。

### 1.2 单词编码器

图1中最底下的部分是个单词编码器，它的输入是一个句子。给定一个由单词$$w_{it}$$组成的句子$$T_i$$，它首先会经过一个嵌入矩阵编码成一个特征向量，例如word2vec等方法：

$$
x_{it} = W_e w_{it}, t\in[1,T]
$$

之后使用一个单层的双向[GRU](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/neural-machine-translation-by-jointly-learning-to-align-and-translate.html)[2]对$$x_{it}$$进行编码：

$$
\overrightarrow{h}_{it} = \overrightarrow{GRU}(x_{it}),t\in[1,T]
$$

$$
\overleftarrow{h}_{it} = \overleftarrow{GRU}(x_{it}), \quad t\in[T,1]
$$

双向GRU的输出是通过拼接前向GRU和反向GRU的方式得到的。

$$
h_{it} = [\overrightarrow{h}_{it}; \overleftarrow{h}_{it}]
$$

### 1.3 单词Attention

单词编码器之上是一个单词Attention模块，它首先会将上一层得到的$$h_{it}$$输入一个MLP中得到它的非线性表示：

$$
u_{it} = \text{tanh}(W_w h_{it} + b_w)
$$

接着便是Attention部分，首先需要使用softmax计算每个特征的权值。在论文中使用了Memory Network[3]，Memory Network是于2014年有FAIR提出的一种类似于神经图灵机的结构，它的核心部件是一个叫做记忆单元的部分，用来长期保存特征向量，也就是论文中的上下文向量$$u_w$$，它的值也会随着训练的进行而更新。Memory Network经过几年的发展也有了很多性能更优的版本，但是由于坑比较深且业内没有广泛使用，暂时没有学习它的计划，感兴趣的同学请自行学习相关论文和代码。结合了Memory Network的权值的计算方式为：

$$
\alpha_{it} = \frac{\text{exp}(u_{it}^\top u_w)}{\sum_t \text{exp} (u_{it}^\top u_w)}
$$

最后得到的这个句子的编码$$s_i$$便是以$$h_{it}$$作为向量值，$$\alpha_{it}$$作为权值的加权和：

$$
s_i = \sum_t \alpha_{it} h_{it}
$$

### 1.4 句子编码器

句子编码器的也是使用了一个双向GRU，它的结构和单词编码器非常相似，数学表达式为：

$$
\overrightarrow{h}_{i} = \overrightarrow{GRU}(s_{i}),t\in[1,T]
$$

$$
\overleftarrow{h}_{i} = \overleftarrow{GRU}(s_{i}),t\in[T,1]
$$

$$
h_{i} = [\overrightarrow{h}_{i}; \overleftarrow{h}_{i}]
$$

### 1.5 句子Attention

HAN的句子Attention部分也是使用了Memory Network带上下文向量的Attention结构，它的输入是句子编码器得到的特征向量，输出的是整个文本的特征向量$$v$$。

$$
u_{i} = \text{tanh}(W_s h_{i} + b_s)
$$

$$
\alpha_{i} = \frac{\text{exp}(u_{i}^\top u_s)}{\sum_i \text{exp} (u_{i}^\top u_s)}
$$

$$
v = \sum_i \alpha_{i} h_{i}
$$

### 1.5 句子分类

使用softmax激活函数我们可以根据文本向量$$v$$得到其每个类别的预测概率$$p$$：

$$
p = \text{softmax}(W_c v + b_c) 
$$

由于使用了softmax激活函数，那么它的损失函数则应该是负log似然：

$$
L = -\sum_d \text{log} p_{dj}
$$

其中$$d$$是批量中样本的下标，$$j$$是分类任务中类别的下标。

## 2. 总结


## Reference

\[1\] Yang Z, Yang D, Dyer C, et al. Hierarchical attention networks for document classification\[C\]//Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016: 1480-1489.

[2] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate\[J\]. arXiv preprint arXiv:1409.0473, 2014.

[3] Weston J, Chopra S, Bordes A. Memory networks[J]. arXiv preprint arXiv:1410.3916, 2014.


