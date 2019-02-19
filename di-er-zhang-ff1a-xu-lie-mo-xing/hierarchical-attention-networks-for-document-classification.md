# Hierarchical Attention Networks for Document Classification

tags: Attention

## 前言

本文提出了一个Hierarchical Attention Network（HAN）模型用来做文章分类的任务，该算法提出的动机是考虑到在一个句子中，不同的单词对于决定这个句子的含义起着不同的作用；然后在一篇文章中，不同的句子又对于该文档的分类起着不同的作用。所以这篇层次Attention模型分别在单词层次和句子层次添加了一个Attention机制。实验结果表明这种机制可以提升文章分类的效果，同时通过Attention的权值向量的权值我们可以看出究竟哪些句子以及哪些单词对文档分类起着更重要的作用。

## 1. HAN算法详解

### 1.1 网络结构

HAN的网络结构如图1所示，它的核心结构由两个部分组成，下面是一个单词编码器加上基于单词编码的Attention层，上面是一个句子编码器和一个基于句子编码的Attention层。

![](/assets/HAN_1.png)

在详细介绍网络结构之前我们先给出几个重要参数的定义。假设一篇文章由$$L$$个句子组成，第$$s_i(i\in[1,L])$$个句子包含$$T_i$$个单词，$$w_{it}$$是第$$i$$个句子中的第$$t(t\in[1,T_i])$$个单词。

## 1.2 单词编码器

给定一个由单词$$w_{it}$$组成的句子$$T_i$$，它首先会经过一个嵌入矩阵编码成一个特征向量，例如word2vec等方法：

$$
x_{it} = W_e w_{it}, t\in[1,T]
$$

之后使用一个单层的双向GRU对$$x_{it}$$进行编码：

$$
\overrightarrow{h}_{it} = \overrightarrow{GRU}(x_{it}),t\in[1,T]
$$

$$
\overleftarrow{h}_{it} = \overleftarrow{GRU}(x_{it}), \quad t\in[T,1]
$$



## Reference

\[1\] Yang Z, Yang D, Dyer C, et al. Hierarchical attention networks for document classification\[C\]//Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016: 1480-1489.

