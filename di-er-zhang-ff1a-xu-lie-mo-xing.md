# Hierarchical Attention Networks for Document Classification

tags: Attention

## 前言

本文提出了一个Hierarchical Attention Network（HAN）模型用来做文章分类的任务，该算法提出的动机是考虑到在一个句子中，不同的单词对于决定这个句子的含义起着不同的作用；然后在一篇文章中，不同的句子又对于该文档的分类起着不同的作用。所以这篇层次Attention模型分别在单词层次和句子层次添加了一个Attention机制。实验结果表明这种机制可以提升文章分类的效果，同时通过Attention的权值向量的权值我们可以看出究竟哪些句子以及哪些单词对文档分类起着更重要的作用。

## 1. HAN算法详解

### 1.1 网络结构

HAN的网络结构如图1所示，

![](/assets/HAN_1.png)

## Reference

\[1\] Yang Z, Yang D, Dyer C, et al. Hierarchical attention networks for document classification\[C\]//Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies. 2016: 1480-1489.

