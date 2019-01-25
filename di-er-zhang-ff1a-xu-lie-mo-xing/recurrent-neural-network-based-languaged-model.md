# Recurrent Neural Network based Language Model

tags: NLP, Language Model

## 前言

在深度学习兴起之前，NLP领域一直是统计模型的天下，例如词对齐算法GIZA++，统计机器翻译开源框架MOSES等等。在语言模型方向，n-gram是当时最为流行的语言模型方法。一个最为常用的n-gram方法是回退（backoff） n-gram，因为n值较大时容易产生特别稀疏的数据，这时候
回退n-gram会使用(n-1)-gram的值代替n-gram。

这篇文章介绍了如何使用RNN构建语言模型，至此揭开了神经语言模型的篇章。由于算法比较简单，这里多介绍一些实验中使用的一些trick，例如动态测试过程等，希望能对你以后的实验设计有所帮助。（TODO：待之后对神经语言模型有系统的了解后，考虑将本文融合进综述的文章中）

## 1. 算法介绍



## Reference

[1] Mikolov T, Karafiát M, Burget L, et al. Recurrent neural network based language model[C]//Eleventh Annual Conference of the International Speech Communication Association. 2010.