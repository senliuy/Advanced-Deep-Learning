# Recurrent Neural Network based Language Model

tags: NLP, Language Model

## 前言

在深度学习兴起之前，NLP领域一直是统计模型的天下，例如词对齐算法GIZA++，统计机器翻译开源框架MOSES等等。在语言模型方向，n-gram是当时最为流行的语言模型方法。一个最为常用的n-gram方法是回退（backoff） n-gram，因为n值较大时容易产生特别稀疏的数据，这时候
回退n-gram会使用(n-1)-gram的值代替n-gram。

n-gram的问题是其捕捉句子中长期依赖的能力非常有限，解决这个问题的策略有cache模型和class-based模型，但是提升有限。另外n-gram算法过于简单，其是否有能力取得令人信服的效果的确要打一个大的问号。

一个更早的使用神经网络进行语言模型学习的策略是Bengio团队的使用前馈神经网络进行学习[2]。他们要求输入的数据由固定的长度，从另一个角度看它就是一个使用神经网络编码的n-gram模型，也无法解决长期依赖问题。基于这个问题，这篇文章使用了RNN作为语言模型的学习框架。

这篇文章介绍了如何使用RNN构建语言模型，至此揭开了循环神经语言模型的篇章。由于算法比较简单，这里多介绍一些实验中使用的一些trick，例如动态测试过程等，希望能对你以后的实验设计有所帮助。（TODO：待之后对神经语言模型有系统的了解后，考虑将本文融合进综述的文章中）

## 1. 算法介绍



## Reference

[1] Mikolov T, Karafiát M, Burget L, et al. Recurrent neural network based language model[C]//Eleventh Annual Conference of the International Speech Communication Association. 2010.

[2] Bengio Y, Ducharme R, Vincent P, et al. A neural probabilistic language model[J]. Journal of machine learning research, 2003, 3(Feb): 1137-1155.