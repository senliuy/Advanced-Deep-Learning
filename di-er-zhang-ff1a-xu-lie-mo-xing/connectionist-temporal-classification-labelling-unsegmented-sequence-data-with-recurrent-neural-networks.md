# Connectionist Temporal Classification : Labelling Unsegmented Sequence Data with Recurrent Neural Networks

本文主要参考自Hannun等人在distill.pub发表的文章（[https://distill.pub/2017/ctc/](https://distill.pub/2017/ctc/)），感谢Hunnun等人对CTC的梳理。

## 简介

在语音识别中，我们的数据集是音频文件和其对应的文本，不幸的是，音频文件和文本很难再单词的单位上对齐。除了语言识别，在OCR，机器翻译中，都存在类似的Sequence to Sequence结构，同样也需要在预处理操作时进行对齐，但是这种对齐有时候是非常困难的。如果不使用对齐而直接训练模型时，由于人的语速的不同，或者字符间距离的不同，导致模型很难收敛。

CTC\(Connectionist Temporal Classification\)是一种避开输入与输出的一种方式，是非常适合语音识别或者OCR这种应用的。

\[CTC\_1.png\]

给定输入序列X=\[x1,x2,...,xT\]以及对应的标签数据Y=\[y1,y2,..,yU\],例如语音识别中的音频文件和文本文件。我们的工作是找到X到Y的一个映射，这种对时序数据进行分类的算法叫做Temporal Classification。

对比传统的分类方法，时序分类有如下难点：

1. X和Y的长度都是变化的；
2. X和Y的长度是不相等的；
3. 我们并不关心X和Y之间的对齐。

CTC提供了解决方案，对于一个给定的输入序列X, CTC给出所有可能的Y的输出分布。根据这个分布，我们可以输出最可能的结果或者给出某个输出的概率。

损失函数：给定输入序列X，我们希望最大化Y的后验概率P\(Y\|X\), P\(Y\|X\)应该是可导的，这样我们能执行梯度下降算法；

测试：给定一个训练好的模型和输入序列X，我们希望输出概率最高的Y:

```
Y^* = argmax_Y p(Y|X)
```

当然，在测试时，我们希望Y^\*能够尽快的被搜索到。

## 算法详解

给定输入X，CTC输出每个可能输出及其条件概率。问题的关键是我们要找到CTC的输出和标签数据的对齐，这样我们才能够计算Loss。

