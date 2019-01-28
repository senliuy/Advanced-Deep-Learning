# Recurrent Neural Network based Language Model

tags: NLP, Language Model

## 前言

在深度学习兴起之前，NLP领域一直是统计模型的天下，例如词对齐算法GIZA++，统计机器翻译开源框架MOSES等等。在语言模型方向，n-gram是当时最为流行的语言模型方法。一个最为常用的n-gram方法是回退（backoff） n-gram，因为n值较大时容易产生特别稀疏的数据，这时候
回退n-gram会使用(n-1)-gram的值代替n-gram。

n-gram的问题是其捕捉句子中长期依赖的能力非常有限，解决这个问题的策略有cache模型和class-based模型，但是提升有限。另外n-gram算法过于简单，其是否有能力取得令人信服的效果的确要打一个大的问号。

一个更早的使用神经网络进行语言模型学习的策略是Bengio团队的使用前馈神经网络进行学习[2]。他们要求输入的数据由固定的长度，从另一个角度看它就是一个使用神经网络编码的n-gram模型，也无法解决长期依赖问题。基于这个问题，这篇文章使用了RNN作为语言模型的学习框架。

这篇文章介绍了如何使用RNN构建语言模型，至此揭开了循环神经语言模型的篇章。由于算法比较简单，这里多介绍一些实验中使用的一些trick，例如动态测试过程等，希望能对你以后的实验设计有所帮助。（TODO：待之后对神经语言模型有系统的了解后，考虑将本文融合进综述的文章中）

## 1. 算法介绍

### 1.1 RNN

这篇文章中使用了最简单的RNN版本，而现在市场上普遍选择LSTM，GRU甚至NAS等具有捕捉更长时间长期依赖的节点。在RNN中，第$$t$$个时间片$$x(t)$$读取的是$$t-1$$时刻的状态s(t-1)和$$t$$时刻的数据$$w(t)$$。$$w(t)$$是$$t$$时刻单词的one-hot编码，单词量在3万-20万之间；$$s(t-1)$$是$$t-1$$时刻的隐藏层状态，实验中隐层节点数一般是30-500个，$$t=0$$时使用0.1进行初始化。上面过程表示为：

$$
x(t) = w(t) + s(t-1)
$$

$$t$$时刻的隐藏层状态是$$x(t)$$经过sigmoid激活函数(f)得到的值，其中$$u_{ji}$$是权值矩阵：

$$
s_j(t) = f(\sum_i x_i(t)u_{ji})
$$

有的时候我们需要在每个时间片有一个输出，只需要在隐层节点$$s_j(t)$$处添加一个softmax激活函数即可：

$$
y_k(t)= g(\sum_j s_j(t)v_{kj})
$$

### 1.2 训练数据

训练语言模型的数据是不需要人工标注的，我们要做的就是寻找大量的单语言数据即可。在制作训练数据和训练标签时，我们通过取第$$0$$到$$t-1$$时刻的单词作为网络输入，第$$t$$时刻的单词作为标签值。

由于输出使用了softmax激活函数，所以损失函数的计算使用的是交叉熵，输出层的误差向量为：

$$
\text{error}(t) = \text{desired}(t) - y(t)
$$

上式中$$\text{desired}(t)$$是模型预测值，$$y(t)$$是标签值。不知道上式的得来的同学自行搜索交叉熵的更新的推导公式，此处不再赘述。更新过程使用标准的SGD即可。

### 1.3 训练细节

**初始化**：使用的是均值为0，方差为0.1的高斯分布进行初始化。
**学习率**：初始值为0.1，当模型在验证集上的精度不再提升时将学习率减半，一般10-20个Epoch之后模型就收敛了。
**正则**：

## Reference

[1] Mikolov T, Karafiát M, Burget L, et al. Recurrent neural network based language model[C]//Eleventh Annual Conference of the International Speech Communication Association. 2010.

[2] Bengio Y, Ducharme R, Vincent P, et al. A neural probabilistic language model[J]. Journal of machine learning research, 2003, 3(Feb): 1137-1155.