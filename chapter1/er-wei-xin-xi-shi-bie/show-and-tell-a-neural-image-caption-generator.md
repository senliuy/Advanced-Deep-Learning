# Show and Tell: A Neural Image Caption Generator

## 前言

图像描述（Image Catpion）是非常经典的图像二维信息识别的一个英语。类似的还有表格识别，公式识别等。如图1所示，Image Caption的输入时衣服图像，输出是对这个图像的描述。Image Caption的难点有二：

1. 模型不仅要能够对图像中的每一个物体进行分类，还需要能够理解和描述它们的空间关系。
2. 描述的生成要考虑语义信息，当前的输出高度依赖之前生成的内容。

这篇论文提供了一个Image Caption的基础框架：即用CNN作为特征提取器用于将图像转换为特征向量，之后用一个RNN作为解码器（生成器），用于生成对图像的描述。

![](/assets/SAndT_1.png)

## 1. Show and Tell详解

### 1.1 网络结构

Image Caption也是采用的[Encoder-Decoder](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/learning-phrase-representations-using-rnn-encoder-decoder-for-statistical-machine-translation.html)的算法框架, 作者当初设计这个算法的时候，也是借鉴了神经机器翻译的思想，故而采用了类似的网络架构。论文中给出的网络结构如图2所示，为了便于理解，这里将RNN按时间片展开了，它们实际上是一个LSTM。图2的左半部分是编码器，由CNN组成，图中给的是GoogLeNet，在实际场景中我们可以根据自己的需求选择其它任意CNN。图2的右侧是一个单项[LSTM](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/about-long-short-term-memory.html)，其内部结构不再赘述。

![](/assets/SAndT_2.png)

在训练时，输入图像编码的Feature Map只在最开始的$$t_{-1}$$时刻输入，这里作者说通过实验结果表明如果每个时间片都输入会容易造成训练的过拟合且对噪声非常敏感。在预测第$$t+1$$时刻的内容时，我们会用到$$t$$时刻的输出的词编码作为特征输入，整个过程表示为：

$$
x_{-1} = CNN(I)
\\
x_{t} = W_e S_t, \quad t \in \{0 ... N-1\}
\\
p_{t+1} = \text{LSTM}(x_t)
$$

其中$$I$$是输入图像的Feature Map，在训练时$$S_t$$是$$t$$时刻的标签真值，在测试时这个值则是上一个时间片的预测结果。另外，$$S_0$$和$$S_N$$是两个特殊字符，表示句子的开始与结束。$$W_t$$是词向量的编码矩阵，$$p_{t+1}$$是预测结果的概率分布，通过最大化概率分布可以得到该时刻的输出内容。


### 1.2 损失函数

和机器翻译类似，Image Caption的目标函数也是最大化标签值得概率，这里的标签即使训练集的描述内容$$S$$，表示为:


$$
\theta ^ * = \arg \max _ {\theta} \sum _{ (I, \theta) } \log p(S|I; \theta)
$$


其中$$I$$是输入图像，$$\theta$$是模型的参数。$$\log p(S|I; \theta)$$ 表示为$$N$$个输出的概率和，第$$t$$时刻的内容是$$0$$到$$t-1$$时刻以及图像编码的后验概率：


$$
\log p(S|I; \theta) = \sum _{t=0} ^N \log p (S_t | I, S_0, \dots, S_{t-1})
$$

所以模型的损失函数是所有时间片的负log似然之和，表示为：

$$
L(I,S) = - \sum _ {i = 1} ^N \log p_t (S_t)
$$

### 1.3 测试（推理）

在Image Caption的推理过程中有两种策略，一种是贪心，另一种则是Beam Search。两者的异同可以看我在[CTC](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/connectionist-temporal-classification-labelling-unsegmented-sequence-data-with-recurrent-neural-networks.html)文章中给出的讲解。

## 2. 总结

作为一个领域的奠基的文章，算法的结构和思想还是非常简单的，整个结构几乎照搬了机器翻译的Encoder-Decoder架构。这么简简单单的照搬也能大幅刷新STOA的结果，可见深度学习的厉害之处。

受限于当时的数据集太小，作者尝试过把图像作为特征输入到每个时间片，但是导致了过拟合。随着数据集的增大，每个时间片加入图像特征无疑会更容易收敛。
