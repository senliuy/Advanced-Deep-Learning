# Show and Tell: A Neural Image Caption Generator

## 前言

图像描述（Image Catpion）是非常经典的图像二维信息识别的一个英语。类似的还有表格识别，公式识别等。如图1所示，Image Caption的输入时衣服图像，输出是对这个图像的描述。Image Caption的难点有二：

1. 模型不仅要能够对图像中的每一个物体进行分类，还需要能够理解和描述它们的空间关系。
2. 描述的生成要考虑语义信息，当前的输出高度依赖之前生成的内容。

这篇论文提供了一个Image Caption的基础框架：即用CNN作为特征提取器用于将图像转换为特征向量，之后用一个RNN作为解码器（生成器），用于生成对图像的描述。

![](/assets/SAndT_1.png)

## Show and Tell详解

Image Caption也是采用的[Encoder-Decoder](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/learning-phrase-representations-using-rnn-encoder-decoder-for-statistical-machine-translation.html)的算法框架, 作者当初设计这个算法的时候，也是借鉴了神经机器翻译的思想，故而采用了类似的网络架构。

和机器翻译类似，Image Caption的目标也是最大化标签值得概率，这里的标签即使训练集的描述内容$$S$$，表示为:

$$
\theta ^ * = \arg \max _ {\theta} \sum _{ (I, \theta) } \log p{S|I; \theta}
$$

