# Show and Tell: A Neural Image Caption Generator

## 前言

图像描述（Image Catpion）是非常经典的图像二维信息识别的一个英语。类似的还有表格识别，公式识别等。如图1所示，Image Caption的输入时衣服图像，输出是对这个图像的描述。Image Caption的难点有二：

1. 描述需要考虑整个图像在空间上的全局信息，而不仅仅是一个局部；
2. 描述的生成要考虑语义信息，当前的输出高度依赖之前生成的内容。

这篇论文提供了一个Image Caption的基础框架：即用CNN作为特征提取器用于将图像转换为特征向量，之后用一个RNN作为解码器（生成器），用于生成对图像的描述。

![](/assets/SAndT_1.png)



