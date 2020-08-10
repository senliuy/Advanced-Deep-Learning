# 基于Seq2Seq的公式识别引擎

### 前言

公式识别是OCR领域一个非常有挑战性的工作，工作的难点在于它是一个二维的数据，因此无法用传统的CRNN进行识别。这里介绍的是源自GitHub的一篇源码以及其相关的博客，这篇源码的代码水平以及使用的一些技巧都是非常巧妙的，笔者基于这份源码训练得到的公式识别模型也达到了一个比较高的准确率，下面是这篇文章相关的链接：

* 源码：[https://github.com/guillaumegenthial/im2latex](https://github.com/guillaumegenthial/im2latex)
* 博客：[https://guillaumegenthial.github.io/image-to-latex.html](https://guillaumegenthial.github.io/image-to-latex.html)

  ![Producing LaTeX code from an image](https://guillaumegenthial.github.io/assets/img2latex/img2latex_task.svg)

### 1. 基础介绍

#### 1.1 Seq2Seq模型

**编码器**：Seq2Seq模型是在机器翻译中最先引入的概念，这个模型由编码器（Encoder）和解码器（Decoder）组成，这里以`how are you`的英法翻译作为范例。首先，我们先通过word2vec将单词编码成特征向量​。在这个例子中，我们又三个单词，则他们被编译成一个矩阵​。然后我们将这三个特征向量依次输入LSTM中得到三个特征编码​，最终得到整体的特征编码​。上述整体过程如图1所示。

![Vanilla Encoder](https://guillaumegenthial.github.io/assets/img2latex/seq2seq_vanilla_encoder.svg)

**解码器**：解码器的作用是将上面得到的​解码成对应的识别结果，它是通过将单词的编码结果依次输入到解码器中得到的。具体的讲，解码器使用了另外一组LSTM作为网络模型，​作为隐藏状态。首先它使用起始字符​作为输入，然后通过LSTM计算下一个隐层状态​。然后通过一个激活函数​ 得到这个时间片的输出​，其中​的大小是和字典相同的。然后使用一个softmax作用于​得到一个概率向量​，其中​表示这个数是这个单词的概率，那么这个时间片最终的预测结果就是取​对应的单词，即​，下图中表示为`comment`。接着下个时间片去`comment`编码的特征向量作为输入，​作为隐层节点的状态输入到下一个LSTM时间片中得到概率向量​，剩下的时间片以此类推。解码器的结构如图2所示。

![Vanilla Decoder](https://guillaumegenthial.github.io/assets/img2latex/seq2seq_vanilla_decoder.svg)

#### 1.2 带Attention的Seq2Seq模型

Attention的本质是通过一组全连接为每个特征学习一个权值，特征既可以按空间划分，也可以按时间划分。通过Attention机制学习到的权值，往往在较为重要的部分拥有较高的值，而在次要的部分拥有较低的值。而将Attention加在哪个部分的哪个维度是一件非常有意思的事情，一个比较常见的方式是将Attention作用在编码层的特征部分，而划分方式则以时间片为单位。如图3所示，在LSTM中，​是每个时间片的隐层状态，为​赋予一个权值是一个非常常见的策略。首先，我们使用函数​ 为解码器每个时间片的隐层状态​计算一个评分，然后归一化这个评分并将它作为每个时间片的状态的权值。

![Attention Mechanism](https://guillaumegenthial.github.io/assets/img2latex/seq2seq_attention_mechanism_new.svg)

#### 1.3 Seq2Seq模型的训练

因为在模型最开始的训练阶段每个时间片的预测都很不稳定，如果使用上一个时间片的预测作为新的时间片的输出，这将导致模型非常难以收敛。在Seq2Seq模型的训练中，一个技巧是使用标签句子的当前时间片的内容作为训练过程的下个时间片的输入，如图4所示。

![Training](https://guillaumegenthial.github.io/assets/img2latex/img2latex_training.svg)

在上面的训练过程中，解码器的输出是一个概率分布，表示的是字典中的每个单词的输出概率，那么这个标签句子的预测概率便是每个时间片的乘积。

训练的目标便是最大化目标句子的输出概率，这个目标往往会转化成最小化这个等式的负log，即：

专业一点的说这也就是最小化目标句子分布和预测句子分布的交叉熵。

#### 1.4 Seq2Seq解码

Seq2Seq解码一般使用贪心或者beam search进行解码，关于这两个方式的介绍，可以参考我的CTC文章中的1.3章节。

### 2. 算法详解

Latex是公式的一种文本化的表示形式，目前主流的公式识别引擎都是将图片格式或者手写公式的笔记信息转化成其对应的Latex的表达形式，如图1所示。

公式识别和第1节介绍的机器翻译技术非常类似，不同的是输入数据由文本变成了公式图像，编码器由RNN变成了卷积网络。因为公式数据是一个二维的数据，所以不能采用RNN的架构，而是采用Image Caption类似的架构。

#### 2.1 数据

**2.1.1 归一化**

公式的Latex表示形式往往不是唯一的，这种数据和标签的一对多问题导致模型的训练过程中很难收敛，因此需要将公式图像的标签归一化到一种形式，主要从两个方面进行归一化

* Latex符号的归一化：在Latex语法中，很多公式符号的Latex表示并不是唯一的，例如’\rightarrow‘ 和'\to'都是​等。这种类似的多对一的符号需要归一化到一种，使每个符号都有一种表示形式。
* Latex语法的归一化：一个公式的Latex的表达形式，例如方程组既可以用\begin{array}也可以用\begin{matrix}等。另外Latex中左右中括号的使用也比较随意，这也是需要归一化的一点，原则上是所有句样本的使用方式均保持统一且Latex标签长度越短越好。

**2.1.2 字典**

在源码中，字典是根据训练集样本构建的。首先遍历训练集，统计训练集中每个token出现的频数，频数高于阈值的加入字典中，频数低于阈值的统一设为’\_UNK‘\(unknown\)字符。另外源码中增加了两个字符’\_PAD‘和’\_END‘分别用于表示填充字符和结束字符。

**2.1.2 图像数据**

源码中的公式图像数据是使用pdflatex和ImageMagic合成的，过程是image.py中的`convert_to_png`函数。具体可以分成下面几步：

1. 根据Latex公式生成一个内容仅有公式数据的.tex文件；
2. 使用pdflatex将.tex转成公式的pdf文件；
3. 使用ImageMagic将pdf文件转换成png；
4. 这时的png是一个仅包含一个公式的A4纸大小的文件，因此需要从中裁剪出公式区域；
5. 为了提升模型的收敛速度，算法使用了图像buckets的策略，因此这一步便是匹配公式图像对应的bucket，并通过padding的形式将图像padding到bucket的大小。

图像bucket是指将类似大小的尺寸resize或者padding到相同的尺寸，然后在训练时每个批次抽样的图像的大小保持相同，这个策略在输入图像的尺寸变化非常大的数据非常有帮助。这样做的优点有二：1. 使用尽可能小的bucket可以尽可能的避免过分resize引发的图像拉伸的问题，而对于padding来说则可以减少空白区域的大小，这些对模型的收敛都是很有帮助的；2. 大小不同的批次可以提升模型的训练速度，并在测试的时候提升预测的准确率。

#### 2.2 模型

**2.2.1 编码器**

Im2Latex模型将Seq2Seq模型中的编码器由RNN换成了CNN，即使用卷积网络将公式的图像数据转换成一个序列向量​，每个向量表示公式图像的一个区域，这些向量将作为特征送到解码器中。假设公式图像的大小是​，经过若干组卷积核池化操作后，得到了​的特征向量，这些特征向量经过reshape操作后得到一个特征数为512，时间片个数为​的向量序列。

![Convolutional Encoder - produces a sequence of vectors](https://guillaumegenthial.github.io/assets/img2latex/img2latex_encoder.svg)

因为在得到向量序列的过程中使用了`reshape`操作，因此丢失了图像的位置信息，为了弥补这个问题，作者添加了 [Attention is All you Need](https://senliuy.gitbook.io/advanced-deep-learning/di-er-zhang-ff1a-xu-lie-mo-xing/attention-is-all-you-need)中使用的位置编码。位置编码的维度绝大多数时候都采用和特征向量相同的维度，以便于拼接或者单位加的操作。

