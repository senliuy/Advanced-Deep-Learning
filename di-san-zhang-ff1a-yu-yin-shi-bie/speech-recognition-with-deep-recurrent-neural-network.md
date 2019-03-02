# Speech Recognition with Deep Recurrent Neural Network

## 介绍

语音识别是具有巨大市场和研究潜力的一个领域，语音识别已经有了几十年的研究历史了。2000年前，涌现了大量的语音识别技术，例如：混合高斯模型（GMM），隐马尔科夫模型（HMM），梅尔倒谱系数（MFCC），n元祖语言模型（n-gram LM）等等（图1）。在21世纪的第一个十年，这些技术被成功应用到实际系统中。但同时，语音识别的技术仿佛遇到了瓶颈期，不论科研进展还是实际应用均进展非常缓慢。

2012年，深度学习兴起。仅仅一年之后，Hinton的著名学子Alex Graves的这篇使用深度学习思想解决语音识别问题的这篇文章{{"graves2013speech"|cite}}引起了广泛关注，为语音识别开辟了新的研究方向，这篇文章可以说是目前所有深度学习解决语音识别方向的奠基性文章了。目前深度学习均采用和其类似的RNN+CTC的框架，甚至在OCR领域也是采用了同样的思路。

###### 图1: 语音识别传统模型

![](/assets/ASR_RNN_1.png)

## 算法细节

在这篇论文涉及的实验里，使用了MFCC提取音频特征，多层双向RNN {{"schuster1997bidirectional"|cite}} 编码特征（节点使用LSTM），CTC构建声学模型。由于CTC没有构建语音模型的能力，论文使用了RNN Transducer {{"graves2012sequence"|cite}} 联合训练声学模型和语言模型。结构如图2。

###### 图2：基于深度学习的语音识别架构

![](/assets/ASR_RNN_2.jpeg)

### MFCC

首先，作者使用MFCC将音波的每个时间片转换成一个39维的特征向量。MFCC（Mel-Frequency Cepstral Coefficients）的全称是梅尔频率倒谱系数，是一种基于傅里叶变换的提取音频特征的方法。之后也有使用一维卷积提取特征的方法，由于MFCC和深度学习关系不大，需要详细了解的可以自行查阅相关文档，在这里可以简单理解为一种对音频的特征提取的方法。

### 多层双向LSTM

在这篇实验中，作者使用多层双向LSTM提取音频的时序特征，关于LSTM能解决RNN的梯度消失/爆炸以及长期依赖的问题已在本书[2.4](https://senliuy.gitbooks.io/computer-vision/content/di-er-zhang-ff1a-xu-lie-mo-xing/about-long-short-term-memory.html)节分析过，此处不再赘述。双向LSTM（BLSTM）和双向RNN（BRNN）的不同之处仅在于BLSTM的隐节点使用和LSTM带有三个门机制的节点。所以我们首先讲解一下BRNN。

BRNN添加了一个沿时间片反向传播的节点，计算方式和RNN隐节点相同，但是第t个时间片的计算需要使用第t+1个时间片的隐节点

正向：$$\vec{h}_t = \sigma(W_{x\vec{h}}x_t + W_{\vec{h}\vec{h}}\vec{h}_{t-1} + b_{\vec{h}})$$

反向：$$\vec{h'}_t = \sigma(W_{x\vec{h'}}x_t + W_{\vec{h'}\vec{h'}}\vec{h'}_{t+1} + b_{\vec{h'}})$$

$$y_t = W_{\vec{h}y}\vec{h}_t + W_{\vec{h'}y}\vec{h'}_t + b_y$$

其中，$$\vec{h}_t$$和$$\vec{h'}$$分别表示正向和反向传输的隐层节点的输出。多层RNN的实现是通过stacking的形式完成的，即第n层，第t个时间片的节点使用第n-1层和第t-1个时间片的隐层节点的输出作为输入，

$$h_t^n = \sigma(W_{h^{n-1}h^n}h_t^{n-1} + W_{h^{n}h^{n}}h_{t-1}^{n} + b_h^n)$$

对于一个N层的RNN，第t个时间片的输出是

$$y_t = W_{h^Ny}h^N_t + b_y$$

### RNN Transducer

CTC使用RNN得到的特征向量作为输入，所以CTC建模的是声学模型，但是很多时候我们也需要在模型中加入语言模型。RNN Transducer便是一种联合建立声学模型和语言模型的一种方法。更具体的讲，CTC建模的是每个时间片$$y_t$$的概率分布

$$Pr(k|t) = \frac{exp(y_t[k])}{\sum_{k'=1}^{K}exp(y_t[k'])}$$

而RNN Transducer建模的是当前时间片$$y_t$$和上个时间片输出的概率分布$$p_u$$的联合概率

$$l_t = W_{\vec{h}N_l}\vec{h}_t + W_{\vec{h'}N_l}\vec{h'}_t + b_l$$

$$h_{t,u} = tanh(W_{lh}l_{t,u} + W_{pb}p_u + b_h)$$

$$y_{t,u}=W_{hy}h_{t,u} + b_y$$

$$Pr(k|t,u) = \frac{exp(y_{t,u}[k])}{\sum_{k'=1}^{K}exp(y_{t,u}[k']}$$

RNN Transducer的解码依然可以使用beam search，CTC和beam search的讲解在[2.3节](https://senliuy.gitbooks.io/computer-vision/content/di-er-zhang-ff1a-xu-lie-mo-xing/connectionist-temporal-classification-labelling-unsegmented-sequence-data-with-recurrent-neural-networks.html)已详细分析。

## 总结

作为深度学习进行语音识别的开山之作，这篇文章提供了MFCC+LSTM+CTC的框架流程。后面几年的深度学习相关论文均是在其基础上的扩展，虽然内容很简单，但是熟悉这一套流程框架还是有必要的。

