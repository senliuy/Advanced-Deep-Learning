# Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

## 简介

在很多时序分类（Temporal Classification)的应用中，输入数据X和输出数据Y的标签长度并不相等，而且不存在单调的映射关系，例如机器翻译，对话系统等等。为了解决这个问题，作者提出了RNN Encoder-Decoder{{"cho2014learning"|cite}}模型，RNN Encoder-Decoder是由两个RNN模型级联而成的，通过Encoder将输入数据编码成特征向量，再通过Decoder将特征向量解码成输出数据。

这篇论文的第二个贡献就是GRU\(Gated Recurrent Unit\)的提出，GRU和LSTM均是采用门机制的思想改造RNN的神经元，和LSTM相比，GRU更加简单，高效，且不容易过拟合，但有时候在更加复杂的场景中效果不如LSTM，算是RNN和LSTM在速度和精度上的一个折中方案。

论文的实现是对SMT中短语表的rescore，即使用MOSES（SMT的一个开源工具）根据平行语料产生短语表，使用GRU的RNN Encoder-Decoder对短语表中的短语对进行重新打分。

## 详解

### 1. RNN Encoder-Decoder

给定训练集$$D=(X,Y)$$，我们希望最大化输出标签的条件概率，即：


$$
p(y_1, y_2, ..., y_T' | x_1, x_2, ..., X_T)
$$


在上式中，$$T \neq T'$$。

#### 1.1 编码

RNN Encoder-Decoder的编码过程是先是通过一个RNN将变长的输入序列转换成固定长度的特征向量，再通过RNN将特征向量解码成需要的输出，如图1.

###### 图1：RNN Encoder-Decoder by Cho K

![](/assets/GRU_1.png)

输入序列是一个标准的RNN，在计算时间片$$t$$的输出$$h_{<t>}$$时，将$$h_{<t-1>}$$和$$x_{<t>}$$输入激活函数中，表示为


$$
h_{<t>} = f(h_{<t-1>}, x_{<t>})
$$


经过$$T$$个时间片后，得到一个$$h\times1$$的特征向量$$\mathbf{c}$$，其中$$h$$是隐层节点的节点数。$$f(.)$$是一个RNN单元，在这篇论文中，使用的是GRU，GRU的详细内容会在下面详细讲解。

#### 1.2 解码

RNN Encoder-Decoder的解码过程是另外一个RNN，解码器的作用是将特征向量$$\mathbf{c}$$，前一个时间片的输出$$y_{<t-1>}$$，以及前一个隐层节点$$h_{<t-1>}$$作为输入，得到$$h_{<t>}$$，表示为


$$
h_{<t>} = f(h_{<t-1>}, y_{<t-1>}, \mathbf{c})
$$


其中，$$y_{<t>}$$也是关于$$\mathbf{c}$$，$$y_{<t-1>}$$以及$$h_{<t-1>}$$的条件分布


$$
y_{<t>} = P(y_{<t>} | y_{<t-1>}, y_{<t-2>}, ..., y_{<1>}, \mathbf{c}) = g(h_{<t>}, y_{<t-1>}, \mathbf{c})
$$


RNN Encoder-Decoder的优化便是最大化$$p(y|x)$$的log条件似然


$$
\mathop{\max}_\mathbf{\theta} \frac{1}{N} \sum^{N}_{n=1}logp_\mathbf{\theta} (\mathbf{y}_n|\mathbf{x}_n)
$$


其中$$\mathbf{\theta}$$是编码解码器的所有参数。

RNN Encoder-Decoder不仅可以用于产生输出数据，根据训练好的模型，使用条件概率模型$$p_\mathbf{\theta} (\mathbf{y}_n|\mathbf{x}_n)$$，也可以对现有的的标签进行评分。在这篇论文的实验中，作者便是对SMT中的短语表进行了重新打分。

另外一篇著名的Seq2Seq的论文 {{"sutskever2014sequence"|cite}}几乎和这篇论文同时发表，在Seq2Seq中，编码器得到的特征向量仅用于作为解码器的第一个时间片的输入，结构如图2

###### 图2：RNN Encoder-Decoder by Sutskever I

![](/assets/GRU_3.jpg)

# 2. GRU

对于短语rescore这个任务，作者最先使用的是RNN模型，遗憾的是RNN表现并不是非常理想。究其原因，在时间序列数据中，$$\mathbf{h}_{<t>}$$怎样更新，使用多少比例更新值，这些是可以仔细设计的，或者可以通过数据学习到的。根据LSTM中提出的门机制的思想，作者提出了一种更简单，更高效且更不容易过拟合的GRU，图3便是GRU的结构。

###### 图3：GRU的结构

![](/assets/GRU_2.jpg)

在上图中，有两个门：重置门（reset gate）以及更新门（update gate）,两个门的计算均是通过当前时间片的输入数据$$\mathbf{x}_t$$以及上一个时间片的隐节点$$\mathbf{h}_{<t-1>}$$计算而来：

重置门$$r_j$$：


$$
r_j = \sigma([\mathbf{W}_r \mathbf{x}]_j + [\mathbf{U}_r\mathbf{h}_{<t-1>}]_j)
$$


更新门$$z_j:$$


$$
z_j = \sigma([\mathbf{W}_z \mathbf{x}]_j + [\mathbf{U}_z\mathbf{h}_{<t-1>}]_j)
$$


其中，$$[.]_j$$表示向量的第j个元素，$$\sigma$$是sigmoid激活函数。

重置们$$r_j$$用于控制前一时刻的状态$$\mathbf{h}_{<t-1>}$$对更新值的影响，当前一时刻的状态对当前状态的影响并不大时$$r_j = 0$$，则更新值只受该时刻的输入数据$$x_{t}$$的影响：


$$
\hat{h}_j^{<t>} = \phi([\mathbf{W}\mathbf{X}]_j + [\mathbf{U}(\mathbf{r}\odot\mathbf{h}_{<t-1>})]_j)
$$


其中$$\phi$$是tanh激活函数，$$\odot$$是向量的按元素相乘。

而$$z_t$$用于控制该时间片的隐节点使用多少比例的上个状态，多少比例的更新值，当$$z_t = 1$$时，则完全使用上个状态，即$$\mathbf{h}_{<t>} = \mathbf{h}_{<t-1>}$$，相当于残差网络的short-cut。

$$h_j^{<t>} = z_jh_j^{<t-1>} + (1-z_j)\hat{h}_j^{<t-1>}$$

GRU的两个门机制是可以通过SGD和整个网络的参数共同调整的。

## 3. 总结

RNN Encoder-Decoder模型的提出和RNN门机制的隐层单元（LSTM/GRU）在解决长期依赖问题得到非常好的效果是分不开的。因为解码器使用的是编码器最后一个时间片的输出，加入我们使用的是经典RNN结构，则编码器得到的特征向量将包含大量的最后一个时间片的特征，而早期时间片的特征会在大量的计算过程中被抹掉。

## 参考文献
\[1\] [https://zhuanlan.zhihu.com/p/28297161](https://zhuanlan.zhihu.com/p/28297161)

