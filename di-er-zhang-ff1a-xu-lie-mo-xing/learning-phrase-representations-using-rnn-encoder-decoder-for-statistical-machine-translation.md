# Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

## 简介

在很多时序分类（Temporal Classification\)的应用中，输入数据X和输出数据Y的标签长度并不相等，而且不存在单调的映射关系，例如机器翻译，对话系统等等。为了解决这个问题，作者提出了RNN Encoder-Decoder模型，RNN Encoder-Decoder是由两个RNN模型级联而成的，通过Encoder将输入数据编码成特征向量，再通过Decoder将特征向量解码成输出数据。

这篇论文的第二个贡献就是GRU\(Gated Recurrent Unit\)的提出，GRU和LSTM均是采用门机制的思想改造RNN的神经元，和LSTM相比，GRU更加简单，高效，且不容易过拟合，但有时候在更加复杂的场景中效果不如LSTM，算是RNN和LSTM在速度和精度上的一个折中方案。

论文的实现是对SMT中短语表的rescore，即使用MOSES（SMT的一个开源工具）根据平行语聊产生短语表，使用GRU的RNN Encoder-Decoder对短语表中的短语对进行重新打分。

## 详解

### 1. RNN Encoder-Decoder

给定训练集D=\(X,Y\)，我们希望最大化输出标签的条件概率，即：

```
p(y_1, y_2, ..., y_T' | x_1, x2, ..., X_T)
```

在上式中，T \neq T'。

#### 1.1 编码

RNN Encoder-Decoder的编码过程是先是通过一个RNN将变长的输入序列转换成固定长度的特征向量，再通过RNN将特征向量解码成需要的输出，如图1.

\[GRU\_1\]

输入序列是一个标准的RNN，在计算时间片t的输出h\_&lt;t&gt;时，将h\_&lt;t-1&gt;和x\_&lt;t&gt;输入激活函数中，表示为

```
h_<t> = f(h_<t-1>, x_t)
```

经过T个时间片后，得到一个h\*1的特征向量$$\mathbf{c}$$，其中h是隐层节点的节点数。

#### 1.2 解码

RNN Encoder-Decoder的解码过程是另外一个RNN，解码器的作用是将特征向量c，前一个时间片的输出y\__&lt;t-1&gt;，以及前一个隐层节点h\_\_&lt;t-1&gt;作为输入，得到h_\_&lt;t&gt;，表示为

```
h_<t> = f(h_<t-1>, y_<t-1>, \mathbf{c})
```

其中，y\_&lt;t&gt;也是关于c，y\__&lt;t-1&gt;以及h\__&lt;t-1&gt;的条件分布

```
y_<t> = P(y_t | y_{t-1}, y_{t-2}, ..., y_1, \mathbf{c}) =  g(h_<t>, y_<t-1>, \mathbf{c})
```

RNN Encoder-Decoder的优化便是最大化p\(y\|x\)的log条件似然

```
\mathop{\max}_\mathbf{\theta} \frac{1}{N} \sum^{N}_{n=1}logp_\mathbf{\theta} (\mathbf{y}_n|\mathbf{x}_n)
```

其中\mathbf{\theta}是编码解码器的所有参数。

RNN Encoder-Decoder不仅可以用于产生输出数据，根据训练好的模型，使用条件概率模型p\_\mathbf{\theta} \(\mathbf{y}\_n\|\mathbf{x}\_n\)，也可以对现有的的标签进行评分。在这篇论文的实验中，作者便是对SMT中的短语表进行了重新打分。

# 2. GRU

对于短语rescore这个任务，作者最先使用的是RNN模型，遗憾的是RNN表现并不是非常理想。根据LSTM中提出的门机制的思想，作者提出了一种更简单，更高效且更不容易过拟合的GRU，图2便是GRU的结构。

\[GRU\_2\]

  



