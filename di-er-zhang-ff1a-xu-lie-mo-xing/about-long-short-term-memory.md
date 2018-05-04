# About Long Short Term Memory

## 1. 背景

### Recurrent Neural Networks \(RNN\)

在使用深度学习处理时序问题时，RNN是最常使用的模型之一。RNN之所以在时序数据上有着优异的表现是因为RNN在t时间片时会将t-1时间片的隐节点作为当前时间片的输出，也就是RNN具有图1的结构。这样有效的原因是之前时间片的信息也用于计算当前时间片的内容，而传统模型的隐节点的输出只取决于当前时间片的输入特征。

\[LSTM\_1.png\]

RNN的数学表达式可以表示为

```
h_t = \sigma(x_t*w_{xt} + h_{t-1} * w_{ht} + b)
```

而传统的DNN的隐节点表示为

```
h_t = \sigma(x_t*w_{xt} + b)
```

RNN的该特性也使RNN在很多学术和工业前景，例如OCR，语音识别，股票预测等领域上有了十足的进展。

### 长期依赖\(Long Term Dependencies\)

在深度学习领域中（尤其是RNN），“长期依赖“问题是普遍存在的。长期依赖产生的原因是当神经网络的节点经过许多阶段的计算后，之前比较长的时间片的特征已经被覆盖，例如下面例子

```
eg1: The cat, which already ate a bunch of food, was full.
      |   |     |      |     |  |   |   |   |     |   |
     t0  t1    t2      t3    t4 t5  t6  t7  t8    t9 t10
eg2: The cats, which already ate a bunch of food, were full.
      |   |      |      |     |  |   |   |   |     |    |
     t0  t1     t2     t3    t4 t5  t6  t7  t8    t9   t10
```

我们想预测'full'之前系动词的单复数情况，显然full是取决于第二个单词’cat‘的单复数情况，而非其前面的单词food。根据图1展示的RNN的结构，随着数据时间片的增加，RNN丧失了学习连接如此远的信息的能力（图2）。

\[LSTM\_2.png\]

## 2. LSTM

LSTM的全称是Long Short Term Memory，顾名思义，它具有记忆长短期信息的能力的神经网络。LSTM首先在1997年由Hochreiter & Schmidhuber \[1\] 提出，由于深度学习在2012年的兴起，LSTM又经过了若干代大牛[^1]的发展，由此便形成了比较系统且完整的LSTM框架，并且在很多领域得到了广泛的应用。本文着重介绍深度学习时代的LSTM。

LSTM提出的动机是为了解决上面我们提到的长期依赖问题。传统的RNN节点输出仅由权值，偏置以及激活函数决定（图3）。RNN是一个链式结构，每个时间片使用的是相同的参数。

\[LSTM3\]

而LSTM之所以能够解决RNN的长期依赖问题，是因为LSTM引入了门（gate）机制用于控制特征的流通和损失。对于上面的例子，LSTM可以做到在t9时刻将t2时刻的特征传过来，这样就可以非常有效的判断t9时刻使用单数还是负数了。LSTM是由一系列LSTM单元（LSTM Unit）组成，其链式结构如下图。

\[LSTM4\]

在后面的章节中我们再对LSTM的详细结构进行讲解，首先我们先弄明白LSTM单元中的每个符号的含义。每个黄色方框表示一个神经网络层，由权值，偏置以及激活函数组成；每个粉色圆圈表示元素级别操作，$$$$\otimes是LSTM最重要的门机制，是个0到1之间的小数；箭头表示向量流向；相交的箭头表示向量的拼接；分叉的箭头表示向量的复制。总结如图5.

\[LSTM5\]

LSTM的核心部分是在图4中最上边类似于传送带的部分（图6），这一部分一般叫做单元状态（cell state）它自始至终存在于LSTM的整个链式系统中。

\[LSTM\_6\]

其中

```
C_t = f_t * C_{t-1} + i_t * \tilde{C}_t
```

其中f\_t叫做遗忘门，f\_t是个位于\[0, 1\]直接的一个标量，由数据X\_t和隐节点h\_{t-1}经由一个神经网络层得到（图7）。通常我们使用Sigmoid作为激活函数

\[LSTM\_7\]



# reference

\[1\] Hochreiter, S, and J. Schmidhuber. “Long short-term memory.” Neural Computation 9.8\(1997\):1735-1780.

[^1]: Felix Gers, Fred Cummins, Santiago Fernandez, Justin Bayer, Daan Wierstra, Julian Togelius, Faustino Gomez, Matteo Gagliolo, and Alex Gloves

