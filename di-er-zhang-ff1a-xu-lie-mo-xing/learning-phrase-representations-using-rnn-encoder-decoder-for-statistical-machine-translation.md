# Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation

简介

在很多时序分类（Temporal Classification\)的应用中，输入数据X和输出数据Y的标签长度并不相等，而且不存在单调的映射关系，例如机器翻译，对话系统等等。为了解决这个问题，作者提出了RNN Encoder-Decoder模型，RNN Encoder-Decoder是由两个RNN模型级联而成的，通过Encoder将输入数据编码成特征向量，再通过Decoder将特征向量解码成输出数据。

这篇论文的第二个贡献就是GRU\(Gated Recurrent Unit\)的提出，GRU和LSTM均是采用门机制的思想改造RNN的神经元，和LSTM相比，GRU更加简单，高效，且不容易过拟合，但有时候在更加复杂的场景中效果不如LSTM，算是RNN和LSTM在速度和精度上的一个折中方案。

论文的实现是对SMT中短语表的rescore，即使用MOSES（SMT的一个开源工具）根据平行语聊产生短语表，使用GRU的RNN Encoder-Decoder对短语表中的短语对进行重新打分。

详解



