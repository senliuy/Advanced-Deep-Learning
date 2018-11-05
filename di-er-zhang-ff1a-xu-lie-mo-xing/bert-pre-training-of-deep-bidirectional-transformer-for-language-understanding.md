# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## 前言

BERT（**B**idirectional **E**ncoder **R**epresentations from **T**ransformers）近期提出之后，作为一个Word2Vec的替代者，其在NLP领域的11个方向大幅刷新了精度，可以说是近年来自残差网络最优突破性的一项技术了。论文的主要特点以下几点：

1. 使用了Transformer \[2\]作为算法的主要框架，Trabsformer能更彻底的捕捉语句中的双向关系；
2. 使用了Mask Language Model\(MLM\) \[3\] 和 Next Sentence Prediction\(NSP\) 的多任务训练目标；
3. 使用更强大的机器训练更大规模的数据，使BERT的结果达到了全新的高度，并且Google开源了BERT模型，用户可以直接使用BERT作为Word2Vec的转换矩阵并高效的将其应用到自己的任务中。

BERT的源码和模型近期已经在Github上[开源](https://github.com/google-research/bert)。

## 1. BERT 详解

### 1.1 网络架构

BERT的网络架构使用的是《Attention is all you need》中提出的多层Transformer结构，其最大的特点是抛弃了传统的RNN和CNN，通过Attention机制将任意位置的两个单词的距离转换成1，有效的解决了NLP中棘手的长期依赖问题。Transformer的结构在NLP领域中已经得到了广泛应用，并且作者已经发布在TensorFlow的[tensor2tensor](https://github.com/tensorflow/tensor2tensor)库中。

Transformer的网络架构如图1所示，Transformer是一个encoder-decoder的结构，由若干个编码器和解码器堆叠形成。图1的左侧部分为编码器，由Multi-Head Attention和一个全连接组成，用于将输入语料转化成特征向量。右侧部分是解码器，其输入为编码器的输出以及已经预测的结果，由Masked Multi-Head Attention, Multi-Head Attention以及一个全连接组成，用于输出最后结果的条件概率。关于Transformer的详细解析参考我之前总结的[文档](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/attention-is-all-you-need.html)。

<figure>
<img src="/assets/BERT_1.png" alt="图1：BERT 中采用的Transformer网络" />
<figcaption>图1：BERT 中采用的Transformer网络</figcaption>
</figure>


图1中的左侧部分是一个Transformer Block，对应到图2中的一个“Trm”。

<figure>
<img src="/assets/BERT_2.png" alt="图2：BERT的网络结构" />
<figcaption>图2：BERT的网络结构</figcaption>
</figure>


BERT提供了简单和复杂两个模型，对应的超参数分别如下：

* $$\mathbf{BERT}_{\mathbf{BASE}}$$: L=12，H=768，A=12，参数总量110M；

* $$\mathbf{BERT}_{\mathbf{LARGE}}$$: L=24，H=1024，A=16，参数总量340M；

在上面的超参数中，L表示网络的层数（即Transformer blocks的数量），A表示Multi-Head Attention中self-Attention的数量，filter的尺寸是4H。

论文中还对比了BERT和GPT\[4\]和ELMo\[5\]，它们两个的结构图如图3所示。

<figure>
<img src="/assets/BERT_3.png" alt="图3：OpenAI GPT和ELMo" />
<figcaption>图3：OpenAI GPT和ELMo</figcaption>
</figure>

BERT对比这两个算法的优点是只有BERT表征会**基于所有层中的左右两侧语境**。BERT能做到这一点得益于Transformer中Attention机制将任意位置的两个单词的距离转换成了1。

### 1.2 输入表示

BERT的输入的编码向量（长度是512）是3个嵌入特征的单位和，如图4，这三个词嵌入特征是：

1. WordPiece 嵌入\[6\]：WordPiece是指将单词划分成一组有限的公共子词单元，能在单词的有效性和字符的灵活性之间取得一个折中的平衡。例如图4的示例中‘playing’被拆分成了‘play’和‘ing’；
2. 位置嵌入（Position Embedding）：位置嵌入是指将单词的位置信息编码成特征向量，位置嵌入是向模型中引入单词位置关系的至关重要的一环。位置嵌入的具体内容参考我之前的[分析](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/attention-is-all-you-need.html)；
3. 分割嵌入（Segment Embedding）：用于区分两个句子，例如B是否是A的下文（对话场景，问答场景等）。对于句子对，第一个句子的特征值是0，第二个句子的特征值是1。


## Reference

\[1\] Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\[J\]. arXiv preprint arXiv:1810.04805, 2018.

\[2\] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need \[C\]//Advances in Neural Information Processing Systems. 2017: 5998-6008.

\[3\] Wilson L Taylor. 1953. cloze procedure: A new tool for measuring readability. Journalism Bulletin, 30\(4\):415–433.

\[4\] Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding with unsupervised learning. Technical report, OpenAI.

\[5\] Matthew Peters, Waleed Ammar, Chandra Bhagavatula, and Russell Power. 2017. Semi-supervised sequence tagging with bidirectional language models. In ACL.

\[6\] Yonghui Wu, Mike Schuster, Zhifeng Chen, Quoc V Le, Mohammad Norouzi, Wolfgang Macherey, Maxim Krikun, Yuan Cao, Qin Gao, Klaus Macherey, et al. 2016. Google’s neural machine translation system: Bridging the gap between human and machine translation. arXiv:1609.08144.

