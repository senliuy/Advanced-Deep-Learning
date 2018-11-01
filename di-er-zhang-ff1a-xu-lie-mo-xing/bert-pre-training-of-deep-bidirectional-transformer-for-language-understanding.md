# [^1]BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

## 前言

接触NLP领域不久，也来凑热点学习一下BERT，文章中可能存在些许错误，权当学习笔记了。

BERT近期提出之后，作为一个Word2Vec的替代者，其在NLP领域的11个方向大幅刷新了精度，可以说是近年来自残差网络最优突破性的一项技术了。仔细研读了一下文章，论文的主要特点以下几点：

1. 使用了Transformer \[2\]作为算法的主要框架；
2. 使用了Mask Language Model \[3\] 和 Next Sentence Prediction 的多任务训练目标；
3. 更彻底的捕捉语句中的双向关系；
4. 使用更强大的机器训练更大规模的数据，使BERT的结果达到了全新的高度，并且Google开源了BERT模型，用户可以直接使用BERT作为Word2Vec的转换矩阵并高效的将其应用到自己的任务中。

## 1. BERT 详解

### 1.1 网络架构

BERT的网络架构使用的是《Attention is all you need》中提出的多层双向Transformer[^1]结构，Transformer的结构在NLP领域中已经得到了广泛应用，并且作者已经发布在TensorFlow的[tensor2tensor](https://github.com/tensorflow/tensor2tensor)库中。

一个Transformer节点的结构如图1所示。


## Reference

\[1\] Devlin J, Chang M W, Lee K, et al. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding\[J\]. arXiv preprint arXiv:1810.04805, 2018.

\[2\] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need \[C\]//Advances in Neural Information Processing Systems. 2017: 5998-6008.

\[3\] Wilson L Taylor. 1953. cloze procedure: A new tool for measuring readability. Journalism Bulletin, 30\(4\):415–433.



[^1]: Enter footnote here.待删：此处先理解为神经网络的一个复杂的节点，Transformer的详细内容会在近期给出分析。

