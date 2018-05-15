# Deep Speech: Scaling up end-to-end speech recognition

## 简介

百度的第一篇使用深度学习解决语音识别问题的文章。文章使用了DNN+RNN+CTC的网络结构，在模型以及算法方向并没有创新性，但是文章的数据扩充，和GPU的性能优化仍然十分具有参考性。

## Deep Speech结构

Deep Speech的结构如图1

###### 图1：Deep Speech网络结构

\[Deep\_Speech1\_1.png\]

网络含有5个隐层节点，其中前三层是全连接层，第四层使用的是双向RNN。最后一层使用的是softmax激活函数。

