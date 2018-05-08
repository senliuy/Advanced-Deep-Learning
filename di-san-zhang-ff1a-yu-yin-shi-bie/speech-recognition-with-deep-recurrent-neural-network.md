# Speech Recognition with Deep Recurrent Neural Network

## 介绍

语音识别是具有巨大市场和研究潜力的一个领域，语音识别已经有了几十年的研究历史了。2000年前，涌现了大量的语音识别技术，例如：混合高斯模型（GMM），隐马尔科夫模型（HMM），梅尔倒谱系数（MFCC），n元祖语言模型（n-gram LM）等等（图1）。在21世纪的第一个十年，这些技术被成功应用到实际系统中。但同时，语音识别的技术仿佛遇到了瓶颈期，不论科研进展还是实际应用均进展非常缓慢。

2012年，深度学习兴起。仅仅一年之后，Hinton的著名学子Alex Graves的这篇使用深度学习思想解决语音识别问题的这篇文章引起了广泛关注，为语音识别开辟了新的研究方向，这篇文章可以说是目前所有深度学习解决语音识别方向的奠基性文章了。目前深度学习均采用和其类似的RNN+CTC的框架，甚至在OCR领域也是采用了同样的思路。

###### 图1: 语音识别传统模型

## 算法细节

在这篇论文涉及的实验里，使用了MFCC提取音频特征，多层双向RNN \[2\] 编码特征（节点使用LSTM），CTC构建声学模型。由于CTC没有构建语音模型的能力，论文使用了RNN Transducer联合训练声学模型和语言模型。结构如图2。

###### 图2：基于深度学习的语音识别架构



## Reference

\[1\] Graves A, Mohamed A, Hinton G. Speech recognition with deep recurrent neural networks\[C\]//Acoustics, speech and signal processing \(icassp\), 2013 ieee international conference on. IEEE, 2013: 6645-6649.

\[2\] M. Schuster and K. K. Paliwal, “Bidirectional Recurrent Neural Networks,” IEEE Transactions on Signal Processing, vol. 45, pp. 2673–2681, 1997.

