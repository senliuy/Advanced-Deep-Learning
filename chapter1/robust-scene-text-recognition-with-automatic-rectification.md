# Robust Scene Text Recognition with Automatic Rectification

tags: RARE, OCR, STN, TPS, Attention

## 前言

RARE实现了对不规则文本的end-to-end的识别，算法包括两部分：

1. 基于[STN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/spatial-transform-networks.html)\[2\]的不规则文本区域的矫正：与STN不同的是，RARE在Localisation部分预测的并不是仿射变换矩阵，而是K个TPS（Thin Plate Spines）\[3\]\[4\]的基准点，其中TPS基于样条（spines）的插值和平滑技术，在1.1节中会详细介绍其在RARE中的计算过程。
2. 基于SRN的文字识别：SRN（Sequence Recognition Network）是基于Attention \[5\]的序列模型，包括有CNN和[LSTM](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/about-long-short-term-memory.html)构成的编码（Encoder）模块和基于[Attention](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/neural-machine-translation-by-jointly-learning-to-align-and-translate.html)和LSTM的解码（Decoder）模块构成，此部分会在1.2节介绍。

在测试阶段，RARE使用了基于贪心或Beam Search的方法寻找最优输出结果。

RARE的流程如图1。

###### 图1：RARE的算法框架



## 1. RARE详解

### 1.1 Spatial Transformer Network

场景文字检测的难点有很多，仿射变换是众多变换中的一种（图1.(a)）

## Reference

\[1\] Shi B, Wang X, Lyu P, et al. Robust scene text recognition with automatic rectification\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 4168-4176.

\[2\] Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks\[C\]//Advances in neural information processing systems. 2015: 2017-2025.

\[3\] F. L. Bookstein. Principal warps: Thin-plate splines and the decomposition of deformations.IEEE Trans. Pattern Anal. Mach. Intell., 11\(6\):567–585, 1989.

\[4\] [https://en.wikipedia.org/wiki/Thin\_plate\_spline](https://en.wikipedia.org/wiki/Thin_plate_spline)

\[5\] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate\[J\]. arXiv preprint arXiv:1409.0473, 2014.

