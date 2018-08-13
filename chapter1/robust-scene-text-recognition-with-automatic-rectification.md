# Robust Scene Text Recognition with Automatic Rectification

tags: RARE, OCR, STN, TPS, Attention

## 前言

RARE实现了对不规则文本的end-to-end的识别，算法包括两部分：

1. 基于[STN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/spatial-transform-networks.html)\[2\]的不规则文本区域的矫正：与STN不同的是，RARE在Localisation部分预测的并不是仿射变换矩阵，而是K个TPS（Thin Plate Spines）\[3\]\[4\]的基准点，其中TPS基于样条（spines）的数据插值和平滑技术，在1.1节中会详细介绍其在RARE中的计算过程。
2. 基于SRN的文字识别：SRN（Sequence Recognition Network）是基于[Attention](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/neural-machine-translation-by-jointly-learning-to-align-and-translate.html) \[5\]的序列模型，包括有CNN和[LSTM](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/about-long-short-term-memory.html)构成的编码（Encoder）模块和基于Attention和LSTM的解码（Decoder）模块构成，此部分会在1.2节介绍。

在测试阶段，RARE使用了基于贪心或Beam Search的方法寻找最优输出结果。

RARE的流程如图1。

###### 图1：RARE的算法框架，其中实线表示预测流程，虚线表示反向迭代过程。

![](/assets/RARE_1.png)

## 1. RARE详解

### 1.1 Spatial Transformer Network

场景文字检测的难点有很多，仿射变换是其中一种，Jaderberg\[2\]等人提出的STN通过预测仿射变换矩阵的方式对输入图像进行矫正。但是真实场景的不规则文本要复杂的多，可能包括扭曲，弧形排列等情况（图2）,这种方式的变换是传统的STN解决不了的，因此作者提出了基于TPS的STN。TPS非常强大的一点在于其可以近似所有和生物有关的形变。

###### 图2：自然场景中的变换，左侧是输入图像，右侧是矫正后的效果，其中涉及的变换包括：\(a\) loosely-bounded text; \(b\) multi-oriented text; \(c\) perspective text; \(d\) curved text.

![](/assets/RARE_2.png)

TPS是一种基于样条的数据插值和平滑技术。要详细了解STN的细节和动机，可以自行去看论文，我暂无计划解析这篇1989年提出的和深度学习关系不大的论文。对于TPS可以这么简单理解，给我们一块光滑的薄铁板，我们弯曲这块铁板使其穿过空间中固定的几个点，TPS得到的便是我们弯曲铁板所耗费的最小的功。

TPS也常用于对扭曲图像的矫正，1.1.2节中介绍计算流程，至于为什么能work，我也暂时没有搞懂。

纵观整个矫正算法，RARE的STN也分成3部分：

1. **localization network**: 预测TPS矫正所需要的$$K$$个基准点（fiducial point）；
2. **Grid Generator**：基于基准点进行TPS变换，生成输出Feature Map的采样窗格（Grid）；
3. **Sampler**：每个Grid执行双线性插值。

STN的算法流程如图3。

###### 图3：RARE中的STN

![](/assets/RARE_3.png)

#### 1.1.1 localization network

Localization network是一个有卷积层，池化层和全连接构成的卷积网络（图4）。由于一个点由$$(x,y)$$定义，所以一个要预测$$K$$个基准点的卷积网络需要由$$2K$$个输出。为了将基准点的范围控制到$$[-1,1]$$，输出层使用$$tanh$$作为激活函数。如图3所示，其中绿色点纪委Localization network预测的基准点。

得到网络的输出后，其被reshape成一个$$2\times K$$的矩阵$$\mathbf{C}$$，即$$\mathbf{C} = [\mathbf{c}_1, \mathbf{c}_2, ..., \mathbf{c}_K] \in \mathfrak{R}^{2\times K}$$。 

###### 图4: Localization network的结构
![](/assets/RARE_4.png)

## Reference

\[1\] Shi B, Wang X, Lyu P, et al. Robust scene text recognition with automatic rectification\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 4168-4176.

\[2\] Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks\[C\]//Advances in neural information processing systems. 2015: 2017-2025.

\[3\] F. L. Bookstein. Principal warps: Thin-plate splines and the decomposition of deformations.IEEE Trans. Pattern Anal. Mach. Intell., 11\(6\):567–585, 1989.

\[4\] [https://en.wikipedia.org/wiki/Thin\_plate\_spline](https://en.wikipedia.org/wiki/Thin_plate_spline)

\[5\] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate\[J\]. arXiv preprint arXiv:1409.0473, 2014.

