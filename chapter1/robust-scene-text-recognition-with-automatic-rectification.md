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

Localization network是一个有卷积层，池化层和全连接构成的卷积网络（图4）。由于一个点由$$(x,y)$$定义，所以一个要预测$$K$$个基准点的卷积网络需要由$$2K$$个输出。为了将基准点的范围控制到$$[-1,1]$$，输出层使用$$tanh$$作为激活函数，在论文的实验部分给出$$K=20$$。如图2和图3所示的绿色'+'即为Localization network预测的基准点。

得到网络的输出后，其被reshape成一个$$2\times K$$的矩阵$$\mathbf{C}$$，即$$\mathbf{C} = [\mathbf{c}_1, \mathbf{c}_2, ..., \mathbf{c}_K] \in \mathfrak{R}^{2\times K}$$。

###### 图4: Localization network的结构

![](/assets/RARE_4.png)

### 1.1.2 Grid Generator

当给定了输出Feature Map的时候，我们可以再其顶边和底边分别均匀的生成$$K$$个点，如图5，这些点便被叫做基-基准点（base fiducial point），表示为$$\mathbf{C'} = [\mathbf{c'}_1, \mathbf{c'}_2, ..., \mathbf{c'}_K] \in \mathfrak{R}^{2\times K}$$，在RARE的STN中，输出的Feature Map的尺寸是固定的，所以$$\mathbf{C'}$$为一个常量。

那么，1.1.2节要介绍的Grid Generator的作用就是如何利用$$C$$和$$C'$$，将图5中的图像$$I$$和图5中的图像$$I'$$的转换关系$$T$$。

###### 图5：TPS中的转换关系

![](/assets/RARE_5.png)

当从localization network得到基准点$$\mathbf{C}$$和固定基-基准点$$\mathbf{C}'$$后，转换矩阵$$T\in\mathfrak{R}^{2\times(K+3)}$$的值已经可以确定：

$$
\mathbf{T} = \left(\Delta^{-1}_{\mathbf{C}'}
\left[
\begin{matrix}
\mathbf{C}^T \\
\mathbf{0}^{3\times2}
\end{matrix}
\right]
\right)^T
$$

其中$$\Delta_{\mathbf{C}'} \in \mathfrak{R}^{(K+3)\times{K+3}}$$是一个只由$$\mathbf{C}'$$计算得到的矩阵:

$$
\Delta_{C'} = 
\left[
\begin{matrix}
\mathbf{1}^{K\times1} & \mathbf{C'}^{T} & \mathbf{R} \\
\mathbf{0} & \mathbf{0} & \mathbf{1}^{1\times K} \\
\mathbf{0} & \mathbf{0} & \mathbf{C}'\\
\end{matrix}
\right]
$$

其中$$\mathbf{1}^{K\times1}$$是一个$$K\times1$$的值全是$$1$$的行向量，$$\mathbf{1}^{1\times K}$$同理。$$\mathbf{R}\in \mathfrak{R}^{K\times K}$$是一个由$$r_{i,j}$$组成的$$K\times K$$的矩阵。其中

$$
r_{i,j} = d^2_{i,j}ln(d^2_{i,j}) \\
d_{i,j} = euclidean(c'_i, c'_j)
$$

上式中$$euclidean(a,b)$$表示$$a,b$$两点之间的欧式距离。

由此可见，仅仅使用$$C$$和$$C'$$我们便可以得到转换矩阵$$\mathbf{T}$$。那么对于STN这个反向插值的算法来说，对于矫正图片中$$I' = \{\mathbf{p'_i}\}_{i=1,2,...,N}$$（$$N = W\times H$$, 即输出图像的像素点的个数）的任意一点$$\mathbf{p}'_i = [x'_i, y'_i]^T$$，我们怎样才能找到其在原图$$I$$中对应的点$$\mathbf{p}_i = [x_i, y_i]^T$$呢？这就需要用到我们上面得到的$$\mathbf{T}$$了。

$$
r'_{i,k} = d^2_{i,k} ln(d^2_{i,k}) \\
\hat{\mathbf{p}}'_i = [1, x'_i, y'_i, r'_{i,1}, r'_{i,2}, ..., r'_{i,3}] \\
\mathbf{p}_i = \mathbf{T}\hat{\mathbf{p}}'_i
$$

其中$$d^2_{i,k}$$表示第$$i$$个像素点$$\mathbf{p}'_i$$和第$$k$$个基准点$$c'_k$$之间的欧式距离。

### 1.1.3 Sampler

在1.1.2节中，我们得到了输出Feature Map上一点$$\mathbf{p}'_i = [x'_i, y'_i]^T$$对应的输入Feature Map上像素点的坐标$$\mathbf{p}_i = [x_i, y_i]^T$$的对应关系。在RARE中，使用了双线西插值得到了输出Feature Map在$$[x'_i, y'_i]$$上的值。

RARE中的STN和原始版本的STN都是一个可微分的模型，这也就意味着RARE也是一个可端到端训练的模型。不同点在于RARE将仿射变换矩阵变成了TPS，从而使模型有拥有矫正任何变换的能力，包括但不仅限于仿射变换，图2右侧部分是RARE的STN得到的实验结果。

## 1.2 Sequence Recognition Network

如图1的后半部分所示，RARE的SRN的输入是1.1节得到的校正后的图片，输出则是识别的字符串。SRN是一个基于Attention的序列到序列（Seq-to-Seq）的模型，包含编码器（Encoder）和解码器（Decoder）两部分，编码器用于将输入图像$$I'$$编码成特征向量$$\mathbf{h}$$，解码器则负责将特征向量$$\mathbf{h}$$解码成字符串$$\hat{\mathbf{y}}$$。

### 1.2.1 编码器（Encoder）

## Reference

\[1\] Shi B, Wang X, Lyu P, et al. Robust scene text recognition with automatic rectification\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 4168-4176.

\[2\] Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks\[C\]//Advances in neural information processing systems. 2015: 2017-2025.

\[3\] F. L. Bookstein. Principal warps: Thin-plate splines and the decomposition of deformations.IEEE Trans. Pattern Anal. Mach. Intell., 11\(6\):567–585, 1989.

\[4\] [https://en.wikipedia.org/wiki/Thin\_plate\_spline](https://en.wikipedia.org/wiki/Thin_plate_spline)

\[5\] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate\[J\]. arXiv preprint arXiv:1409.0473, 2014.

