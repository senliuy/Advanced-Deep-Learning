# Arbitrary-Oriented Scene Text Detection via Rotation Proposals

## 前言

在场景文字检测中一个最常见的问题便是倾斜文本的检测，现在基于候选区域的场景文字检测方法，例如CTPN，DeepText等，其检测框均是与坐标轴平行的矩形区域，其根本原因在于数据的标签采用了$$(x,y,w,h)$$的形式。另外一种方法是基于语义分割，例如HMCP，EAST等，但是基于分割算法的场景文字检测效率较低且并不擅长检测长序列文本。

作者提出的RRPN（Rotation Region Proposal Network）可以归结到基于候选区域的类别当中，算法的主要贡献是提出了带旋转角度的锚点，并锚点的角度特征重新设计了IoU，NMS以及ROI池化等算法，RRPN的角度特征使其非常适合对倾斜文本进行检测。

RRPN的这个特征使其不仅可以应用到场景文字检测，在一些存在明显角度特征的场景中，例如建筑物检测，也非常适用。
## 1.RPN回顾

关于RPN的详细内容可参考[Faster R-CNN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.html)一文，在这里我们只进行简单的回顾。

RPN是一个全卷积网络，其首先通过3个尺寸，3个尺度的锚点在Feature Map上对输入图像进行密集采样。然后通过一个由判断锚点是前景还是背景的二分类任务和一个用于预测锚点和Ground Truth的位置相对距离的回归模型组成。

RPN的一个位置的特征向量采样$$3\times3 = 9$$个锚点，每个锚点的损失函数由分类任务（2）和回归任务（4）组成，因此一个特征向量有$$9\times6=54$$个输出，RPN的损失函数可以表示为：

$$
L({p_i},{t_i})=\frac{1}{N_{cls}}\sum_{i}L_{cls}(p_i,p_i^*)+\lambda\frac{1}{N_{reg}}\sum_{i}p_i^*L_{reg}(t_i,t_i^*)
$$

其中$$L_{cls}$$是分类任务，损失函数是$$softmax$$，用于计算该锚点为前景或者背景的概率；$$L_{reg}$$是回归任务，损失韩式是Smooth L1，用于计算锚点和Ground Truth的相对关系。
## Reference

\[1\] Ma J, Shao W, Ye H, et al. Arbitrary-oriented scene text detection via rotation proposals\[J\]. IEEE Transactions on Multimedia, 2018.

