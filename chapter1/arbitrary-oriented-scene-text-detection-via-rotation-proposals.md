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


## 2.RRPN详解

### 2.1. RRPN网络结构

RRPN的网络结构如图1所示，检测过程可以分成三步：

1. 使用卷积网络产生Feature Map，论文中使用的是[VGG-16](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/very-deep-convolutional-networks-for-large-scale-image-recognition.html)，也可以替换成物体检测的主流框架，例如基于[残差网络](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)的[FPN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/mask-r-cnn.html)；

2. 使用RRPN产生带角度的候选区域；
3. 使用RRoI Pooling产生长度固定的特征向量，之后接两层全连接用于候选区域的类别精校。

<figure>
<img src="/assets/RRPN_1.png" alt="图1：RRPN网络结构图" />
<figcaption>图1：RRPN网络结构图</figcaption>
</figure>

### 2.2 R-Anchor

传统的RPN的锚点均是与坐标轴平行的矩形，而RRPN中添加了角度信息，我们将这样的锚点叫做R-Anchor。R-Anchor由$$(x,y,w,h,\theta)$$五要素组成，其中$$(x,y)$$表示bounding box的几何中心（RPN中是左上角）。$$(w,h)$$分别是bounding box的长边和短边。$$\theta$$是锚点的旋转角度，通过$$\theta+k\pi$$将$$\theta$$的范围控制在$$[-\frac{\pi}{4},\frac{3\pi}{4})$$。

对比另外一种用4个点$$(x_1,y_1,x_2,y_2,x_3,y_3,x_4,y_4)$$表示任意四边形的策略相比，R-Anchor有以下3条优点：

1. 两个四边形的相对角度更好计算；
2. 回归的值更少，模型更好训练；
3. 更容易进行图像扩充（2.3节）。

R-Anchor的锚点由3个尺寸，3个比例以及6个角度组成：3个尺寸分别是8，16，32；3个比例分别是$$1:2$$，$$1:5$$，$$1:8$$；6个角度分别是$$-\frac{\pi}{6}, 0, \frac{\pi}{6},\frac{\pi}{3},\frac{\pi}{2},\frac{2\pi}{3}$$。锚点的形状如图2所示。因此在RRPN中每个特征向量共有$$3\times3\times6=54$$个锚点。

<figure>
<img src="/assets/RRPN_2.jpeg" alt="图2：RRPN的锚点" />
<figcaption>图2：RRPN的锚点</figcaption>
</figure>

### 2.3 RRPN的图像扩充

为了缓解过拟合的问题，并增加模型对选择区域的检测能力，RRPN使用了数据扩充的方法增加样本的数量。RRPN使用的扩充方法之一是将输入图像选择$$\alpha$$。

对于一张尺寸为$$I_w\times I_h$$的输入图像，设其中一个Ground Truth表示为$$(x,y,w,h,\theta)$$，旋转$$\alpha$$后得到的Ground Truth为$$(x',y',w',h',\theta')$$，其中Ground Truth的尺寸并不会改变，即$$w'=w$$，$$h'=h$$。$$\theta'=\theta+\alpha+k\pi$$，$$k\pi$$用于将$$\theta'$$的范围控制到$$[-\frac{\pi}{4},\frac{3\pi}{4})$$之间。$$(x',y')$$的计算方式为：

$$
\begin{array}
\left[
x'\\y'\\1
\right]
\end{array}=
$$

## Reference

\[1\] Ma J, Shao W, Ye H, et al. Arbitrary-oriented scene text detection via rotation proposals\[J\]. IEEE Transactions on Multimedia, 2018.

