# Arbitrary-Oriented Scene Text Detection via Rotation Proposals

tags: OCR, RRPN, Faster R-CNN

## 前言

在场景文字检测中一个最常见的问题便是倾斜文本的检测，现在基于候选区域的场景文字检测方法，例如[CTPN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/detecting-text-in-natural-image-with-connectionist-text-proposal-network.html)\[2\]，[DeepText](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/deeptext-a-unified-framework-for-text-proposal-generation-and-text-detection-in-natural-images.html)\[3\]等，其检测框均是与坐标轴平行的矩形区域，其根本原因在于数据的标签采用了$$(x,y,w,h)$$的形式。另外一种方法是基于语义分割，例如[HMCP](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/scene-text-detection-via-holistic-multi-channel-prediction.html)\[4\]，[EAST](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/east-an-efficient-and-accurate-scene-text-detector.html)\[5\]等，但是基于分割算法的场景文字检测效率较低且并不擅长检测长序列文本。

作者提出的RRPN（Rotation Region Proposal Network）可以归结到基于候选区域的类别当中，算法的主要贡献是提出了带旋转角度的锚点，并锚点的角度特征重新设计了IoU，NMS以及ROI池化等算法，RRPN的角度特征使其非常适合对倾斜文本进行检测。

RRPN的这个特征使其不仅可以应用到场景文字检测，在一些存在明显角度特征的场景中，例如建筑物检测，也非常适用。

## 1.RPN回顾

关于RPN的详细内容可参考[Faster R-CNN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.html)\[6\]一文，在这里我们只进行简单的回顾。

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

### 2.2. R-Anchor

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

### 2.3. RRPN的图像扩充

为了缓解过拟合的问题，并增加模型对选择区域的检测能力，RRPN使用了数据扩充的方法增加样本的数量。RRPN使用的扩充方法之一是将输入图像选择$$\alpha$$。

对于一张尺寸为$$I_W\times I_H$$的输入图像，设其中一个Ground Truth表示为$$(x,y,w,h,\theta)$$，旋转$$\alpha$$后得到的Ground Truth为$$(x',y',w',h',\theta')$$，其中Ground Truth的尺寸并不会改变，即$$w'=w$$，$$h'=h$$。$$\theta'=\theta+\alpha+k\pi$$，$$k\pi$$用于将$$\theta'$$的范围控制到$$[-\frac{\pi}{4},\frac{3\pi}{4})$$之间。$$(x',y')$$的计算方式为：

$$
\left[\begin{matrix}
x'\\y'\\1
\end{matrix}\right]=
\mathbf{T}(\frac{I_W}{2}, \frac{I_H}{2})
\mathbf{R}(\alpha)
\mathbf{T}(-\frac{I_W}{2},-\frac{I_H}{2})
\left[\begin{matrix}
x\\y\\1
\end{matrix}\right]
$$

$$\mathbf{T}(\delta_x, \delta_y)$$和$$\mathbf{R}(\alpha)$$的定义分别是：

$$
\mathbf{T}(\delta_x, \delta_y)=
\left[\begin{matrix}
1 & 0 & \delta_x \\
0 & 1 & \delta_y \\
0 & 0 & 1
\end{matrix}\right]
$$

$$
\mathbf{R}(\alpha)=
\left[\begin{matrix}
\text{cos }\alpha & \text{sin }\alpha & 0 \\
-\text{sin }\alpha & \text{cos }\alpha & 0 \\
0 & 0 & 1
\end{matrix}\right]
$$

### 2.4. RRPN中正负锚点的判断规则

由于RRPN中引入了角度标签，传统RPN的锚点正负的判断方法是不能应用到RRPN中的。

RRPN的正锚点的判断规则满足下面两点（and关系）：

1. 当锚点与Ground Truth的IoU大于0.7；
2. 当锚点与Ground Truth的夹角小于$$\frac{\pi}{12}$$。

RRPN的负锚点的判断规则满足1或者2（or关系）：

1. 与锚点的IoU小于0.3；
2. 与锚点的IoU大于0.7，但是夹角也大于$$\frac{\pi}{12}$$。

在训练时，只有判断为正锚点和负锚点的样本参与训练，其它不满足的样本并不会训练。

### 2.5. RRPN中IoU的计算方法

RRPN中IoU的计算和RPN思路相同，但是由于引入了角度信息，两个旋转矩形的交集就比较复杂了，如图3所示，两个相交旋转矩形的交集可根据交点的个数分为三种情况，分别是4个，6个，8个交点：

<figure>
<img src="/assets/RRPN_3.jpeg" alt="图3：旋转矩形交集情况" />
<figcaption>图3：旋转矩形交集情况</figcaption>
</figure>

两个矩形的交集的计算方式见下面伪代码：

<figure>
<img src="/assets/RRPN_code1.jpeg"/>
</figure>

首先根据矩形四条边的带定义域的一元一次方程求出所有交点，然后补充完整交集的顶点（即添加位于矩形B中的矩形A的所有顶点），顺时针排序之后最后根据三角形的三个顶点计算相交区域的面积。

### 2.6. RRPN的损失函数

RRPN的损失函数由分类任务和回归任务组成：

$$
L(p,l,v^*,v) = L_{cls}(p,l)+\lambda l L_{reg}(v^*, v)
$$

在上面的公式中，$$l$$表示前/背景的指示值，前景（文字区域）时$$l=1$$，背景时$$l=0$$。$$p=(p_0, p_1)$$表示的样本为前景或者背景的概率，使用了softmax作为激活函数，因此和是1。$$v=(v_x, v_y, v_w, v_h, v_{\theta})$$表示预测值，它计算的是bounding box和锚点的相对关系，因此对尺度不敏感。$$v=(v^*_x, v^*_y,v^*_w,v^*_h,v^*_{\theta})$$表示Ground Truth，也是计算的和锚点的相对关系。$$\lambda$$表示的是两个任务之间的平衡参数。

$$L_{cls}(p,l)$$使用的是log损失函数：

$$
L_{cls}(p,l) = - \text{log}p_l
$$

$$L_{reg}(v^*,v)$$使用的是smooth-$$L_1$$损失函数:

$$
L_{reg}(v^*,v) = \sum_{i\in\{x,y,h,w,\theta\}} \text{smooth}_{L_1}(v_i^*,v_i)
$$

尺度不敏感是通过对v进行归一化实现的,设$$(x_a,y_a,w_a,h_a,\theta_a)$$为当前锚点，$$v=(v_x, v_y, v_w, v_h, v_{\theta})$$的换算为：

$$
v_x = \frac{x-x_a}{w_a},
v_y = \frac{y-y_a}{h_a}
$$

$$
v_h = \text{log}\frac{h}{h_a},
v_w = \text{log}\frac{w}{w_a}
$$

$$
v_\theta = \theta^* \ominus \theta_a 
$$

其中$$a\ominus b = a-b+k\pi$$，k用于控制$$a\ominus b$$的值在$$[-\frac{\pi}{4}, \frac{3\pi}{4})$$的范围内。

和RPN相比，RRPN的最大变化在于在回归任务中添加了对相对角度$$\theta$$的预测。$$\theta$$的变化范围是$$[-\frac{\pi}{4}, \frac{3\pi}{4})$$，而锚点的6个角度分别是$$-\frac{\pi}{6}, 0, \frac{\pi}{6},\frac{\pi}{3},\frac{\pi}{2},\frac{2\pi}{3}$$（逐次增加$$\frac{pi}{12}$$）。结合正负锚点的判断规则，我们可以知道每个锚点都有一个对应的匹配范围，论文中将其命名为匹配域（fit domain），匹配域有两个重要特点：

1. 不同角度的锚点的匹配域是不相交的；
2. 同一个向量的不同角度的锚点的并集是全集。

上面两个特征产生了一个非常重要的性质：**当(x,y,w,h)确定时，并且$$\text{IoU}>0.7$$，一个Ground Truth有且只有一个正锚点与之匹配**。

图4是对该损失函数的可视化，线段的角度代表预测的bounding box的旋转角度，长度代表置信度。

<figure>
<img src="/assets/RRPN_4.jpeg" alt="图4：RRPN损失函数的可视化" />
<figcaption>图4：RRPN损失函数的可视化：(a)input image, (b)0 iters, (c)1500 iters, (d)15000 iters</figcaption>
</figure>

## 3. 位置精校部分详解

### 3.1 Skew NMS

传统的NMS只考虑IoU一个因素，而这点在RRPN中是行不通的，考虑一个偏移了$$\frac{\pi}{12}$$的矩形，虽然IoU只有0.31，但是由于偏移角度比较小，这种情况也应该考虑进去。

Skew-NMS在NMS的基础上加入了IoU信息：

1. 保留IoU大于0.7的最大的候选框；
2. 如果所有的候选框均位于$$[0.3,0.7]$$之间，保留小于$$\frac{\pi}{12}$$的最小候选框。

### 3.2 RRoI Pooling

如图1所示，RRPN得到的候选区域是旋转矩形，而传统的RoI池化只能处理与坐标轴平行的候选区域，因此作者提出了RRoI Pooling用于RRPN中的旋转矩形的池化。

如图5所示，首先需要设置超参数$$H_r$$和$$W_r$$，分别表示池化后得到的Feature Map的高和宽。然后将RRPN得到的候选区域等分成$$H_r \times W_r$$个小区域，每个子区域的大小是$$\frac{w}{W_r}\times\frac{h}{H_r}$$，这时每个区域仍然是带角度的，如5.(a)所示。接着通过仿射变换将子区域转换成平行于坐标轴的矩形，最后通过Max Pooling得到长度固定的特征向量。

<figure>
<img src="/assets/RRPN_5.jpeg" alt="图5：RRoI Pooling" />
<figcaption>图5：RRoI Pooling</figcaption>
</figure>

RROI Pooling的计算流程如算法2的伪代码。其中第一层for循环是遍历候选区域的所有子区域，5-7行通过仿射变换将子区域转换成标准矩形，第二个for循环用于取得每个子区域的最大值，10-11行由于对标准矩形中元素的插值，使用了向下取整的方式。

<figure>
<img src="/assets/RRPN_code2.jpeg"/>
</figure>

在RRoI Pooling之后，模型接来两个全连接层用于判断待检测区域是前景区域还是背景区域。

## 总结

RRPN非常创新的提出了使用带角度的锚点处理场景文字检测中最常见的倾斜问题，为了配合R-Anchor，论文中对RoI，NMS的计算也做了正对性的修改，另外RRoI Pooling层的提出将池化的目标区域扩展到了不仅仅局限于标准矩形。

结合目标检测中的一些Trick，应该能将检测精度进一步提高，使RRPN在特定场景的比赛中也非常有用。

最后还有一点疑问：论文中说RRPN的位置精校部分只进行了二分类，但个人感觉更好的策略是Faster R-CNN中的多任务模型，粗略的看了一下代码好像也是使用的Faster R-CNN的策略。

## Reference

\[1\] Ma J, Shao W, Ye H, et al. Arbitrary-oriented scene text detection via rotation proposals\[J\]. IEEE Transactions on Multimedia, 2018.

\[2\] Wolf, C., Jolion, J.: Object count / area graphs for the evaluation of object detection and segmentation algorithms. International Journal of Document Analysis 8, 280–296 (2006)

\[3\] Zhong Z, Jin L, Zhang S, et al. Deeptext: A unified framework for text proposal generation and text detection in natural images\[J\]. arXiv preprint arXiv:1605.07314, 2016.

\[4\] Yao C, Bai X, Sang N, et al. Scene text detection via holistic, multi-channel prediction\[J\]. arXiv preprint arXiv:1606.09002, 2016.

\[5\] Zhou X, Yao C, Wen H, et al. EAST: an efficient and accurate scene text detector\[C\]//Proc. CVPR. 2017: 2642-2651.

\[6\] Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks (2015), in Neural Information Processing Systems (NIPS)



