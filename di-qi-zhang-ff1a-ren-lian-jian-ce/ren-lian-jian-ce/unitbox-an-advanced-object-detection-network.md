# UnitBox: An Advanced Object Detection Network

## 前言

UnitBox使用了和[DenseBox](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-qi-zhang-ff1a-ren-lian-jian-ce/ren-lian-jian-ce/densebox-unifying-landmark-localization-with-end-to-end-object-detection.html)\[2\]类似的基于图像分割的方法进行人脸检测。在DenseBox中，bounding box的定位使用的是l2损失。l2损失的一个缺点是会使模型在训练过程中更偏向于尺寸更大的物体，因为大尺寸物体的l2损失更容易大于小物体。

为了解决这个问题，UnitBox中使用了IoU损失，顾名思义，IoU损失既是使用Ground Truth和预测bounding box的交并比作为损失函数。

## 1. UnitBox详解

首先回顾DenseBox中介绍的几个重要的知识点，明白了这些知识点才能理解下面要讲解的UnitBox。

1. DenseBox网络结构是全卷积网络，输出层是一个$$\frac{m}{4}\times \frac{n}{4}$$的Feature Map，是一个image-to-image的任务；
2. 输出Feature Map的每个像素点$$(x_i, y_i)$$都是可以确定一个检测框的样本，包样本含置信度$$y$$和该点到bounding box四条边的距离$$(x_t,x_b,y_t,y_b)$$，如图1所示

<figure>
<img src="/assets/UnitBox_1.png" alt="图1：UnitBox的Ground Truth" width="600" align="center"/>
<figcaption>图1：UnitBox的Ground Truth</figcaption>
</figure>

Unitbox的一个最重要的特征是使用IoU损失替代了传统的l2损失，下面我们先从IoU损失入手讲解UnitBox。

### 1.1 IoU损失的前向计算 

前向计算非常简单，如图2中的伪代码所示：

<figure>
<img src="/assets/UnitBox_2.png" alt="图2：IoU损失前向计算伪代码" width="400"/>
<figcaption>图2：IoU损失前向计算伪代码</figcaption>
</figure>

注意结合图1中的$$x$$和$$\tilde{x}$$的定义理解图2中的伪代码，$$X$$计算的是预测bounding box的面积，$$\tilde{X}$$则是ground truth的bounding box的面积，$$I$$是两个区域的交集，$$U$$是两个区域的并集。

$$\mathcal{L} = -ln(IoU)$$本质上是对IoU的交叉熵损失函数：那么可以将IoU看做从伯努利分布中的随机采样，并且$$p(IoU=1)=1$$，于是可以化简成源码中的公式，即

$$
\mathcal{L} = -pln(IoU)-(1-p)ln(IoU)=-ln(IoU)
$$

### 1.2 IoU损失的反向计算

这里我们推导一下IoU损失的反向计算公式，以变量$$x_t$$为例：

$$
\frac{\partial \mathcal{L}}{\partial x_t} = 
\begin{array}
\frac{\partial}{\partial x_t}(-ln(IoU)) \\
= -\frac{1}{IoU}\frac{\partial}{\partial x_t}(IoU) \\
= -\frac{1}{IoU}\frac{\partial}{\partial x_t}(\frac{I}{U}) \\
= \frac{1}{IoU}\frac{I\times\frac{\partial U}{\partial x_t} - U\times\frac{\partial I}{\partial x_t}}{U^2}\\
= \frac{I\times\frac{\partial}{x_t}(X+\tilde{X}-I) - U\times\frac{\partial I}{\partial x_t}}{U^2 IoU} \\
= \frac{I\times (\frac{\partial}{x_t}X - \frac{\partial}{\partial x_t}I) - U \times \frac{\partial I}{\partial x_t}}{U^2 IoU} \\
= \frac{1}{U}\frac{\partial X}{x_t} - \frac{U+I}{UI}\frac{\partial I}{x_t}
\end{array}
$$

其中：

$$
\frac{\partial X}{x_t} = x_l+x_r
$$

$$
\frac{\partial I}{x_t} = 
\left\{
\begin{array}{}
I_w, \quad \text{if } x_t < \tilde{x}_t (\text{or }x_b < \tilde{x}_b) \\
0, \quad \text{otherwise}
\end{array}
\right.
$$

其它三个变量的推导方法类似，这里不再重复。

从这个推导公式中我们可以看出三点信息：

1. 损失函数和$$\frac{\partial X}{x_t}$$成正比，因此预测的面积越大，损失越多；
2. 同时损失函数和$$\frac{\partial I}{x_t}$$成反比，因此我们希望交集尽可能的大；
3. 综合1，2两条我们可以看出当bounding box等于ground truth值时检测效果最好。

因此可以看出优化IoU损失是正向促进物体检测的精度的。

## 1.3 UnitBox网络架构

UnitBox的网络结构如图3所示，下面分析几个重要的方面

<figure>
<img src="/assets/UnitBox_3.png" alt="图3：UnitBox网络结构" width="600"/>
<figcaption>图3：UnitBox网络结构</figcaption>
</figure>

### 1.3.1 网络的输入输出

**输入**：由于使用了全卷积结构，在测试时直接输入原始图片即可。在训练时每个batch的图像的尺寸相同即可。

**输出**：UnitBox的输出标签分成两部分，上半部分椭圆形为置信度热图，具体标签生成方法论文中没有讲，猜测应该是采样类似于DenseBox中的策略。另外一组是bounding box热图，生成策略应该也是类似于DenseBox。

### 1.3.2 骨干网络

骨干网络是VGG-16，用于计算置信度热图的是stage-4阶段的Feature Map，计算方式是先通过线性插值得到相同尺寸的Feature Map，再通过$$1\times1$$卷积将Feature Map的通道数降到1，此时得到的Feature Map表示预测的置信度热图。网络的另外一个分支用于预测bounding box的热图，Feature Map取自VGG-16的stage-5。通过和上面类似的方法得到和原图尺寸相同的4个预测bounding box热图。并且在后面加入了ReLU将负值置0。

至于为什么两个任务使用不同的stage，论文中给出的解释是IoU损失计算的Bounding box是一个整体，因此需要更大的感受野，由于UnitBox仅添加了两组$$1\times1$$卷积，因此速度要比DenseBox快很多。

## 2. 总结

UnitBox的提出虽然目的是为了解决人脸检测问题，但是从其算法角度讲也可以扩展到其它类型的检测任务中，但是像Pascal VOC或者是COCO这样的多类别检测任务，目前基于语义分割的方法还需要改进，因为其本身的置信度热图的设计是位于$$[0,1]$$直接的一个值，暂时是无法进行多分类的。

写这篇文章的目的主要是介绍UnitBox中引入的IoU损失，IoU损失有如下优点

* IoU损失将位置信息作为一个整体进行训练，而l2损失把它们当做互相独立的四个变量进行训练，这样得到的结果更准确；

* 无论输入的样本是什么样子，IoU的值均介于$$[0,1]$$，这种天然的归一化的损失使模型具有更强的处理多尺度图像的能力。

## Reference

\[1\]Zhou X, Yao C, Wen H, et al. EAST: an efficient and accurate scene text detector\[C\]//Proc. CVPR. 2017: 2642-2651.

\[2\] Qin H, Yan J, Li X, et al. Joint training of cascaded cnn for face detection\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 3456-3465.

