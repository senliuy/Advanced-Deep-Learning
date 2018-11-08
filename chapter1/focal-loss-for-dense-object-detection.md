# Focal Loss for Dense Object Detection

## 前言

何凯明，RBG等人一直是Two-Stage方向的领军人，在这篇论文中，他们也开始涉足One-Stage的物体检测算法。大牛就是牛，一次就刷新了精度。下面我们就来分析这几个大牛的作品。

目前主流的检测算法分为两个方向：（1）以[R-CNN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/fast-r-cnn.html)\[2\]系列为代表的two-stage方向；（2）以[YOLO](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/you-only-look-once-unified-real-time-object-detection.html)\[3\]系列为代表的one-stage方向。虽然one-stage方向的速度更快，但是其精度往往比较低。究其原因，有两个方面：

1. 正样本（Positive Example）和负样本（Negative Example）的不平衡；
2. 难样本（Hard Example）和易样本（Easy Example）的不平衡。

这些不平衡造成模型的效果不准确的原因如下：

1. Negative example的数量过多，导致Postive example的loss被覆盖，就算Postive example的loss非常大也会被数量庞大的 negative example中和掉，这这些positive example往往是我们要检测的前景区域；

2. Hard example往往是前景和背景区域的过渡部分，因为这些样本很难区分，所以叫做Hard Example。剩下的那些Easy example往往很好计算，导致模型非常容易就收敛了。但是损失函数收敛了并不代表模型效果好，因为我们其实更需要把那些hard example训练好。

四种example的情况见图1。

<figure>
<img src="/assets/RetinaNet_1.jpeg" alt="图1：物体检测中的四种Example" />
<figcaption>图1：物体检测中的四种Example</figcaption>
</figure>

Faster R-CNN之所以能解决两个不平衡问题是因为其采用了下面两个策略：

1. 根据IoU采样候选区域，并将正负样本的比例设置成1：1。这样就解决了正负样本不平衡的问题；

2. 根据score过滤掉easy example，避免了训练loss被easy example所支配的问题。

而在这篇论文中他们采用的解决方案是基于交叉熵提出了一个新的损失函数Focal Loss（FL）。


$$
\text{FL}(p_t) = - \alpha_t (1-p_t)^{\gamma}\text{log}(p_t)
$$


FL是一个尺度动态可调的交叉熵损失函数，在FL中有两个参数$$\alpha_t$$和$$\gamma$$，其中$$\alpha_t$$主要作用是解决正负样本的不平衡，$$\gamma$$主要是解决难易样本的不平衡。

最后，作者基于[残差网络](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)\[4\]，[FPN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/mask-r-cnn.html)\[5\]搭建了检测网络RetinaNet，该网络使用的策略都是他们自己提出的而且目前效果非常好的基础结构，再结合Focal Loss，刷新检测算法的精度也不意外。

## 1. Focal Loss

Focal Loss是交叉熵损失的改进版本，一个二分类交叉熵可以表示为：


$$
\text{CE}(p,y) = 
\left\{
\begin{array}{}
-\text{log}(p) & \text{if}\quad y=1\\
-\text{log}(1-p) & \text{otherwise}
\end{array}
\right.
$$


上面公式可以简写成：


$$
\text{CE}(p,y) = \text{CE}(p_t) = -\text{log}(p_t)
$$


其中：


$$
p_t = 
\left\{
\begin{array}{}
p & \text{if}\quad y=1\\
1-p & \text{otherwise}
\end{array}
\right.
$$


### 1.1 $$\alpha$$：解决正负样本不平衡

平衡交叉熵的提出是为了解决**正负样本不平衡**的问题的。它的原理很简单，为正负样本分配不同的权重比值$$\alpha \in [0,1]$$，当$$y=1$$时取$$\alpha$$，为$$-1$$时取$$1-\alpha$$。我们使用和$$p_t$$类似的方法将上面$$\alpha$$的情况表示为$$a_t$$，即：


$$
\alpha_t = 
\left\{
\begin{array}{}
\alpha & \text{if}\quad y=1\\
1-\alpha & \text{otherwise}
\end{array}
\right.
$$


那么这个$$\alpha-\text{balanced}$$交叉熵损失可以表示为式\(6\)。


$$
\text{CE}(p_t) = -\alpha_t \text{log} (p_t)
$$


$$\alpha$$的值往往需要根据验证集进行调整，论文中给出的是0.25。

### 1.2 $$\gamma$$：解决难易样本不平衡

FL中$$\gamma$$的引入是为了解决**难易样本不平衡**的问题的。图2是FL中example预测概率和loss值之间的关系。其中蓝色曲线是交叉熵（$$\gamma=0$$时Focal Loss退化为交叉熵损失）的曲线。

<figure>
<img src="/assets/RetinaNet_2.png" alt="图2：CE损失和FL损失曲线图" />
<figcaption>图2：CE损失和FL损失曲线图</figcaption>
</figure>

从图2的曲线中我们可以看出对于一些well-classified examples \(easy examples\)虽然它们**单个example**的loss可以收敛到很小，但是由于它们的数量过于庞大，把一些hard example的loss覆盖掉。导致求和之后他们依然会支配整个批次样本的收敛方向。

一个非常简单的策略是继续缩小easy examples的训练比重。作者的思路很简单，给每个乘以$$(1-p_t)^\gamma$$。因为easy example的score $$p_t$$往往接近1，那么$$(1-p_t)^\gamma$$值会比较小，因此example得到了抑制，相对的hard example得到了放大，例如图2中$$\gamma>0$$的那四条曲线。

FL的求导结果如公式\(7\):


$$
\frac{d\text{FL}}{dx} = y(1-p_t)^\gamma(\gamma p_t\text{log}(p_t) + p_t - 1)
$$


$$\gamma$$的值也可以根据验证集来调整，论文中给出的值是2。

### 1.3 FL的最终形式

结合1.1的$$\alpha$$和1.2的$$\gamma$$，我们便有了公式（1）中FL的最终形式。作者也通过实验验证了结合两个策略的实验效果最好。

Focal Loss的最终形式并不是一定要严格的是\(1\)的情况，但是它应满前文的分析，即能缩小easy example的比重。例如在论文附录A中给出的另外一种Focal Loss：$$\text{FL}^\star$$，曲线见图3。它能取得和FL类似的效果。

<figure>
<img src="/assets/RetinaNet_3.png" alt="图3：CE损失和FL*损失曲线图" />
<figcaption>图3：CE损失和FL*损失曲线图</figcaption>
</figure>




$$
\text{FL}^\star = 
-\frac{\text{log}(\sigma(\gamma yx + \beta))}{\gamma}
$$


最后作者指出如果将单标签softmax换成多标签的sigmoid效果会更好，这里应该和我们在YOLOv3中分析的情况类似。

## 2. RetinaNet

算法使用的检测框架RetinaNet并没有特别大的创新点，基本上是残差网络+FPN的最state-of-the-art的方法，如图4。

<figure>
<img src="/assets/RetinaNet_4.png" alt="图4：RetinaNet网络结构图" />
<figcaption>图4：RetinaNet网络结构图</figcaption>
</figure>


对于残差网络和FPN不清楚的参考论文或者我之前的分析。这里我们列出RetinaNet的几个重点：

1. 融合的特征层是P3-P7；
2. 每个尺度的Feature Map有一组锚点（3*3=9）；
3. 分类任务和预测任务的FPN部分的参数共享，其它参数不共享。

## 3. 测试

测试的时候计算所有锚点的score，再从其中选出top-1000个进行NMS，NMS的阈值是0.5。

## 4. 总结

Focal Loss的论文非常简单有效，非常符合何凯明等人的风格。FL中引入的两个参数$$\alpha$$和$$\gamma$$分别用于抑制正负样本和难易样本的不平衡，动机明确。

Focal Loss几乎可以应用到很多imbalance数据的领域，还是非常有实用价值的。

## Reference

\[1\] Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object detection\[J\]. IEEE transactions on pattern analysis and machine intelligence, 2018.

\[2\] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in CVPR, 2014

\[3\] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

\[4\] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

\[5\] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and ´ S. Belongie. Feature pyramid networks for object detection. In CVPR, 2017. 2, 4, 5, 7

