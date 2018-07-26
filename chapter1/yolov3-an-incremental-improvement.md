# YOLOv3: An Incremental Improvement

## 前言

YOLOv3论文的干货并不多，用作者自己的话说是一篇“Tech Report”。这篇主要是在YOLOv2的基础上的一些Trick尝试，有的Trick成功了，包括：

1. 考虑到检测物体的重叠情况，用多标签的方式替代了之前softmax单标签方式；
2. 骨干架构使用了更为有效的残差网络，网络深度也更深；
3. 多尺度特征使用的是FPN的思想；
4. 锚点聚类成了9类。

也有一些尝试失败了，在介绍完YOLOv3的细节后我们在说明这些尝试会更好理解。在分析论文时，我们依然会使用一份Keras的[源码](https://github.com/qqwweee/keras-yolo3)辅助理解。

## YOLOv3详解

### 1. 多标签任务

不管是在检测任务的标注数据集，还是在日常场景中，物体之间的相互覆盖都是不能避免的。因此一个锚点的感受野肯定会有包含两个甚至更多个不同物体的可能，在之前的方法中是选择和锚点IoU最大的Ground Truth作为匹配类别，用softmax作为激活函数。

YOLOv3提供的解决方案是将一个$$N$$ 路softmax分类器替换成$$N$$ 个sigmoid分类器，这样每个类的输出仍是$$[0,1]$$ 之间的一个值，但是他们的和不再是1。

虽然YOLOv3改变了输出层的激活函数，但是其锚点和Ground Truth的匹配方法仍旧采用的是YOLOv1的方法，即每个Ground Truth匹配且只匹配唯一一个与其IoU最大的锚点。但是在输出的时候由于各类的概率之和不再是1，只要置信度大于阈值，该锚点便被作为检测框输出。

训练标签的制作和测试过程候选框的输出分别在`./yolo3/model.py`的`yolo_eval`和`preprocess_true_boxes`函数中实现的。

### 2. 骨干网络

YOLOv3使用了由残差块构成的全卷积网络作为骨干网络，网络深度达到了53层，因此作者将其命名为Darknet-53。Darknet-53的详细结构见图1。

###### 图1：Darknet-53

![](/assets/YOLOv3_1.png)

Keras非常友好的支持了网络可视化，Darknet-53的网络结构见链接[]

## Reference

\[1\] Redmon J, Farhadi A. Yolov3: An incremental improvement\[J\]. arXiv preprint arXiv:1804.02767, 2018.

