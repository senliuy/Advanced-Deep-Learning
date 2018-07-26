# YOLOv3: An Incremental Improvement

## 前言

YOLOv3论文的干货并不多，用作者自己的话说是一篇“Tech Report”。这篇主要是在YOLOv2的基础上的一些Trick尝试，有的Trick成功了，包括：

1. 考虑到检测物体的重叠情况，用多标签的方式替代了之前softmax单标签方式；
2. 骨干架构使用了更为有效的残差网络，网络深度也更深；
3. 多尺度特征使用的是FPN的思想；
4. 锚点聚类成了9类。

也有一些尝试失败了，在介绍完YOLOv3的细节后我们在说明这些尝试会更好理解。在分析论文时，我们依然会使用一份Keras的[源码](https://github.com/qqwweee/keras-yolo3)辅助理解。

## YOLOv3详解

## 1. 多标签任务

不管是在检测任务的标注数据集，还是在日常场景中，物体之间的相互覆盖都是不能避免的。因此一个锚点的感受野肯定会有包含两个甚至更多个不同物体的可能，在之前的方法中是选择和锚点IoU最大的Ground Truth作为匹配类别，用softmax作为激活函数。

YOLOv3将IoU超过阈值的

## Reference

\[1\] Redmon J, Farhadi A. Yolov3: An incremental improvement\[J\]. arXiv preprint arXiv:1804.02767, 2018.

