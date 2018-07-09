# You Only Look Once: Unified, Real-Time Object Detection

## 前言

在R-CNN系列的论文中，目标检测被分成了候选区域提取和候选区域分类及精校两个阶段。不同于这些方法，YOLO将整个目标检测任务整合到一个回归网络中。对比Fast R-CNN提出的两步走的端到端方案，YOLO的单阶段的使其是一个更彻底的端到端的算法（图1）。YOLO的检测过程分为三步：

1. 图像Resize到448\*448；
2. 将图片输入卷积网络；
3. NMS得到最终候选框。

图1：YOLO算法框架。



虽然在一些数据集上的表现不如Fast R-CNN及其后续算法，但是YOLO带来的最大提升便是检测速度的提升。在YOLO算法中，检测速度达到了每秒45帧，而一个更快速的Fast Yolo版本则达到了155帧/秒。另外在YOLO的背景检测错误率要低于Fast R-CNN。最后，YOLO算法具有更好的通用性，通过Pascal数据集训练得到的模型在艺术品问检测中得到了比Fast R-CNN更好的效果。

YOLO是可以用在Fast R-CNN中的，结合YOLO和Fast R-CNN两个算法，得到的效果比单Fast R-CNN要更好。

## YOL算法详解

### 1. 统一检测



## Reference

\[1\] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

