# SSD: Single Shot MultiBox Detector

## 前言

在YOLO\[2\]的文章中我们介绍到YOLO存在三个缺陷：

1. 两个bounding box功能的重复降低了模型的精度；
2. 全连接层的使用不仅使特征向量失去了位置信息，还产生了大量的参数，影响了算法的速度；
3. 只使用顶层的特征向量使算法对于小尺寸物体的检测效果很差。

为了解决这些问题，SSD应运而生。SSD的全称是Single Shot MultiBox Detector，Single Shot表示SSD是像YOLO一样的单次检测算法，MultiBox指SSD每次可以检测多个物体，Detector表示SSD是用来进行物体检测的。

针对YOLO的三个问题，YOLO做出的改进如下：

1. 使用了类似Faster R-CNN中RPN网络提出的锚点（Anchor）机制，增加了bounding box的多样性；
2. 使用全卷积的网络结构，提升了SSD的速度；
3. 使用网络中多个阶段的Feature Map，提升了特征多样性。

从某个角度讲，SSD和RPN的相似度也非常高，网络结构都是全卷积，都是采用了锚点进行采样，不同之处有下面两点：

1. RPN只使用卷积网络的顶层特征，不过在FPN和Mask R-CNN中已经对这点进行了改进；
2. RPN是一个二分类任务（前/背景），而SSD是一个包含了物体类别的多分类任务。

在论文中作者说SSD的精度超过了Faster R-CNN，速度超过了YOLO。下面我们将结合基于TensorFlow的[源码](https://github.com/balancap/SSD-Tensorflow)和论文对SSD进行详细剖析。这里说明一下，这份源码使用了slim库，slim库是TensorFLow的一个高层封装，和keras的功能类似。

## SSD详解

### 1. 算法流程

SSD的流程和YOLO是一样的，输入一张图片得到一系列候选区域，使用NMS得到最终的检测框。与YOLO不同的是，SSD使用了不同阶段的Feature Map用于检测，YOLOh和SSD的对比如图1所示

###### 图1：SSD vs YOLO

\[SSD\_1.png\]

#### 1.1 SSD的骨干架构

从图1中我们可以看出，SSD输入图片的尺寸是300\*300，另外SSD也由一个输入图片尺寸是512\*512的版本，这个版本的SSD虽然慢一些，但是是检测精度达到了76.9%。

SSD采用的是VGG-16的作为骨干网络，VGG的详细内容参考文章[Very Deep Convolutional NetWorks for Large-Scale Image Recognition](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/very-deep-convolutional-networks-for-large-scale-image-recognition.html)。在VGG的第四个block中，conv4\_3的Feature Map的尺寸是38\*38。

## Reference

\[1\] Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector\[C\]//European conference on computer vision. Springer, Cham, 2016: 21-37.

\[2\] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

