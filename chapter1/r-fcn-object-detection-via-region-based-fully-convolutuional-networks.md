# R-FCN: Object Detection via Region-based Fully Convolutuional Networks

## 简介：

位移不变性是卷积网络一个重要的特征，该特征使得卷积网格在图像分类任务上取得了非常好的效果。但是在物体检测的场景中，我们需要知道检测物体的具体位置，这时候我们需要网络对物体的位置非常敏感，即我们需要网络具有“位移可变性”。R-FCN \[1\]的提出便是解决分类任务中位移不变性和检测任务中位移可变性直接的矛盾的。

同时，作者分析了Faster R-CNN \[2\]存在的性能瓶颈，即ROI之后使用Fast R-CNN \[3\]对RPN提取的候选区域进行分类和位置精校。在R-FCN中，ROI之后便不存在可学习的参数，从而将Faster-RCNN的速度提高了2.5-20倍。

## 1. 动机

在R-CNN系列论文中，物体检测一般分成两步：

1. 提取候选区域；
2. 候选区域分类和位置精校。

在R-FCN之前，state-of-the-art的Faster-RCNN使用RPN网络进行候选区域（Proposal Region）选择，然后再使用Fast R-CNN进行分类。在Faster R-CNN中，首先使用ROI层将不同大小的候选区域归一化到统一大小，之后接若干个全连接层，最后使用一个多任务作为损失函数。多任务包含两个子任务：

1. softmax的分类任务；
2. 用于位置精校的回归任务

Faster R-CNN之所以这样做主要是因为其使用了VGG \[4\]作为特征提取器。在第一章中，我们了解到VGG之后的GoogLeNet \[5\]和ResNet \[6\]均是使用了全卷积的结构，即使用1\*1卷积代替全连接。1\*1卷积具备全连接层增加非线性性的作用，同时还保证了特征点的位置敏感性。可见在物体检测任务中引入1\*1卷积会非常有帮助的。

在Faster R-CNN中，为了保证特征的“位移敏感性”，作者根据RPN提取了约2000个候选区域，然后使用全连接层计算损失函数，然而候选区域有大量的特征冗余，造成了一部分计算资源的浪费。

## Reference

\[1\] Dai J, Li Y, He K, et al. R-fcn: Object detection via region-based fully convolutional networks\[C\]//Advances in neural information processing systems. 2016: 379-387.

\[2\] Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks \(2015\), in Neural Information Processing Systems \(NIPS\)

\[3\] R. Girshick, “Fast R-CNN,” in IEEE International Conference on Computer Vision \(ICCV\), 2015.

\[4\] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition\[J\]. arXiv preprint arXiv:1409.1556, 2014.

\[5\]

\[6\] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

