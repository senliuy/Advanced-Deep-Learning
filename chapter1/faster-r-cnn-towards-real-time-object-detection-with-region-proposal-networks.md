# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

## 简介

Fast-RCNN \[1\]虽然实现了端到端的训练，而且也通过共享卷积的形式大幅提升了R-CNN的计算速度，但是其仍难以做到实时。其中一个最大的性能瓶颈便是候选区域的计算。在之前的物体检测系统中，Selective Search是最常用的候选区域提取方法，它贪心的根据图像的低层特征合并超像素（SuperPixel）。另外一个更快速的版本是EdgeBoxes \[4\]，虽然EdgeBoxes的提取速度达到了0.2秒一张图片，当仍然难以做到实时，而且EdgeBoxes为了速度牺牲了提取效果。Selective Search速度慢的一个重要原因是不同于检测网络使用GPU进行运算，SS使用的是CPU。从工程的角度讲，使用GPU实现SS是一个非常有效的方法，但是其忽视了共享卷积提供的非常有效的图像特征 。

由于卷积网络具有强大的拟合能力，很自然的我们可以想到可以使用卷积网络提取候选区域，由此，便产生了Faster R-CNN最重要的核心思想：RPN \(Region Proposal Networks\)。通过SPP-net \[5\]的实验得知，卷积网络可以很好的提取图像语义信息，例如图像的形状，边缘等等。所以，这些特征理论上也应该能够用提取候选区域。在论文中，作者给RPN的定义如下：RPN是一种可以端到端训练的全卷积网络 \[6\]，主要是用来产生候选区域。

RPN是通过一个叫做锚点（anchor）的机制实现的。

## 算法详解

## 参考文献

\[1\] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in CVPR, 2014

\[2\] R. Girshick, “Fast R-CNN,” in IEEE International Conference on Computer Vision \(ICCV\), 2015.

\[3\] J. R. Uijlings, K. E. van de Sande, T. Gevers, and A. W. Smeulders, “Selective search for object recognition,” International Journal of Computer Vision \(IJCV\), 2013.

\[4\] C. L. Zitnick and P. Dollar, “Edge boxes: Locating object ´ proposals from edges,” in European Conference on Computer Vision \(ECCV\), 2014.

\[5\] K. He, X. Zhang, S. Ren, and J. Sun, “Spatial pyramid pooling in deep convolutional networks for visual recognition,” in European Conference on Computer Vision \(ECCV\), 2014.

\[6\] J. Long, E. Shelhamer, and T. Darrell, “Fully convolutional networks for semantic segmentation,” in IEEE Conference on Computer Vision and Pattern Recognition \(CVPR\), 2015.

