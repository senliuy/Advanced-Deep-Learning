# Mask R-CNN

## 前言

个人非常喜欢何凯明的文章，两个原因，1\) 简单，2\) 好用。对比目前科研届普遍喜欢把问题搞复杂，通过复杂的算法尽量把审稿人搞蒙从而提高论文的接受率的思想，无论是著名的残差网络还是这篇Mask R-CNN，大神的论文尽量遵循著名的奥卡姆剃刀原理：即在所有能解决问题的算法中，选择最简单的那个。霍金在出版《时间简史》中说“书里每多一个数学公式，你的书将会少一半读者”。Mask R-CNN更是过分到一个数学公式都没有，而是通过对问题的透彻的分析，提出针对性非常强的解决方案，下面我们来一睹Mask R-CNN的真容。

## 动机

语义分割和物体检测是计算机视觉领域非常经典的两个重要应用。在语义分割领域，FCN\[2\]是代表性的算法；在物体检测领域，代表性的算法是Faster R-CNN\[3\]。很自然的会想到，结合FCN和Faster R-CNN不仅可以是模型同时具有物体检测和语义分割两个功能，还可以是两个功能互相辅助，共同提高模型精度，这便是Mask R-CNN的提出动机。Mask R-CNN的结构如图1

###### 图1：Mask R-CNN框架图

\[Mask\_R-CNN1\]

如图1所示，Mask R-CNN分成两步：

1. 使用RPN网络产生候选区域；
2. 分类，bounding box，掩码预测的多任务损失。

在Fast R-CNN的解析文章中，我们介绍Fast R-CNN采用ROI池化来处理候选区域尺寸不同的问题。但是对于语义分割任务来说，一个非常重要的要求便是特征层和输入层像素的一对一，ROI池化显然不满足该要求。为了改进这个问题，作者仿照STN \[4\]中提出的双线性插值提出了ROIAlign，从而使Faster R-CNN的特征层也能进行语义分割。

下面我们结合代码详细解析Mask R-CNN，代码我使用的是基于TensorFlow和Keras实现的版本：[https://github.com/matterport/Mask\_RCNN](https://github.com/matterport/Mask_RCNN)。

## Mask R-CNN详解

### 1. Backbone Structure



## Reference

\[1\] He K, Gkioxari G, Dollár P, et al. Mask r-cnn\[C\]//Computer Vision \(ICCV\), 2017 IEEE International Conference on. IEEE, 2017: 2980-2988.

\[2\] J. Long, E. Shelhamer, and T. Darrell. Fully convolutional networks for semantic segmentation. In CVPR, 2015. 1, 3, 6

\[3\] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015. 1, 2, 3, 4, 7

\[4\] M. Jaderberg, K. Simonyan, A. Zisserman, and K. Kavukcuoglu. Spatial transformer networks. In NIPS, 2015. 4

\[5\] 

### 附录A: 双线性插值





