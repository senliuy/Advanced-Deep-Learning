# Deep TextSpotter: An End-to-End Trainable Scene Text Localization and Recognition Framework

tags: Deep TextSpotter, OCR, YOLOv2, STN, CTC

## 前言

Deep TextSpotter的创新点并不多，基本上遵循了传统OCR或者物体检测的两步走的流程（图1），即先进行场景文字检测，再进行文字识别。在这个算法中，检测模块基于[YOLOv2](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/yolo9000-better-faster-stronger.html)\[2\]，识别模块基于[STN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/spatial-transform-networks.html)\[3\]，损失函数则使用了精度的[CTC](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/connectionist-temporal-classification-labelling-unsegmented-sequence-data-with-recurrent-neural-networks.html)\[4\]。这几个算法在当时都是state-of-the-art的，因此其效果达到了最优也不难理解。这三个知识点已分别在本书的第四章，第五章和第二章进行了解析，算法细节可参考具体内容或者阅读论文。这里不在对上面三个算法的细节再做重复，只会对Deep TextSpotter的流程做一下梳理和解释。

Deep TextSpotter的一个创新点是将NMS放到了识别之后，使用识别置信度替代了传统的检测置信度。

###### 图1： Deep TextSpotter算法流程

![](/assets/DeepTextSpotter_1.png)

## 1. Deep TextSpotter解析

### 1.1 全卷积网络

**为什么使用YOLOv2**：在YOLOv2的文章中我们讲过，YOLOv2使用了高分辨率的迁移学习提高了网络对高分辨率图像的检测效果，这个能力在端到端的文字检测及识别中非常重要。因为过分的降采样将造成文本区域的识别问题。

**Feature Map的尺寸**：网络的框架也采样去YOLOv2中在$$3\times3$$卷积中插入$$1\times1$$卷积进行非线性化的结构。对于一张尺寸为$$W\times H$$的输入图像，在网络中会通过5个Max Pooling进行降采样，得到尺寸为$$\frac{W}{32} \times \frac{H}{32}$$的的Feature Map。在Deep TextSpotter中，每隔20个Epoch会更换一次输入图像的尺寸，尺寸的变化范围是$$\{352,416,480,544,608\}$$

**全卷积**：Deep TextSpotter使用了Global Average Pooling代替全连接实现非线性化，从而使网络成为全卷积网络，原因已多次提及：保留特征向量的位置信息。

### 1.2 候选区域提取

**输出向量**：Deep TextSpotter的检测部分预测了6个值，它们分别是坐标$$r_x$$，$$r_y$$，尺寸$$r_w$$，$$r_h$$，检测置信度$$r_p$$以及比YOLOv2增加的一个旋转角度$$r_\theta$$。其中角度使用了弧度值，即 $$\theta \in (-\frac{\pi}{2}, \frac{\pi}{2})$$。其它几个预测值则采用了YOLOv2中使用的预测相对值。

**锚点聚类**

## Reference

\[1\] Bušta M, Neumann L, Matas J. Deep textspotter: An end-to-end trainable scene text localization and recognition framework\[C\]//Computer Vision \(ICCV\), 2017 IEEE International Conference on. IEEE, 2017: 2223-2231.

\[2\] Redmon J, Farhadi A. YOLO9000: better, faster, stronger\[J\]. arXiv preprint, 2017.

\[3\] Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks\[C\]//Advances in neural information processing systems. 2015: 2017-2025.

\[4\] Connectionist Temporal Classification : Labelling Unsegmented Sequence Data with Recurrent Neural Networks. Graves, A., Fernandez, S., Gomez, F. and Schmidhuber, J., 2006. Proceedings of the 23rd international conference on Machine Learning, pp. 369--376. DOI: 10.1145/1143844.1143891

