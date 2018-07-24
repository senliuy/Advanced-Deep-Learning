# YOLO9000: Better, Faster, Stronger

## 前言

在这篇像极了奥运格言（Faster，Higher，Stronger）的论文[1]中，作者提出了YOLOv2和YOLO9000两个模型。

其中YOLOv2采用了若干技巧对[YOLOv1](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/you-only-look-once-unified-real-time-object-detection.html)的速度和精度进行了提升。其中比较有趣的有以下几点：

1. 使用聚类产生的锚点代替[Faster R-CNN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.html)和[SSD](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/ssd-single-shot-multibox-detector.html)手工设计的锚点；
2. 在高分辨率图像上进行迁移学习，提升网络对高分辨图像的响应能力；
3. 训练过程图像的尺寸不再固定，提升网络对不同训练数据的泛化能力。

除了以上三点，YOLO还使用了残差网络的直接映射的思想，R-CNN系列的预测相对位移的思想，Batch Normalization，全卷积等思想。YOLOv2将算法的速度和精度均提升到了一个新的高度。正是所谓的速度更快（Faster），精度更高（Better/Higher）

论文中提出的另外一个模型YOLO9000非常巧妙的使用了WordNet[5]的方式将检测数据集COCO和分类数据集ImageNet整理成一个多叉树，再通过提出的联合训练方法高效的训练多叉树对应的损失函数。YOLO9000是一个非常强大（Stronger）且有趣的模型，非常具有研究前景。

在下面的章节中，我们将论文分成YOLOv2和YOLO9000两个部分并结合论文和源码对算法进行详细解析。

## YOLOv2: Better, Faster

YOLOv1之后，一系列算法和技巧的提出极大的提高了深度学习在各个领域的泛化能力。作者总结了可能在物体检测中有用的方法和技巧（图1）并将它们结合成了我们要介绍的YOLOv2。所以YOLOv2并没有像SSD或者Faster R-CNN具有很大的难度，更多的是技巧方向的提升。在下面的篇幅中，我们将采用和论文相同的结构并结合基于Keras的[源码](https://github.com/yhcc/yolo2)对YOLOv2中涉及的技巧进行讲解。

######图1：YOLOv2中使用的技巧及带来的性能提升

![](/assests/YOLOv2_1.png)




## Reference

\[1\] Redmon J, Farhadi A. YOLO9000: better, faster, stronger\[J\]. arXiv preprint, 2017.

\[2\] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

\[3\] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.

\[4\] Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector\[C\]//European conference on computer vision. Springer, Cham, 2016: 21-37.

\[5\] G. A. Miller, R. Beckwith, C. Fellbaum, D. Gross, and K. J. Miller. Introduction to wordnet: An on-line lexical database. International journal of lexicography, 3(4):235–244, 1990.







