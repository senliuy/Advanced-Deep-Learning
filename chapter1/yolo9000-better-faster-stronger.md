# YOLO9000: Better, Faster, Stronger

## 前言

在这篇像极了奥运格言（Faster，Higher，Stronger）的论文\[1\]中，作者提出了YOLOv2和YOLO9000两个模型。

其中YOLOv2采用了若干技巧对[YOLOv1](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/you-only-look-once-unified-real-time-object-detection.html)的速度和精度进行了提升。其中比较有趣的有以下几点：

1. 使用聚类产生的锚点代替[Faster R-CNN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.html)和[SSD](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/ssd-single-shot-multibox-detector.html)手工设计的锚点；
2. 在高分辨率图像上进行迁移学习，提升网络对高分辨图像的响应能力；
3. 训练过程图像的尺寸不再固定，提升网络对不同训练数据的泛化能力。

除了以上三点，YOLO还使用了残差网络的直接映射的思想，R-CNN系列的预测相对位移的思想，Batch Normalization，全卷积等思想。YOLOv2将算法的速度和精度均提升到了一个新的高度。正是所谓的速度更快（Faster），精度更高（Better/Higher）

论文中提出的另外一个模型YOLO9000非常巧妙的使用了WordNet\[5\]的方式将检测数据集COCO和分类数据集ImageNet整理成一个多叉树，再通过提出的联合训练方法高效的训练多叉树对应的损失函数。YOLO9000是一个非常强大（Stronger）且有趣的模型，非常具有研究前景。

在下面的章节中，我们将论文分成YOLOv2和YOLO9000两个部分并结合论文和源码对算法进行详细解析。

## 1. YOLOv2: Better, Faster

YOLOv2使用的是和YOLOv1相同的思路，算法流程参考[YOLOv1](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/you-only-look-once-unified-real-time-object-detection.html)的介绍，但看这篇文章肯定会感到一头雾水，因为论文并没有详细介绍YOLOv2的详细流程。而且我也不打算介绍，因为这只是对YOLOv1无畏的重复，强烈推荐读者能搞懂YOLOv1之后再来读这篇文章。

### 1.1. Better

YOLOv1之后，一系列算法和技巧的提出极大的提高了深度学习在各个领域的泛化能力。作者总结了可能在物体检测中有用的方法和技巧（图1）并将它们结合成了我们要介绍的YOLOv2。所以YOLOv2并没有像SSD或者Faster R-CNN具有很大的难度，更多的是在YOLOv1基础上的技巧方向的提升。在下面的篇幅中，我们将采用和论文相同的结构并结合基于Keras的[源码](https://github.com/yhcc/yolo2)对YOLOv2中涉及的技巧进行讲解。

###### 图1：YOLOv2中使用的技巧及带来的性能提升

![](/assets/YOLOv2_1.png)

#### 1.1.1. Batch Normalization

YOLOv2中作者舍弃了Dropout而使用Batch Normalization（BN）来减轻模型的过拟合问题，从图1中我们可以看出BN带来了2.4%的mAP的性能提升。

Batch Normalization和Dropout均有正则化的作用。但是Batch Normalization具有提升模型优化的作用，这点是Dropout不具备的。所以BN更适合用于数据量比较大的场景。

关于BN和Dropout的异同，可以参考Ian Goodfellow在Quora上的[讨论](https://www.quora.com/What-is-the-difference-between-dropout-and-batch-normalization#)。

#### 1.1.2. High Resolution Classifier

之前的深度学习模型很多均是生搬在ImageNet上训练好的模型做迁移学习。由于迁移学习的模型是在尺寸为$$224\times224$$的输入图像上进行训练的，进而限制了检测图像的尺寸也是$$224\times224$$。在ImageNet上图像的尺寸一般在500左右，降采样到224的方案对检测任务的负面影响要远远大于分类任务。

为了提升模型对高分辨率图像的响应能力，作者先使用尺寸为$$448\times448$$的ImageNet图片训练了10个Epoch（并没有训练到收敛，可能考虑$$448\times448$$的图片的一个Epoch时间要远长于$$224\times224$$的图片），然后再在检测数据集上进行模型微调。图1显示该技巧带来了3.7%的性能提升

#### 1.1.3。 Convolution With Anchor Boxes

YOLOv2使用了DarkNet-19作为骨干网络（图2），在这里我们需要注意两点：

1. YOLOv2输入网络的图像尺寸并不是图2画的$$224\times224$$, 而是使用了$$416\times416$$的输入图像，原因我们随后会介绍；
2. 在$$3\times3$$卷积中间添加了$$1\times1$$卷积，Feature Map之间的一层非线性变化提升了模型的表现能力；
3. Darknet-19进行了5次降采样，但是在最后一层卷积并没有添加池化层，目的是为了获得更高分辨率的Feature Map；
4. Darknet-19中并不含有全连接，使用的是全局平均池化的方式产生长度固定的特征向量。

###### 图2：Darknet网络结构

![](/assets/YOLOv2_2.png)

首先，YOLOv2使用的是$$416\times416$$的输入图像，考虑到很多情况下待检测物体的中心点容易出现在图像的中央，所以使用$$416\times416$$经过5次降采样之后生成的Feature Map的尺寸是$$13\times13$$，这种奇数尺寸的Feature Map获得的中心点的特征向量更准确。其实这也和YOLOv1产生$$7\times7$$的理念是相同的。

其次，YOLOv2也学习了Faster R-CNN和RPN中的锚点机制，锚点也可以被叫做先验框，即给出一个检测框可能的形状，向先验框的收敛总是比向固定box的收敛要容易的多。不同于以上两种算法的是YOLOv2使用的是在训练集上使用了K-means聚类产生的候选框，而上面两种方法的候选框是人工设计的。

最后，关于全卷积的作用，$$1\times1$$卷积带来的非线性变化我们已经在之前的文章中多次提及，这里便不再说明。

#### 1.1.4. Dimension Clusters

在1.1.3节中我们介绍到YOLOv2使用的

### 1.2. Stronger

## YOLO9000: Stronger

## Reference

\[1\] Redmon J, Farhadi A. YOLO9000: better, faster, stronger\[J\]. arXiv preprint, 2017.

\[2\] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

\[3\] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.

\[4\] Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector\[C\]//European conference on computer vision. Springer, Cham, 2016: 21-37.

\[5\] G. A. Miller, R. Beckwith, C. Fellbaum, D. Gross, and K. J. Miller. Introduction to wordnet: An on-line lexical database. International journal of lexicography, 3\(4\):235–244, 1990.

