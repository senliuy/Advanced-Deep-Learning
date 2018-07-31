# Learning to Segment Every Thing

## 前言

首先回顾一下YOLO系列论文，[YOLO9000](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/yolo9000-better-faster-stronger.html)\[2\]通过半监督学习的方式将模型可检测的类别从80类扩展到了9418类，YOLO9000能成功的原因之一是物体分类和物体检测使用了共享的特征，而这些特征是由分类和检测的损失函数共同训练得到的。之所以采用半监督学习的方式训练YOLO9000，一个重要原因就是检测数据的昂贵性。所以作者采用了数据量较小的COCO的检测标签，数据量很大的ImageNet的分类标签做半监督学习的样本，分别训练多任务模型的检测分支和分类分支，进而得到了可以同时进行分类和检测的特征。

之所以在最开始回顾YOLO9000，是因为我们这里要分析的$$\mathbf{Mask}^X$$** R-CNN** 和YOLO9000的动机和设计是如此的相同：

1. 他们都是在多任务模型中使用半监督学习来完成自己的任务，YOLO9000是用来做检测，$$\mathbf{Mask}^X$$** R-CNN** 是用来做语义分割；

2. 之所以使用半监督学习，因为他们想实现一个通用的模型，所以面临了数据量不够的问题：对比检测任务，语义分割的数据集更为稀缺（COCO的80类，Pascal VOC的20类），但是Visual Genome（VG）数据集却有3000类，108077张带有bounding box的样本；

3. 它们的框架算法都是继承自另外的框架：YOLO9000继承自YOLOv2, $$\mathbf{Mask}^X$$** R-CNN** 继承自[Mask R-CNN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/mask-r-cnn.html)\[3\]。

不同于YOLO9000通过构建WordTree的数据结构来使用两个数据集，$$\mathbf{Mask}^X$$** R-CNN** 提出了一个叫做权值迁移函数\(weight transfer function\)的迁移学习方法，将物体检测的特征迁移到语义分割任务中，进而实现了对VG数据集中3000类物体的语义分割。这个权值传递函数便是$$\mathbf{Mask}^X$$** R-CNN** 的精华所在。

### $$\mathbf{Mask}^X$$** R-CNN** 详解

#### 1. 权值迁移函数：$$\mathcal{T}$$

$$\mathbf{Mask}^X$$** R-CNN** 基于Mask R-CNN（图1）。Mask R-CNN通过向[Faster R-CNN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.html)\[4\]中添加了一路FPN的分支来达到同时进行语义分割和目标检测的目的。在RPN之后，FPN和[Fast R-CNN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/fast-r-cnn.html)\[5\]是完全独立的两个模块，此时若直接采用数据集C分别训练两个分支的话是行得通的，其实这也就是YOLO9000的训练方式。

###### 图1： Mask R-CNN 流程

![](/assets/Mask_R-CNN_1.png)

但是$$\mathbf{Mask}^X$$** R-CNN** 不会这么简单就结束的，它在检测分支\(Fast R-CNN\)和分割分支中间加了一条叫做权值迁移函数的线路，用于将检测的信息传到分割任务中，如图2所示。

###### 图2： $$\mathbf{Mask}^X$$** R-CNN** 流程

![](/assets/MaskX_RCNN.png)

图2的整个框架是搭建在Mask R-CNN之上的，除了最重要的权值迁移函数之外，还有几点需要强调一下：

1. 着重强调，$$\mathcal{T}$$ 的输入参数是**权值**（图2中的两个六边形），而非Feature Map；
2. 虽然Mask R-CNN中解耦了检测和分割任务，但是权值迁移函数$$\mathcal{T}$$是类别无关的；

对于一个类别$$c$$，$$w_{det}^c$$表示检测任务的权值，$$w_{seg}^c$$表示分割任务的权值。权值迁移函数将$$w_{det}^c$$看做自变量，$$w_{seg}^c$$看做因变量，学习两个权值的映射函数$$\mathcal{T}$$：


$$
W_{seg}^c = \mathcal{T}(w_{det}^c; \theta)
$$


其中$$\theta$$的是类别无关的，可学习的参数。$$\mathcal{T}$$ 可以使用一个小型的MLP。$$w_{det}^c$$可以使分类的权值$$w_{cls}^c$$，bounding box的预测权值$$w_{reg}^c$$或是两者拼接到一起$$[w_{cls}^c，w_{reg}^c]$$。

#### 2. $$\mathbf{Mask}^X$$** R-CNN** 的训练

含有掩码标签和检测标签的COCO数据集定义为$$A$$，只含有检测标签的VG数据集定义为$$B$$，则所有的数据集$$C$$便是$$A$$和$$B$$的并集：$$C=A\cup B$$。图2显示$$\mathbf{Mask}^X$$** R-CNN**的损失函数由Fast R-CNN的检测任务和RPN的分类任务组成。当训练检测任务时，使用数据集C；当训练分割任务时，仅使用包括分割标签的COCO数据集，即$$A$$。

训练$$\mathbf{Mask}^X$$** R-CNN**时，有两种类型：

1. 多阶段训练：


## Reference

\[1\] Hu R, Dollár P, He K, et al. Learning to segment every thing\[J\]. Cornell University arXiv Institution: Ithaca, NY, USA, 2017.

\[2\] Redmon J, Farhadi A. YOLO9000: better, faster, stronger\[J\]. arXiv preprint, 2017.

\[3\] He K, Gkioxari G, Dollár P, et al. Mask r-cnn\[C\]//Computer Vision \(ICCV\), 2017 IEEE International Conference on. IEEE, 2017: 2980-2988.

\[4\] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.

\[5\] R. Girshick, “Fast R-CNN,” in IEEE International Conference on Computer Vision \(ICCV\), 2015.

