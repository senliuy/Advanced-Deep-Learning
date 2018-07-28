# YOLOv3: An Incremental Improvement

## 前言

YOLOv3论文的干货并不多，用作者自己的话说是一篇“Tech Report”。这篇主要是在[YOLOv2](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/yolo9000-better-faster-stronger.html) \[2\]的基础上的一些Trick尝试，有的Trick成功了，包括：

1. 考虑到检测物体的重叠情况，用多标签的方式替代了之前softmax单标签方式；
2. 骨干架构使用了更为有效的残差网络，网络深度也更深；
3. 多尺度特征使用的是[FPN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/mask-r-cnn.html) \[3\]的思想；
4. 锚点聚类成了9类。

也有一些尝试失败了，在介绍完YOLOv3的细节后我们在说明这些尝试会更好理解。在分析论文时，我们依然会使用一份Keras的[源码](https://github.com/qqwweee/keras-yolo3)辅助理解。

## YOLOv3详解

### 1. 多标签任务

不管是在检测任务的标注数据集，还是在日常场景中，物体之间的相互覆盖都是不能避免的。因此一个锚点的感受野肯定会有包含两个甚至更多个不同物体的可能，在之前的方法中是选择和锚点IoU最大的Ground Truth作为匹配类别，用softmax作为激活函数。

YOLOv3多标签模型的提出，对于解决覆盖率高的图像的检测问题效果是十分显著的，图1是同一幅图在YOLOv2和YOLOv3下得到的检测结果。可以明显的看出YOLOv3的效果好很多，不仅检测的更精确，最重要的是在后排被覆盖很多的物体（例如美队和冬兵）也能很好的在YOLOv3中检测出来。

###### 图1：YOLOv2 vs YOLOv3![](/assets/YOLOv3_1_1.jpg)![](/assets/YOLOv3_1_2.jpg)

YOLOv3提供的解决方案是将一个$$N$$ 路softmax分类器替换成$$N$$ 个sigmoid分类器，这样每个类的输出仍是$$[0,1]$$ 之间的一个值，但是他们的和不再是1。

虽然YOLOv3改变了输出层的激活函数，但是其锚点和Ground Truth的匹配方法仍旧采用的是YOLOv1 \[4\]的方法，即每个Ground Truth匹配且只匹配唯一一个与其IoU最大的锚点。但是在输出的时候由于各类的概率之和不再是1，只要置信度大于阈值，该锚点便被作为检测框输出。

训练标签的制作和测试过程候选框的输出分别在`./yolo3/model.py`的`yolo_eval`和`preprocess_true_boxes`函数中实现的。

### 2. 骨干网络

YOLOv3使用了由残差块构成的全卷积网络作为骨干网络，网络深度达到了53层，因此作者将其命名为Darknet-53。Darknet-53的详细结构见图2。

###### 图2：Darknet-53

![](/assets/YOLOV3_2.png)

### 3. 多尺度特征

YOLOv3汲取了[FPN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/mask-r-cnn.html)的思想，从不从尺度上提取了特征。对比YOLOv的只在最后两层提取特征，YOLOv3则将尺度扩大到了最后三层，图3是在图2的基础上加上多尺度特征提取部分的图示。

###### 图3：Darknet-53 with FPN

![](/assets/YOLOv3_3.png)

在多尺度特征部分强调几个关键点：
1. YOLOv2采用的是降采样的形式进行Feature Map的拼接，YOLOv3则是采用同SSD相同的双线性插值的上采样方法拼接的Feature Map；
2. 每个尺度的Feature Map负责对3个先验框（锚点）的预测，源码中的掩码（Mask）负责完成此任务。

### 4. 锚点聚类

在YOLOv2的文章中我们介绍了锚点是聚类的，作者尝试了折中考虑了速度和精度之后选择的类别数$$k=5$$。但是在YOLOv3中，$$k=9$$，得到的9组锚点是: $$(10\times 13), (16\times 30), (33\times 23), (30\times 61), (62\times 45), (59\times 119), (116\times 90), (156\times 198), (373\times 326)$$

其中$$13\times 13$$的卷积核分配的尺度是$$(116\times 90), (156\times 198), (373\times 326)$$, $$26\times 26$$的卷积核分配的尺度是$$(30\times 61), (62\times 45), (59\times 119)$$, $$52\times 52$$的卷积核分配的尺度是$$(10\times 13), (16\times 30), (33\times 23))$$。这么做的原因是深度学习中层数越深，Feature Map对小尺寸物体的响应能力越弱。

### 5. YOLOv3一些失败的尝试

1. 尝试捕捉位移$$(x,y)$$ 和检测框边长$$(w, h)$$的线性关系，这时方式得到的效果并不好且模型不稳定；
2. 使用线性激活函数代替sigmoid激活函数预测位移$$(x,y)$$，该方法导致模型的mAP下降；
3. 使用focal loss\[5\], mAP也降了。

## 总结

至此，YOLO系列的算法整理完毕，作者的兴趣点也转向了GAN，感觉短期内不会有大的进展了。

从算法的角度讲，当业界都沉迷于R-CNN系列的方法时，作者另辟蹊径引入了单次检测的YOLO，虽然效果略差，但是其速度优势也占据了很大市场。但是作者并就看不起R-CNN系列，在YOLOv2中引入了RPN的锚点机制，在YOLOv3中引入了FPN，正所谓师夷长技以制夷，一段时间内从精度和时间实现了对R-CNN系列的全面压制。





