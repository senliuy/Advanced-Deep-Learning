# You Only Look Once: Unified, Real-Time Object Detection

## 前言

在R-CNN系列的论文中，目标检测被分成了候选区域提取和候选区域分类及精校两个阶段。不同于这些方法，YOLO将整个目标检测任务整合到一个回归网络中。对比Fast R-CNN提出的两步走的端到端方案，YOLO的单阶段的使其是一个更彻底的端到端的算法（图1）。YOLO的检测过程分为三步：

1. 图像Resize到448\*448；
2. 将图片输入卷积网络；
3. NMS得到最终候选框。

###### 图1：YOLO算法框架

\[YOLOv1\_1\]

虽然在一些数据集上的表现不如Fast R-CNN及其后续算法，但是YOLO带来的最大提升便是检测速度的提升。在YOLO算法中，检测速度达到了每秒45帧，而一个更快速的Fast Yolo版本则达到了155帧/秒。另外在YOLO的背景检测错误率要低于Fast R-CNN。最后，YOLO算法具有更好的通用性，通过Pascal数据集训练得到的模型在艺术品问检测中得到了比Fast R-CNN更好的效果。

YOLO是可以用在Fast R-CNN中的，结合YOLO和Fast R-CNN两个算法，得到的效果比单Fast R-CNN要更好。

下面，我们结合YOLO的[TensorFlow源码](https://github.com/nilboy/tensorflow-yolo)详细解析YOLO算法的每个技术细节和算发动机。

## YOL算法详解

### 1. 统一检测

YOLO检测速度远远超过R-CNN系列的重要原因是YOLO将整个物体检测统一成了一个回归问题。YOLO的输入是整张待检测图片，输出则是得到的检测结果，整个过程只经过一次卷积网络。Faster R-CNN虽然使用全卷积的思想实现了候选区域的权值共享，但是每个候选区域的特征向量任然要单独的计算分类概率和bounding box。

YOLO实现统一检测的方法是增加网络的输出节点数量，其实也算是空间换时间的一种策略。在Faster R-CNN的Fast R-CNN部分，网络有分类和回归两个任务，网络输出节点个数是C+5，其中K是数据集的类别个数。而YOLO的输出层O节点个数达到了S\*S\*\(C+B\*5\)，下面我们来讲解输出节点每个字符的含义。

#### 1.1 S\*S 窗格

YOLO将输入图像分成S\*S的窗格（Grid），如果Ground Truth的中心落在某个Grid单元（Ceil）内，则该单元负责该物体的检测，如图2所示。

###### 图2：S\*S窗格

\[YOLOv1\_2\]

什么是某个单元负责落在该单元内的物体检测呢？举例说明一下，首先我们将输出层O\_{S\times S\ times \(C+B\*5\)}看做一个三维矩阵，如果物体的中心落在第\(i,j\)个单元内，那么网络只优化一个C+B\*5维的向量，即向量O\[i,j,:\]。S是一个超参数，在源码中，S=7，即配置文件`./yolo/config.py`的CELL\_SIZE变量。

```
CELL_SIZE = 7
```

#### 1.2 Bounding Box

B是每个单元预测的Bounding box的数量，B的个数同样是一个超参数。在`./yolo/config.py`文件中B=2，YOLO使用多个Bounding box是为了每个cell计算top-B个可能的预测结果，这样做虽然牺牲了一些时间，但却提升了模型的检测精度。

```
BOXES_PER_CELL = 2
```

注意不管YOLO使用了多少个Bounding box，每个cell的Bounding box均有相同的优化目标值。在`./yolo/yolo_net.py`中，Ground Truth的label值被复制了B次。

```
boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
```

每个Bounding box要预测5个值：x,y,w,h以及置信度。其中\(x,y\)是box

## Reference

\[1\] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

