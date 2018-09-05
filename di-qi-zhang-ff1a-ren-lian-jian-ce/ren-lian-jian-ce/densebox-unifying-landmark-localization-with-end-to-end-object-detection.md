# DenseBox: Unifying Landmark Localization with End to End Object Detection

## 前言

DenseBox百度IDL的作品，提出的最初动机是为了解决普适的物体检测问题。其在2015年初就被提出来了，甚至比Fast R-CNN还要早，但是由于论文发表的比较晚，虽然算法上非常有创新点，但是依旧阻挡不了Fast R-CNN一统江山。

DenseBox的主要贡献如下：

1. 使用全卷积网络，任务类型类似于语义分割，并且实现了端到端的训练和识别，而R-CNN系列算法是从Faster R-CNN中使用了RPN代替了Selective Search才开始实现端到端训练的，而和语义分割的结合更是等到了2017年的Mask R-CNN才开始；
2. 多尺度特征，而R-CNN系列直到FPN才开始使用多尺度融合的特征；
3. 结合关键点的多任务系统，DenseBox的实验是在人脸检测数据集（MALF）上完成的，结合数据集中的人脸关键点可以使算法的检测精度进一步提升。

## 1. DenseBox详解

### 1.1 训练标签

DenseBox没有使用整幅图作为输入，因为作者考虑到一张图上的背景区域太多，计算时间会严重浪费在对没用的背景区域的卷积上。而且使用扭曲或者裁剪将不同比例的图像压缩到相同尺寸会造成信息的丢失。作者提出的策略是从训练图片中裁剪出包含人脸的patch，这些patch包含的背景区域足够完成模型的训练，详细过程如下：

1. 根据Ground Truth从训练数据集中裁剪出大小是人脸的区域的高的4.8倍的正方形作为一个patch，且人脸在这个patch的中心；
2. 将这个patch resize到$$240\times240$$大小。

举例说明：一张训练图片中包含一个$$60\times80$$的人脸，那么第一步会裁剪出大小是$$384\times384$$的一个patch。在第二步中将这个patch resize到$$240\times240$$。这张图片便是训练样本的输入。

训练集的标签是一个$$60\times60\times5$$的热图（图1），$$60\times60$$表示热图的尺寸，从这个尺寸我们也可以看出训练样本经过了两次降采样。$$5$$表示热图的通道数，组成方式如下：

<figure>
<img src="/assets/DenseBox_1.png" alt="图1：DenseBox的Ground Truth" />
<figcaption>图1：DenseBox的Ground Truth</figcaption>
</figure>

1. 图1中最前面的热图用于标注人脸区域置信度，前景为1，背景为0。DenseBox并没有使用左图的人脸矩形区域而是使用半径（$$r_c$$）为Ground Truth的高的0.3倍的圆作为标签值，而圆形的中心就是热图的中心，即有图中的白色圆形部分；

2. 图1中后面的四个热图表示像素点到最近的Ground Truth的四个边界的距离，如图2所示。

<figure>
<img src="/assets/DenseBox_2.png" alt="图2：DenseBox中距离热图示意图" width="600"/>
<figcaption>图2：DenseBox中距离热图示意图</figcaption>
</figure>

如果训练样本中的人脸比较密集，一个patch中可能出现多个人脸，如果某个人脸和中心点处的人脸的高的比例在$$[0.8,1.25]$$之间，则认为该样本为正样本。

作者认为DenseBox的标签设计是和感受野密切相关的，具体的讲，结合1.2节要分析的网络结构我们可以计算得到热图中每个像素的感受野是$$48\times48$$，这和我们每个patch中每个人脸的尺寸是非常接近的。在DenseBox中，每个像素点有5个预测值，而这5个预测值便可以确定一个检测框，所以DenseBox本质上也是一个密集采样，每个图片的采样个数是$$60\times60=3600$$个。

### 1.2 网络结构

DenseBox使用了16层的VGG-19作为骨干网络，但是只使用了其前12层，如图3所示。

<figure>
<img src="/assets/DenseBox_3.png" alt="图3：DenseBox中的网络结构" width="600"/>
<figcaption>图3：DenseBox中的网络结构</figcaption>
</figure>



## Reference

\[1\] Qin H, Yan J, Li X, et al. Joint training of cascaded cnn for face detection\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 3456-3465.