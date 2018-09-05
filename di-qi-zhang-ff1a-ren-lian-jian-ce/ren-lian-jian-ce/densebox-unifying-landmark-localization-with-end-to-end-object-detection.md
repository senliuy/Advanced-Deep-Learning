# DenseBox: Unifying Landmark Localization with End to End Object Detection

## 前言

DenseBox百度IDL的作品，提出的最初动机是为了解决普适的物体检测问题。其在2015年初就被提出来了，甚至比Fast R-CNN还要早，但是由于论文发表的比较晚，虽然算法上非常有创新点，但是依旧阻挡不了Fast R-CNN一统江山。

DenseBox的主要贡献如下：

1. 使用全卷积网络，任务类型类似于语义分割，并且实现了端到端的训练和识别，而R-CNN系列算法是从Faster R-CNN中使用了RPN代替了Selective Search才开始实现端到端训练的，而和语义分割的结合更是等到了2017年的Mask R-CNN才开始；
2. 多尺度特征，而R-CNN系列直到FPN才开始使用多尺度融合的特征；
3. 结合关键点的多任务系统，DenseBox的实验是在人脸检测数据集（MALF）上完成的，结合数据集中的人脸关键点可以使算法的检测精度进一步提升。

## 1. DenseBox详解

### 1.1 训练标签

DenseBox没有使用整幅图作为输入，因为作者考虑到一张图上的背景区域太多，计算时间会严重浪费在对没用的背景区域的卷积上。

## Reference

\[1\] Qin H, Yan J, Li X, et al. Joint training of cascaded cnn for face detection\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 3456-3465.