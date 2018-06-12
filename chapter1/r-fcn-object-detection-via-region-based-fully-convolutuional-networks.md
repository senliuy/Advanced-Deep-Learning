# R-FCN: Object Detection via Region-based Fully Convolutuional Networks

## 简介：

位移不变性是卷积网络一个重要的特征，该特征使得卷积网格在图像分类任务上取得了非常好的效果。但是在物体检测的场景中，我们需要知道检测物体的具体位置，这时候我们需要网络对物体的位置非常敏感，即我们需要网络具有“位移可变性”。R-FCN \[1\]的提出便是解决分类任务中位移不变性和检测任务中位移可变性直接的矛盾的。

同时，作者分析了Faster R-CNN \[2\]存在的性能瓶颈，即ROI之后使用Fast R-CNN \[3\]对RPN提取的候选区域进行分类和位置精校。在R-FCN中，ROI之后便不存在可学习的参数，从而将Faster-RCNN的速度提高了2.5-20倍。

## 1. 动机

## Reference

\[1\] Dai J, Li Y, He K, et al. R-fcn: Object detection via region-based fully convolutional networks\[C\]//Advances in neural information processing systems. 2016: 379-387.

\[2\] Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks \(2015\), in Neural Information Processing Systems \(NIPS\)

\[3\] R. Girshick, “Fast R-CNN,” in IEEE International Conference on Computer Vision \(ICCV\), 2015.

