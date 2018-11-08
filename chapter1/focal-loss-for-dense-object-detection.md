# Focal Loss for Dense Object Detection

## 前言

何凯明，RBG等人一直是Two-Stage方向的领军人，在这篇论文中，他们也开始涉足One-Stage的物体检测算法。大牛就是牛，一次就刷新了精度。下面我们就来分析这几个大牛的作品。

目前主流的检测算法分为两个方向：（1）以R-CNN\[2\]系列为代表的two-stage方向；（2）以YOLO\[3\]系列为代表的one-stage方向。虽然one-stage方向的速度更快，但是其精度往往比较低。究其原因，有两个方面：

1. 正样本（Positive Example）和负样本（Negative Example）的不平衡；
2. 难样本（Hard Example）和易样本（Easy Example）的不平衡。

这些不平衡造成模型的效果不准确的原因如下：

1. Negative example的数量过多，导致Postive example的loss被覆盖，就算Postive example的loss非常大也会被数量庞大的 negative example中和掉，这这些positive example往往是我们要检测的前景区域；

2. Hard example往往是前景和背景区域的过渡部分，因为这些样本很难区分，所以叫做Hard Example。剩下的那些Easy example往往很好计算，导致模型非常容易就收敛了。但是损失函数收敛了并不代表模型效果好，因为我们其实更需要把那些hard example训练好。

四种Example的情况见图1。

![](/assets/RetinaNet_1.png)

解决正负样本的不平衡的传统策略是使用平衡的交叉熵损失函数，对于难易样本的不平衡通常是使用Hard Negative Mining的策略。

而作者的解决方案是基于交叉熵提出了一个新的损失函数Focal Loss（FL）。


$$
\text{FL}(p_t) = - \alpha_t (1-p_t)^{\gamma}log(p_t)
$$


FL是一个尺度动态可调的交叉熵损失函数，在FL中有两个参数$$\alpha_t$$和$$\gamma$$，其中$$\alpha_t$$主要作用是解决正负样本的不平衡，$$\gamma$$主要是解决难易样本的不平衡。

最后，作者基于残差网络\[4\]，FPN\[5\]搭建了检测网络RetinaNet，该网络使用的策略都是他们自己提出的而且目前效果非常好的基础结构，再结合Focal Loss，刷新检测算法的精度也不意外。

## 1. Focal Loss

## 2. RetinaNet

## 3. 总结

## Reference

\[1\] Lin T Y, Goyal P, Girshick R, et al. Focal loss for dense object detection\[J\]. IEEE transactions on pattern analysis and machine intelligence, 2018.

\[2\] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in CVPR, 2014

\[3\] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

\[4\] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

\[5\] T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, and ´ S. Belongie. Feature pyramid networks for object detection. In CVPR, 2017. 2, 4, 5, 7

