# Scene Text Detection via Holistic, Multi-Channel Prediction

## 前言

本文是在边缘检测经典算法[HED](https://senliuy.gitbooks.io/advanced-deep-learning/content/qi-ta-ying-yong/holistically-nested-edge-detection.html)\[2\]之上的扩展，在这篇论文中我们讲过HED算法可以无缝转移到语义分割场景中。而这篇论文正是将场景文字检测任务转换成语义分割任务来实现HED用于文字检测的。图1是HED在身份证上进行边缘检测得到的掩码图，从图1中我们可以看出HED在文字检测场景中也是有一定效果的。

###### 图1：HED在身份证上得到的掩码图

![](/assets/HMCP_1.png)

HED之所以能用于场景文字检测一个重要的原因是文字区域具有很强的边缘特征。

论文的题目为Holistic，Multi-Channel Prediction（HMCP），其中Holistic表示算法基于HED，Multi-Channel表示该算法使用多个Channel的标签训练模型。也就是为了提升HED用于文字检测的精度，这篇文章做的改进是将模型任务由单任务模型变成是由文字分割，单词分割和单词之间的相对关系构成的多任务系统。由于HMCP采用的是语义分割的形式，所以其检测框可以扩展到多边形或者是带旋转角度的四边形，这也更符合具有严重仿射变换的真实场景。

## 1. HMCP详解

HMCP的流程如图2：(a)是输入图像，(b)是预测的三个mask，分别是文本行掩码，字符掩码和字符间关系掩码，(c)便是根据(b)的三个掩码得到的检测框。

###### 图2：HMCP流程

![](/assets/HMCP_2.png)

那么问题来了：
1. 如何构建三个channel掩码的标签值；
2. 如何根据预测的掩码构建文本行。

### 1.1 HMCP的标签值



## Reference

\[1\] Yao C, Bai X, Sang N, et al. Scene text detection via holistic, multi-channel prediction\[J\]. arXiv preprint arXiv:1606.09002, 2016.

\[2\] Xie S, Tu Z. Holistically-nested edge detection \[C\]//Proceedings of the IEEE international conference on computer vision. 2015: 1395-1403.

