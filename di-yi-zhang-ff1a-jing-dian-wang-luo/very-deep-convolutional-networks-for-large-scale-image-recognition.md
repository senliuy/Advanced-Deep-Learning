# Very Deep Convolutional NetWorks for Large-Scale Image Recognition

## 1. 前言

时间来到2014年，随着AlexNet在ImageNet数据集上的大放异彩，探寻针对ImageNet的数据集的最优网络成为了提升该数据集精度的一个最先想到的思路。牛津大学计算机视觉组（Visual Geometry Group）和Google的Deep Mind的这篇论文便是对卷积网络的深度和其性能的探索，由此该网络也被命名为VGG。

VGG的结构非常清晰：

* 按照2\*2的Pooling层，网络可以分成若干段；
* 每段之内由若干个same卷机操作构成，段之内的Feature Map数量固定不变；
* 第i段的Feature Map数量是第i-1段的Feature Map数量的2倍

VGG的结构非常容易扩展到其它数据集。在VGG中，段数每增加1，Feature Map的尺寸减少一半，所以通过减少段的数目将网络应用到例如MNIST，CIFAR等图像尺寸更小的数据集。段内的卷积的数量是可变的，因为卷积的个数并不会影响图片的尺寸，我们可以根据任务的复杂度自行调整段内的卷积数量。

VGG的表现效果也非常好，在ILSVRC2014 \[1\]分类中排名第二（第一是GoogLeNet \[2\]，没有办法），定位比赛排名第一。

VGG将其模型开源在其官方网站上，为其它任务提供了非常好的迁移学习的材料，使得VGG马上占有了大量商业市场。关于不同框架的VGG模型，自行在晚上搜索。

## 2. VGG介绍

VGG关于网络结构的探索可以总结为图1，图1包含了大量信息，我们一一分析之

###### 图1：VGG家族

\[VGG\_1\]

图2反应了VGG家族的各个模型的性能

图2：VGG家族的性能表现

\[VGG\_2\]

### 2.1 家族的特征

我们来看看VGG家族的共同特征

* 输入图像的尺寸均是224\*224；
* 均为5层Max Pooling，表示最终均会产生大小为7\*7的Feature Map，这是一个大小比较合适的尺寸；
* 卷积部分之后（特征层）跟的是两个隐层节点数目为4096的全连接，最后接一个1000类softmax分类器。

### 2.2 VGG-A vs VGG-A-LRN

VGG A-LRN 比 VGG A多了一个AlexNet介绍的LRN层，但是实验数据表明加入加入LRN的VGG-A错误率反而更高了。二期LRN的加入会更加占用内存消耗以及增加训练时间。

### 2.3 VGG-A vs VGG-B vs VGG-D vs VGG-E

对比VGG-A\(11层\), VGG-B\(13层\), VGG-D\(16层\), VGG-E\(19层\)的错误率，我们发现随着网络深度的增加，分类的错误率逐渐降低，当然越深的深度则表示需要的训练时间越长。但是当模型的深度到达一定深度时（VGG-D和VGG-E），错误率的提升便变的非常有限，甚至偶尔会发生深层网络的错误率高于浅层网络的情况，同时考虑网络的训练时间，我们就要折中考虑选择合适的网络深度了。我相信作者一定探索了比VGG-E更深的网络，但是由于表现的不理想并没有列在论文中。后面介绍的残差网络则通过shortcut的机制将网络的深度理论上扩展到了无限大。

## Reference

\[1\] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition\[J\]. arXiv preprint arXiv:1409.1556, 2014.

\[2\] Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions\[C\]. Cvpr, 2015.
