# Very Deep Convolutional NetWorks for Large-Scale Image Recognition

## 1. 前言

时间来到2014年，随着AlexNet在ImageNet数据集上的大放异彩，探寻针对ImageNet的数据集的最优网络成为了提升该数据集精度的一个最先想到的思路。牛津大学计算机视觉组（Visual Geometry Group）和Google的Deep Mind的这篇论文便是对卷积网络的深度和其性能的探索，由此该网络也被命名为VGG。

VGG的结构非常清晰：

* 按照2\*2的Pooling层，网络可以分成若干段；
* 每段之内由若干个same卷机操作构成，段之内的Feature Map数量固定不变；
* Feature Map按段以2倍的速度逐渐递增，第四段和第五段都是512（64-128-256-512-512）。

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

关于VGG家族的keras实现和参数统计见链接（[https://github.com/senliuy/CNN-Structures/blob/master/VGG.ipynb](https://github.com/senliuy/CNN-Structures/blob/master/VGG.ipynb)）。

### 2.1 家族的特征

我们来看看VGG家族的共同特征

* 输入图像的尺寸均是224\*224；
* 均为5层Max Pooling，表示最终均会产生大小为7\*7的Feature Map，这是一个大小比较合适的尺寸；
* 卷积部分之后（特征层）跟的是两个隐层节点数目为4096的全连接，最后接一个1000类softmax分类器。

VGG在卷积核方向的最大改进是将卷积核全部换成更小的3\*3或者1\*1的卷积核，而性能最好的VGG-16和VGG-19由且仅有3\*3卷积构成。原因有如下两点：

1. 一个7\*7的卷积核和3层3\*3的卷积核具有相同的感受野，但是由于3层感受野具有更深的深度，由此可以构建更具判别性的决策函数；
2. 假设Feature Map的数量都是C，3层3\*3卷积核的参数个数是3\*\(3\*3+1\)\*C = 30C，1层7\*7卷积核的参数个数是1\*\(7\*7+1\)\*C=50C, 3层3\*3卷积核具有更少的参数。
3. 但由于神经元数量和层数的增多，训练速度会变得更慢

下图是把LeNet-5的5\*5卷积换成了两层3\*3卷积在MNIST上的收敛表现，实验表明两层3\*3的网络确实比单层5\*5的网络表现好，但是训练速度也慢了一倍。

###### 图3：3\*3 LeNet vs 5\*5 LeNet

\[VGG\_3\]

另外，作者在前两层的全连接出使用drop rate = 0.5的Dropout，然而并没有在图1中反应出来。

### 2.2 VGG-A vs VGG-A-LRN

VGG A-LRN 比 VGG A多了一个AlexNet介绍的LRN层，但是实验数据表明加入加入LRN的VGG-A错误率反而更高了。二期LRN的加入会更加占用内存消耗以及增加训练时间。

### 2.3 VGG-A vs VGG-B vs VGG-D vs VGG-E

对比VGG-A\(11层\), VGG-B\(13层\), VGG-D\(16层\), VGG-E\(19层\)的错误率，我们发现随着网络深度的增加，分类的错误率逐渐降低，当然越深的深度则表示需要的训练时间越长。但是当模型的深度到达一定深度时（VGG-D和VGG-E），网络的错误率趋近饱和，甚至偶尔会发生深层网络的错误率高于浅层网络的情况，同时考虑网络的训练时间，我们就要折中考虑选择合适的网络深度了。我相信作者一定探索了比VGG-E更深的网络，但是由于表现的不理想并没有列在论文中。后面介绍的残差网络则通过shortcut的机制将网络的深度理论上扩展到了无限大。

### 2.4 VGG-B vs VGG-C

VGG-C在VGG-B的基础上添加了3个1\*1的卷积层，1\*1的卷积是在NIN\[3\]中率先使用的，由于1\*1卷积在不影响感受野的前提提升了决策函数的非线性性，由此带来了错误率的下降。

### 2.5 VGG-C vs VGG-D

VGG-D将VGG-C中的1\*1卷积换成了3\*3卷积，该组对比表明3\*3卷积的提升效果要大于1\*1卷积

### 2.6 VGG-D vs VGG-E

当网络层数增加到16层时，网络的精度趋近于饱和。当网络提升到19层时，虽然精度有了些许的提升，但需要的训练时间也大幅增加。

## Reference

\[1\] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition\[J\]. arXiv preprint arXiv:1409.1556, 2014.

\[2\] Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions\[C\]. Cvpr, 2015.

\[3\] Lin, M., Chen, Q., and Yan, S. Network in network. InProc. ICLR, 2014.

