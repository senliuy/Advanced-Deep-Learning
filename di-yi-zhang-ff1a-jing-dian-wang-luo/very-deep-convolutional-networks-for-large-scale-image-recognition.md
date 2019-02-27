# Very Deep Convolutional NetWorks for Large-Scale Image Recognition

## 1. 前言

时间来到2014年，随着AlexNet在ImageNet数据集上的大放异彩，探寻针对ImageNet的数据集的最优网络成为了提升该数据集精度的一个最先想到的思路。牛津大学计算机视觉组（Visual Geometry Group）和Google的Deep Mind的这篇论文便是对卷积网络的深度和其性能的探索，由此该网络也被命名为VGG{{"simonyan2014very"|cite}}。

VGG的结构非常清晰：

* 按照2\*2的Pooling层，网络可以分成若干段；
* 每段之内由若干个same卷机操作构成，段之内的Feature Map数量固定不变；
* Feature Map按段以2倍的速度逐渐递增，第四段和第五段都是512（64-128-256-512-512）。

VGG的结构非常容易扩展到其它数据集。在VGG中，段数每增加1，Feature Map的尺寸减少一半，所以通过减少段的数目将网络应用到例如MNIST，CIFAR等图像尺寸更小的数据集。段内的卷积的数量是可变的，因为卷积的个数并不会影响图片的尺寸，我们可以根据任务的复杂度自行调整段内的卷积数量。

VGG的表现效果也非常好，在ILSVRC2014分类中排名第二（第一是GoogLeNet {{"szegedy2015going"|cite}}，没有办法），定位比赛排名第一。

VGG将其模型开源在其官方网站上，为其它任务提供了非常好的迁移学习的材料，使得VGG马上占有了大量商业市场。关于不同框架的VGG模型，自行在晚上搜索。

## 2. VGG介绍

VGG关于网络结构的探索可以总结为图1，图1包含了大量信息，我们一一分析之

###### 图1：VGG家族

![](/assets/VGG_1.png)

图2反应了VGG家族的各个模型的性能

图2：VGG家族的性能表现

![](/assets/VGG_2.png)

关于VGG家族的keras实现和参数统计见附件A。

### 2.1 家族的特征

我们来看看VGG家族的共同特征

* 输入图像的尺寸均是$$224\times224$$；
* 均为5层Max Pooling，表示最终均会产生大小为$$7\times7$$的Feature Map，这是一个大小比较合适的尺寸；
* 卷积部分之后（特征层）跟的是两个隐层节点数目为4096的全连接，最后接一个1000类softmax分类器。
* 所有VGG模型均可以表示为:$$m\times(n\times(conv_{33})+max\_pooling)$$

VGG在卷积核方向的最大改进是将卷积核全部换成更小的$$3\times3$$或者$$1\times1$$的卷积核，而性能最好的VGG-16和VGG-19由且仅由$$3\times3$$卷积构成。原因有如下两点：

1. 根据感受野的计算公式$$rfsize = (out-1) \times stride + ksize$$，我们知道一个$$7\times7$$的卷积核和3层$$3\times3$$的卷积核具有相同的感受野，但是由于3层感受野具有更深的深度，由此可以构建更具判别性的决策函数；
2. 假设Feature Map的数量都是C，3层$$3\times3$$卷积核的参数个数是$$3\times(3\times3+1)\times C^2 = 30C^2$$，1层$$7\times7$$卷积核的参数个数是$$1\times(7\times7+1)\times C^2=50C^2$$, 3层$$3\times3$$卷积核具有更少的参数。
3. 但由于神经元数量和层数的增多，训练速度会变得更慢

下图是把LeNet-5的$$5\times5$$卷积换成了两层$$3\times3$$卷积在MNIST上的收敛表现，实验表明两层$$3\times3$$的网络确实比单层$$5\times5$$的网络表现好，但是训练速度也慢了一倍。

###### 图3：$$3\times3$$ LeNet vs $$5\times5$$ LeNet

![](/assets/VGG_3.png)

另外，作者在前两层的全连接出使用drop rate = 0.5的Dropout，然而并没有在图1中反应出来。

### 2.2 VGG-A vs VGG-A-LRN

VGG A-LRN 比 VGG A多了一个AlexNet介绍的LRN层，但是实验数据表明加入加入LRN的VGG-A错误率反而更高了。而且LRN的加入会更加占用内存消耗以及增加训练时间。

### 2.3 VGG-A vs VGG-B vs VGG-D vs VGG-E

对比VGG-A\(11层\), VGG-B\(13层\), VGG-D\(16层\), VGG-E\(19层\)的错误率，我们发现随着网络深度的增加，分类的错误率逐渐降低，当然越深的深度则表示需要的训练时间越长。但是当模型的深度到达一定深度时（VGG-D和VGG-E），网络的错误率趋近饱和，甚至偶尔会发生深层网络的错误率高于浅层网络的情况，同时考虑网络的训练时间，我们就要折中考虑选择合适的网络深度了。我相信作者一定探索了比VGG-E更深的网络，但是由于表现的不理想并没有列在论文中。后面介绍的残差网络则通过shortcut的机制将网络的深度理论上扩展到了无限大。

### 2.4 VGG-B vs VGG-C

VGG-C在VGG-B的基础上添加了3个$$1\times1$$的卷积层，$$1\times1$$的卷积是在NIN{{"lin2013network"|cite}}中率先使用的，由于$$1\times1$$卷积在不影响感受野的前提提升了决策函数的非线性性，由此带来了错误率的下降。

### 2.5 VGG-C vs VGG-D

VGG-D将VGG-C中的$$1\times1$$卷积换成了$$3\times3$$ 卷积，该组对比表明$$3\times3$$ 卷积的提升效果要大于$$1\times1$$卷积

### 2.6 VGG-D vs VGG-E

当网络层数增加到16层时，网络的精度趋近于饱和。当网络提升到19层时，虽然精度有了些许的提升，但需要的训练时间也大幅增加。

## 3. VGG的训练和测试

### 3.1 训练

VGG的训练分为单尺度训练（single-scale training）和多尺度训练（multi-scale training）。在单尺度训练中，原图的短边被固定为一个固定值S（实验中S被固定为了256和384），然后等比例缩放图片。再从缩放的图片中裁剪$$224\times224$$的子图用于训练模型。在多尺度训练中，每张图的短边随机为256到512之间的一个随机值，然后再从缩放的图片中裁剪$$224\times224$$的子图。

### 3.2 测试

测试时可以使用和训练相同的图片裁剪方法，然后通过若干不同裁剪的图片的投票的方式选择最后的分类。

但测试的时候图片是单张输入的，使用裁剪的方式可能会漏掉图片的重要信息，在OverFeat {{"sermanet2013overfeat"|cite}}的论文中，提出了将整幅图做为输入的方式，过程如下：

1. 将测试图片的短边固定为Q，Q可以不等于S；
2. 将Q输入VGG，在conv5层，得到$$W\times H\times512$$的特征向量，W和H一般不等于7；
3. 将第一层全连接层看成$$7\times7\times512\times4096$$ 的卷积层（原本需要先进行flattern操作，再进行FC操作）,对比附件中的vgg-e和使用全卷积的vgg-e-test，可以发现两者具有相同的参数数量。
4. 将第二、三全连接层看成$$1\times1\times4096\times4096$$与$$1\times1\times4096\times numClasses$$的卷积层
5. 如果输入图片大小为 $$224\times224$$，则输出为$$1\times1\times numClasses$$，因为图片大小可以不一致，可以看作某张图片多个切片[^1]的预测结果。最终经过sum-pool，每个通道求和，得到$$1\times1\times numClasses$$的结果。作为最终输出，即取所有平均数作为最终输出。


## 附件A

VGG模型的keras代码和参数统计：[https://github.com/senliuy/CNN-Structures/blob/master/VGG.ipynb](https://github.com/senliuy/CNN-Structures/blob/master/VGG.ipynb)

