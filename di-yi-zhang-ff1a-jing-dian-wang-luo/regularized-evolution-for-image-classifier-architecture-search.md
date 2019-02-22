# Regularized Evolution for Image Classifier Architecture Search

## 前言

在我们之前介绍的NAS系列算法中，模型结构的搜索均是通过强化学习实现的。这篇要介绍的AmoebaNet是通过遗传算法的进化策略（Evolution）实现的模型结构的学习过程。该算法的主要特点是在进化过程中引入了年龄的概念，使进化时更倾向于选择更为年轻的性能好的结构，这样确保了进化过程中的多样性和优胜劣汰的特点，这个过程叫做年龄进化（Aging Evolution，AE）。作者为他的网络取名AmoebaNet，Amoeba中文名为变形体，是对形态不固定的生物体的统称，作者也是借这个词来表达AE拥有探索更广的搜索空间的能力。AmoebaNet取得了当时在ImageNet数据集上top-1和top-5的最高精度。

## 1. AmoebaNet算法详解

### 1.1 搜索空间

AmoebaNet使用的是和[NASNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/learning-transferable-architectures-for-scalable-image-recognition.html)\[2\]相同的搜索空间。仿照NASNet的思想，AmoebaNet也是学习两个Cell：\(1\) Normal Cell，\(2\) Reduction Cell，在这里两个Cell是完全独立的。然后通过重复堆叠Normal Cell和Reduction Cell的形式我们可以得到一个完整的网络，如图1左所示。其中Normal Cell中步长始终为1，因此不会改变Feature Map的尺寸，Reduction Cell的步长为2，因此会将Feature Map的尺寸降低为原来的1/2。因此我们可以连续堆叠更多的Normal Cell以获得更大的模型容量（不能堆叠Reduction Cell），如图1左侧图中Normal Cell右侧的$$\times$$N的符号所示。在堆叠Normal Cell时，AmoebaNet使用了shortcut的机制，即一个Normal Cell的输入来自上一层，另外一个输入来自上一层的上一层，如图1中间部分。

![](/assets/AmoebaNet_1.png)

在每个卷积操作中，我们需要学习两个参数：

1. 卷积操作的类型：类型空间参考NASNet。
2. 卷积核的输入：从该Cell中所有可以选择的Feature Map选择两个，每个Feature Map选择一个操作，通过合并这两个Feature Map得到新的Feature Map。最后将所有没有扇出的Feature Map合并作为最终的输出。上面所说的合并是单位加操作，因此Feature Map的个数不会改变。举例说明一下这个过程，根据图1中的跳跃连接，每个Cell有两个输入，对应图1右的0，1。那么第一个操作（红圈部分）选择0，1作为输入以及average池化和max池化作为操作构成新的Feature Map 2。接着第二个操作可以从（0，1，2）中选择两个作为输入，形成Feature Map 3。依次类推可以得到Feature Map 4，5，6，7。

最终AmoebaNet仅仅有两个变量需要决定，一个是每个Feature Map的卷积核数量$$F$$，另一个是堆叠的Normal Cell的个数$$N$$，这两个参数作为了人工设定的超参数，作者也实验了$$N$$和$$F$$的各种组合。

### 1.2 Aging Evolution



## Reference

\[1\] Real E, Aggarwal A, Huang Y, et al. Regularized evolution for image classifier architecture search\[J\]. arXiv preprint arXiv:1802.01548, 2018.

\[2\] Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition\[J\]. arXiv preprint arXiv:1707.07012, 2017, 2\(6\).