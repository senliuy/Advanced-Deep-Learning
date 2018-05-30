# Very Deep Convolutional NetWorks for Large-Scale Image Recognition

时间来到2014年，随着AlexNet在ImageNet数据集上的大放异彩，探寻针对ImageNet的数据集的最优网络成为了提升该数据集精度的一个最先想到的思路。牛津大学计算机视觉组（Visual Geometry Group）和Google的Deep Mind的这篇论文便是对卷积网络的深度和其性能的探索。

VGG的结构非常清晰：

* 按照Pooling层，网络可以分成若干段；
* 每段之内由若干个3\*3的same卷机操作构成，段之内的Feature Map数量固定不变；
* 第i段的Feature Map数量是第i-1段的Feature Map数量的2倍

VGG的结构非常容易扩展到其它数据集。在VGG中，段数每增加1，Feature Map的尺寸减少一半，所以通过减少段的数目将网络应用到例如MNIST，CIFAR等图像尺寸更小的数据集。段内的卷积的数量是可变的，因为卷积的个数并不会影响图片的尺寸，我们可以根据任务的复杂度自行调整段内的卷积数量。

VGG的表现效果也非常好，在ILSVRC2014 \[1\]分类中排名第二（第一是GoogLeNet \[2\]，没有办法），定位比赛排名第一。

Reference

\[1\] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition\[J\]. arXiv preprint arXiv:1409.1556, 2014.

\[2\] Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions\[C\]. Cvpr, 2015.



