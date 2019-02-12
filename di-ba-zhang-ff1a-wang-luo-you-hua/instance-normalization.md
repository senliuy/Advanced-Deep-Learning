# Instance Normalization

tags: Normalization

## 前言

对于我们之前介绍过的[图像风格迁移]()[2]这类的注重每个像素的任务来说，每个样本的每个像素点的信息都是非常重要的，于是像[BN]()[3]这种每个批量的所有样本都做归一化的算法就不太适用了，因为BN计算归一化统计量时考虑了一个批量中所有图片的内容，从而造成了每个样本独特细节的丢失。同理对于LN这类需要考虑一个样本所有通道的算法来说可能忽略了不同通道的差异，也不太适用于图像风格迁移这类应用。

所以这篇文章提出了Instance Normalization（IN），一种更适合对单个像素有更高要求的场景的归一化算法（IST，GAN等）。IN的算法非常简单，计算归一化统计量时考虑单个样本，单个通道的所有元素。它和BN以及LN的不同从图1中可以非常明显的看出。



## Reference

[1] Vedaldi V L D U A. Instance Normalization: The Missing Ingredient for Fast Stylization[J]. arXiv preprint arXiv:1607.08022, 2016.

[2] Gatys L A, Ecker A S, Bethge M. Image style transfer using convolutional neural networks\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2414-2423.

[3] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift\[J\]. arXiv preprint arXiv:1502.03167, 2015.



