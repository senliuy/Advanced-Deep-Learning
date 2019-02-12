# Instance Normalization

tags: Normalization

## 前言

对于我们之前介绍过的[图像风格迁移]()\[2\]这类的注重每个像素的任务来说，每个样本的每个像素点的信息都是非常重要的，于是像[BN]()\[3\]这种每个批量的所有样本都做归一化的算法就不太适用了，因为BN计算归一化统计量时考虑了一个批量中所有图片的内容，从而造成了每个样本独特细节的丢失。同理对于[LN]()\[4\]这类需要考虑一个样本所有通道的算法来说可能忽略了不同通道的差异，也不太适用于图像风格迁移这类应用。

所以这篇文章提出了Instance Normalization（IN），一种更适合对单个像素有更高要求的场景的归一化算法（IST，GAN等）。IN的算法非常简单，计算归一化统计量时考虑单个样本，单个通道的所有元素。IN（右）和BN（中）以及LN（左）的不同从图1中可以非常明显的看出。

![](/assets/IN_1.png)

## 1.IN详解

## 1.1 IST中的IN

在Gatys等人的IST算法中，他们提出的策略是通过L-BFGS算法优化生成图片，风格图片以及内容图片再VGG-19上生成的Feature Map的均方误差。这种策略由于Feature Map的像素点数量过于多导致了优化起来非常消耗时间以及内存。IN的作者Ulyanov等人同在2016年提出了Texture network\[5\]（图2），

![](/assets/IN_2.png)

图2中的生成器网络（Generator Network）是一个由卷积操作构成的全卷积网络，在原始的Texture Network中，生成器使用的操作包括卷积，池化，上采样以及**BN**。但是作者发现当训练生成器网络网络时，使用的样本数越少（例如16个），得到的效果越好。但是我们知道BN并不适用于样本数非常少的环境中，因此作者提出了IN，一种不受限于批量大小的算法专门用于Texture Network中的生成器网络。

## 1.2 IN vs BN



## Reference

\[1\] Vedaldi V L D U A. Instance Normalization: The Missing Ingredient for Fast Stylization\[J\]. arXiv preprint arXiv:1607.08022, 2016.

\[2\] Gatys L A, Ecker A S, Bethge M. Image style transfer using convolutional neural networks\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2414-2423.

\[3\] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift\[J\]. arXiv preprint arXiv:1502.03167, 2015.

\[4\] Ba J L, Kiros J R, Hinton G E. Layer normalization\[J\]. arXiv preprint arXiv:1607.06450, 2016.

\[5\] Ulyanov, D., Lebedev, V., Vedaldi, A., and Lempitsky, V. S. \(2016\). Texture networks: Feed-forward synthesis of textures and stylized images. In Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, pages 1349–1357.

