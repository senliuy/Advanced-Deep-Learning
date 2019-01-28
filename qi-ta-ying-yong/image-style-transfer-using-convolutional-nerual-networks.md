# Image Style Transfer Using Convolutional Nerual Networks

tags: Nueral Style Transfer

## 前言

Leon A.Gatys是最早使用CNN做图像风格迁移的先驱之一，这篇文章还有另外一个版本\[2\]，应该是它投到CVPR之前的预印版，两篇文章内容基本相同。

我们知道在训练CNN分类器时，接近输入层的Feature Map包含更多的图像的纹理等细节信息，而接近输出层的Feature Map则包含更多的内容信息。这个特征的原理可以通过我们在[残差网络](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)中介绍的数据处理不等式（DPI）解释：越接近输入层的Feature Map经过的处理（卷积和池化）越少，则这时候损失的图像信息还不会很多。随着网络层数的加深，图像经过的处理也会增多，根据DPI中每次处理信息会减少的原理，靠后的Feature Map则包含的输入图像的信息是不会多余其之前的Feature Map的；同理当我们使用标签值进行参数更新时，越接近损失层的Feature Map则会包含越多的图像标签（内容）信息，越远则包含越少的内容信息。这篇论文正是利用了CNN的天然特征实现的图像风格迁移的。

具体的讲，当我们要在图片$$\vec{p}$$（content）的内容之上应用图片$$\vec{a}$$（style）的风格时，我们会使用梯度下降等算法更新目标图像$$\vec{x}$$（target）的内容，使其在较浅的层有和图片$$\vec{a}$$类似的响应值，同时在较深的层和$$\vec{p}$$也有类似的响应，这样就保证了$$\vec{x}$$和$$\vec{a}$$有类似的风格而且和$$\vec{p}$$有类似的内容，这样生成的图片$$\vec{x}$$就是我们要得到的风格迁移的图片。如图1所示。

在Keras官方源码中，作者提供了神经风格迁移的[源码](https://github.com/keras-team/keras/blob/fcf2ed7831185a282895dda193217c2a97e1e41d/examples/neural_style_transfer.py)，这里对算法的讲解将结合源码进行分析。

## 1. Image Style Transfer（IST）算法详解

IST的原理基于上面提到的网络的不同层会响应不同的类型特征的特点实现的。给定一个训练好的网络，源码中使用的是[VGG19](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/very-deep-convolutional-networks-for-large-scale-image-recognition.html) \[3\]，下面是源码第142-143行，因此在运行该源码时如果你之前没有下载过训练好的VGG19模型文件，第一次运行会有下载该文件的过程，文件名为'vgg19\_weights\_tf\_dim\_ordering\_tf\_kernels\_notop.h5'。

```py
142 model = vgg19.VGG19(input_tensor=input_tensor,
143                     weights='imagenet', include_top=False)
```

论文中有两点在源码中并没有体现，一个是对权值进行了归一化，使用的方法是我们之前介绍的[Weight Normalization](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-ba-zhang-ff1a-wang-luo-you-hua/weight-normalization.html)\[4\]，另外一个是使用平均池化代替最大池化，使用了这两点的话会有更快的收敛速度。

图2有三个部分，最左侧的输入是风格图片$$\vec{a}$$，将其输入到训练好的VGG19中，会得到一批它对应的Feature Map；最右侧则是内容图片$$\vec{p}$$，它也会输入到这个网络中得到它对应的Feature Map；中间是目标图片$$\vec{x}$$，它的初始值是白噪音图片，它的值会通过SGD进行更新，SGD的损失函数时通过$$\vec{x}$$在这个网络中得到的Feature Map和$$\vec{a}$$的Feature Map以及$$\vec{p}$$的Feature Map计算得到的。图2中所有的细节会在后面的章节中进行介绍。

![](/assets/IST_2.png)

传统的深度学习方法是根据输入数据更新网络的权值。而IST的算法是固定网络的参数，更新输入的数据。固定权值更新数据还有几个经典案例，例如材质学习[5]，卷积核可视化等。

### 1.1 内容表示

内容表示是图2中右侧的两个分支所示的过程。我们先看最右侧，$$\vec{p}$$输入VGG19中，我们提取其在第四个block中第二层的Feature Map，表示为conv4_2（源码中提取的是conv5_2）。假设其层数为$$l$$，$$N_l$$是Feature Map的数量，也就是通道数，$$M_l$$是Feature Map的像素点的个数。那么

## Reference

\[1\] Gatys L A, Ecker A S, Bethge M. Image style transfer using convolutional neural networks\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2414-2423.

\[2\] Gatys L A, Ecker A S, Bethge M. A neural algorithm of artistic style\[J\]. arXiv preprint arXiv:1508.06576, 2015.

\[3\] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition\[J\]. arXiv preprint arXiv:1409.1556, 2014.

\[4\] Salimans T, Kingma D P. Weight normalization: A simple reparameterization to accelerate training of deep neural networks\[C\]//Advances in Neural Information Processing Systems. 2016: 901-909.

[5] Gatys L, Ecker A S, Bethge M. Texture synthesis using convolutional neural networks[C]//Advances in Neural Information Processing Systems. 2015: 262-270.

