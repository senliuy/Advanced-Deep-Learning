# U-Net: Convolutional Networks for Biomedical Image Segmentation

tags: U-Net, Semantic Segmentation

## 前言

U-Net是比较早的使用全卷积网络进行语义分割的算法之一，论文中使用包含压缩路径和扩展路径的对称U形结构在当时非常具有创新性，且一定程度上影响了后面若干个分割网络的设计，该网络的名字也是取自其U形形状。

U-Net的实验是一个比较简单的ISBI cell tracking数据集，由于本身的任务比较简单，U-Net紧紧通过30张图片并辅以数据扩充策略便达到非常低的错误率，拿了当届比赛的冠军。

论文源码已开源，可惜是基于MATLAB的Caffe版本。虽然已有各种开源工具的实现版本的U-Net算法陆续开源，但是它们绝大多数都刻意回避了U-Net论文中的细节，虽然这些细节现在看起来已无关紧要甚至已被淘汰，但是为了充分理解这个算法，笔者还是建议去阅读作者的源码，地址如下：[https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/)

## 1. 算法详解

### 1.1 U-Net的网络结构

直入主题，U-Net的U形结构如图1所示。网络是一个经典的全卷积网络（即网络中没有全连接操作）。网络的输入是一张$$572\times572$$的边缘经过镜像操作的图片（input image tile），关于“镜像操作“会在1.2节进行详细分析，网络的左侧（红色虚线）是由卷积和Max Pooling构成的一系列降采样操作，论文中将这一部分叫做压缩路径（contracting path）。压缩路径由4个block组成，每个block使用了3个有效卷积和1个Max Pooling降采样，每次降采样之后Feature Map的个数乘2，因此有了图中所示的Feature Map尺寸变化。最终得到了尺寸为$$32\times32$$的Feature Map。

网络的右侧部分\(绿色虚线\)在论文中叫做扩展路径（expansive path）。同样由4个block组成，每个block开始之前通过反卷积将Feature Map的尺寸乘2，同时将其个数减半（最后一层略有不同），卷积操作依旧使用的是有效卷积操作，最终得到的Feature Map的尺寸是$$388\times388$$。由于该任务是一个二分类任务，所以网络有两个输出Feature Map。

<figure>
<img src="/assets/U-Net_1.png" alt="图1：U-Net网络结构图" />
<figcaption>图1：U-Net网络结构图</figcaption>
</figure>

如图1中所示，网络的输入图片的尺寸是$$572\times572$$，而输出Feature Map的尺寸是$$388\times388$$，这两个图像的大小是不同的，无法直接计算损失函数，那么U-Net是怎么操作的呢？

### 1.2 U-Net究竟输入了什么

首先，数据集我们的原始图像的尺寸都是$$512\times512$$的。为了能更好的处理图像的边界像素，U-Net使用了镜像操作（Overlay-tile Strategy）来解决该问题。镜像操作即是给输入图像加入一个对称的边（图2），那么边的宽度是多少呢？一个比较好的策略是通过感受野确定。因为有效卷积是会降低Feature Map分辨率的，但是我们希望$$512\times512$$的图像的边界点能够保留到最后一层Feature Map。所以我们需要通过加边的操作增加图像的分辨率，增加的尺寸即是感受野的大小，也就是说每条边界增加感受野的一半作为镜像边。

<figure>
<img src="/assets/U-Net_2.png" alt="图1：U-Net镜像操作" />
<figcaption>图2：U-Net镜像操作</figcaption>
</figure>

根据图1中所示的压缩路径的网络架构，我们可以计算其感受野：

$$
rf = (((0 \times2 +2 +2)\times2 +2 +2)\times2 +2 +2)\times2 +2 +2 = 60
$$

图1中左侧网络中的虚线部分显示的是未加边的图片在网络中的形状变化示意图，在$$32\times32$$的Feature Map中，镜像策略加的边已全部被有效卷积所抵消。下面代码片段是镜像操作的核心部分



```
d4a_size= 0;
opts.padInput =   (((d4a_size *2 +2 +2)*2 +2 +2)*2 +2 +2)*2 +2 +2; % 60

```



## Reference

\[1\] Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation\[C\]//International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015: 234-241.

