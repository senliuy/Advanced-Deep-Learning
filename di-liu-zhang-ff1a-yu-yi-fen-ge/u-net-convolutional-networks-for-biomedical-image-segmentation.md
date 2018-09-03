# U-Net: Convolutional Networks for Biomedical Image Segmentation

tags: U-Net, Semantic Segmentation 

## 前言

U-Net是比较早的使用全卷积网络进行语义分割的算法之一，论文中使用包含压缩路径和扩展路径的对称U形结构在当时非常具有创新性，且一定程度上影响了后面若干个分割网络的设计，该网络的名字也是取自其U形形状。

U-Net的实验是一个比较简单的ISBI cell tracking数据集，由于本身的任务比较简单，U-Net紧紧通过30张图片并辅以数据扩充策略便达到非常低的错误率，拿了当届比赛的冠军。

论文源码已开源，可惜是基于MATLAB的Caffe版本。虽然已有各种开源工具的实现版本的U-Net算法陆续开源，但是它们绝大多数都刻意回避了U-Net论文中的细节，虽然这些细节现在看起来已无关紧要甚至已被淘汰，但是为了充分理解这个算法，笔者还是建议去阅读作者的源码，地址如下：https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

## 1. 算法详解

### 1.1 U-Net的网络结构

直入主题，U-Net的U形结构如图1所示。网络是一个经典的全卷积网络（即网络中没有全连接操作）。网络的输入是一张$$572\times572$$的边缘经过镜像操作的图片（input image tile），关于“镜像操作“会在1.2节进行详细分析，网络的左侧（红色虚线）是由卷积和Max Pooling构成的一系列降采样操作，论文中将这一部分叫做压缩路径（contracting path）。压缩路径由4个block组成，每个block使用了3个有效卷积和1个Max Pooling降采样，每次降采样之后Feature Map的个数乘2，因此有了图中所示的Feature Map尺寸变化。最终得到了尺寸为$$32\times32$$的Feature Map。

压缩路径后接了一个由3个$$3\times3$$有效卷积构成的block（蓝色虚线），用于向下一段要介绍的扩展路径过度，姑且将其叫做过度路径(Transition Path)，经过扩展路径之后，Feature Map的尺寸是$$28\times28$$

网络的右侧部分(绿色虚线)在论文中叫做扩展路径（expanding path）。同样由4个block组成，每个block开始之前通过反卷积将Feature Map的尺寸乘2，同时将其个数减半（最后一层略有不同），卷积操作依旧使用的是有效卷积操作，最终得到的Feature Map的尺寸是$$388\times388$$。由于该任务是一个二分类任务，所以网络有两个输出Feature Map。


如图1中所示，网络的输入图片的尺寸是$$572\times572$$，而输出Feature Map的尺寸是$$388\times388$$，这两个图像的大小是不同的，无法直接计算损失函数，那么U-Net是怎么操作的呢？

### 1.2 U-Net究竟输入了什么

论文中说的是U-Net的输入经过了镜像操作，如图2所示。但是镜像操作的具体细节并没有将，通过源码，我们可以得到该方法的详细细节。

镜像操作源码位于`mycaffe_tiled_forward5.m`文件中，核心代码如下：



```
border = round(input_size-output_size)/2;
...
if( strcmp(opts.padding,'mirror'))
  xpad  = border(1);
  xfrom = border(1)+1;
  xto   = border(1)+size(data,1);
  paddedFullVolume(1:xfrom-1,:,:) = paddedFullVolume( xfrom+xpad:-1:xfrom+1,:,:);
  paddedFullVolume(xto+1:end,:,:) = paddedFullVolume( xto-1:-1:xto-xpad,    :,:);
  
  ypad  = border(2);
  yfrom = border(2)+1;
  yto   = border(2)+size(data,2);
  paddedFullVolume(:, 1:yfrom-1,:) = paddedFullVolume( :, yfrom+ypad:-1:yfrom+1,:);
  padd
```


## Reference

\[1\] Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation\[C\]//International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015: 234-241.