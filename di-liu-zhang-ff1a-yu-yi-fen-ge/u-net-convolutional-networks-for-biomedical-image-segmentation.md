# U-Net: Convolutional Networks for Biomedical Image Segmentation

## 前言

U-Net是比较早的使用全卷积网络进行语义分割的算法之一，论文中使用包含压缩路径和扩展路径的对称U形结构在当时非常具有创新性，且一定程度上影响了后面若干个分割网络的设计，该网络的名字也是取自其U形形状。

U-Net的实验是一个比较简单的ISBI cell tracking数据集，由于本身的任务比较简单，U-Net紧紧通过30张图片并辅以数据扩充策略便达到非常低的错误率，拿了当届比赛的冠军。

论文源码已开源，可惜是基于MATLAB的Caffe版本。虽然已有各种开源工具的实现版本的U-Net算法陆续开源，但是它们绝大多数都刻意回避了U-Net论文中的细节，虽然这些细节现在看起来已无关紧要甚至已被淘汰，但是为了充分理解这个算法，笔者还是建议去阅读作者的源码，地址如下：https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/

## 1. 算法详解

### 1.1 U-Net的网络结构

直入主题，U-Net的U形结构如图1：

<figure>
    <img src="/assets/U-Net_1.png" alt="Tree frog" />
    <figcaption>U-Net网络结构图</figcaption>
</figure>

## Reference

\[1\] Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation\[C\]//International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015: 234-241.