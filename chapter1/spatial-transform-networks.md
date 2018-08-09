# Spatial Transformer Networks

## 前言

自LeNet-5的结构被提出之后，其“卷积+池化+全连接”的结构被广泛的应用到各处，但是这并不代表其结构没有了优化空间。传统的池化方式（Max Pooling/Average Pooling）所带来卷积网络的位移不变性和旋转不变性只是局部的和固定的。而且池化并不擅长处理其它形式的仿射变换。

Spatial Transformer Network（STN）的提出动机源于对池化的改进，即与其让网络抽象的学习位移不变性和旋转不变性，不如设计一个显示的模块，让网络线性的学习这些不变性，甚至将其范围扩展到所有仿射变换乃至非放射变换。更加通俗的将，STN可以学习一种变换，这种变换可以将进行了仿射变换的目标进行矫正。这也为什么我把STN放在了OCR这一章，因为在OCR场景中，仿射变换是一种最为常见的一种变化要素。

基于这个动机，作者提出了STN，STN具有显示学习仿射变换的能力，并且STN是可导的，因此可以直接整合进卷积网络中进行端到端的训练。

下面根据一份STN的keras源码（https://github.com/oarriaga/spatial\_transformer\_networks）详解STN的算法细节。

## STN详解

## Reference

\[1\] Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks\[C\]//Advances in neural information processing systems. 2015: 2017-2025.

