# Spatial Transformer Networks

## 前言

自LeNet-5的结构被提出之后，其“卷积+池化+全连接”的结构被广泛的应用到各处，但是这并不代表其结构没有了优化空间。传统的池化方式（Max Pooling/Average Pooling）所带来卷积网络的位移不变性和旋转不变性只是局部的和固定的。而且池化并不擅长处理其它形式的仿射变换。

Spatial Transformer Network（STN）的提出动机源于对池化的改进，即与其让网络抽象的学习位移不变性和旋转不变性，不如设计一个显示的模块，让网络线性的学习这些不变性，甚至将其范围扩展到所有仿射变换乃至非放射变换。更加通俗的将，STN可以学习一种变换，这种变换可以将进行了仿射变换的目标进行矫正。这也为什么我把STN放在了OCR这一章，因为在OCR场景中，仿射变换是一种最为常见的变化情况。

基于这个动机，作者设计了Spatial Transformer Module（STM），STM具有显示学习仿射变换的能力，并且STM是**可导**的，因此可以直接整合进卷积网络中进行端到端的训练，插入STM的卷积网络叫做STN。

下面根据一份STN的keras源码：[https://github.com/oarriaga/spatial\_transformer\_networks](https://github.com/oarriaga/spatial_transformer_networks)详解STN的算法细节。

## 1. STM详解

STM由三个模块组成：

1. Localisation Network：该模块学习仿射变换矩阵（附件A）；
2. Parameterised Sampling Grid：根据Localisation Network得到仿射变换矩阵，得到输入Feature Map和输出Feature Map之间的位置映射关系；
3. Differentiable Image Sampling：计算输出Feature Map的每个像素点的值。

STM的结构见图1：

###### 图1：STM的框架图

![](/assets/STN_1.png)

### 1.1 Localisation Network

Localisation Network是一个小型的卷积网络$$\theta = f_{loc}(U)$$，其输入是Feature Map （$$U\in R^{W\times H\times C}$$），输出是仿射矩阵$$\theta$$ 的六个值。因此输出层是一个有六个节点回归器。
$$\theta = 
  \left[
  \begin{matrix}
   1 & 2 & 3 \\
   4 & 5 & 6 \\
   7 & 8 & 9
  \end{matrix} 
  \right]
$$
下面的是源码中给出的Localisation Network的结构。


```py
locnet = Sequential()
locnet.add(MaxPooling2D(pool_size=(2,2), input_shape=input_shape))
locnet.add(Conv2D(20, (5, 5)))
locnet.add(MaxPooling2D(pool_size=(2,2)))
locnet.add(Conv2D(20, (5, 5)))

locnet.add(Flatten())
locnet.add(Dense(50))
locnet.add(Activation('relu'))
locnet.add(Dense(6, weights=weights))
```

### 1.2 Parameterised Sampling Grid

 Parameterised Sampling Grid利用Localisation Network产生的$$\theta$$

### 1.3 Differentiable Image Sampling

## Reference

\[1\] Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks\[C\]//Advances in neural information processing systems. 2015: 2017-2025.

## 附件A：仿射变换矩阵

仿射变换(Affline Transformation)是一种二维坐标到二维坐标的线性变化，其保持了二维图形的平直性（straightness）和平行性（parallelness）。

