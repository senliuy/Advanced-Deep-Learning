# Spatial Transformer Networks

## 前言

自LeNet-5的结构被提出之后，其“卷积+池化+全连接”的结构被广泛的应用到各处，但是这并不代表其结构没有了优化空间。传统的池化方式（Max Pooling/Average Pooling）所带来卷积网络的位移不变性和旋转不变性只是局部的和固定的。而且池化并不擅长处理其它形式的仿射变换。

Spatial Transformer Network（STN）的提出动机源于对池化的改进，即与其让网络抽象的学习位移不变性和旋转不变性，不如设计一个显示的模块，让网络线性的学习这些不变性，甚至将其范围扩展到所有仿射变换乃至非放射变换。更加通俗的将，STN可以学习一种变换，这种变换可以将进行了仿射变换的目标进行矫正。这也为什么我把STN放在了OCR这一章，因为在OCR场景中，仿射变换是一种最为常见的变化情况。

基于这个动机，作者设计了Spatial Transformer Module（STM），STM具有显示学习仿射变换的能力，并且STM是**可导**的，因此可以直接整合进卷积网络中进行端到端的训练，插入STM的卷积网络叫做STN。

下面根据一份STN的keras源码：[https://github.com/oarriaga/spatial\_transformer\_networks](https://github.com/oarriaga/spatial_transformer_networks)详解STN的算法细节。

## 1. STM

STM由三个模块组成：

1. Localisation Network：该模块学习仿射变换矩阵（附件A）；
2. Parameterised Sampling Grid：根据Localisation Network得到仿射变换矩阵，得到输入Feature Map和输出Feature Map之间的位置映射关系；
3. Differentiable Image Sampling：计算输出Feature Map的每个像素点的值。

STM的结构见图1：

###### 图1：STM的框架图

![](/assets/STN_1.png)

### 1.1 Localisation Network

Localisation Network是一个小型的卷积网络$$\Theta = f_{loc}(U)$$，其输入是Feature Map （$$U\in R^{W\times H\times C}$$），输出是仿射矩阵$$\Theta$$ 的六个值。因此输出层是一个有六个节点回归器。


$$
\theta = 
  \left[
  \begin{matrix}
   \theta_{11} & \theta_{12} & \theta_{13} \\
   \theta_{21} & \theta_{22} & \theta_{23}
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

Parameterised Sampling Grid利用Localisation Network产生的$$\Theta$$进行仿射变换，即由输出Feature Map上的某一位置$$G_i = (x^t_i, y^t_i)$$根据变换参数$$\theta$$ 得到输入Feature Map的某一位置$$(x^s_i, y^s_i)$$：


$$
 \left(\begin{matrix}x_i^s \\y_i^s\end{matrix} \right) 
 = \mathcal{T}_\theta(G_i) 
 = \Theta\left(\begin{matrix}x_i^t\\y_i^t\\1\end{matrix}\right)
 = \left[\begin{matrix}\theta_{11} & \theta_{12} & \theta_{13} \\
   \theta_{21} & \theta_{22} & \theta_{23}\end{matrix}\right]
   \left(\begin{matrix}x_i^t\\y_i^t\\1\end{matrix}\right)
$$


图2展示了STM中的一次仿射变换（b）和直接映射的区别。

###### 图2: STM的仿射变换和普通卷积的直接映射

![](/assets/STN_2.png)

这里需要注意两点：  
1. $$\Theta$$可以是一个更通用的矩阵，并不局限于仿射变换，甚至不局限于6个值；  
2. 映射得到的$$(x^s_i, y^s_i)$$一般不是整数，因此不能$$(x^t_i, y^t_i)$$不能使用$$(x^s_i, y^s_i)$$的值，而是根据它进行插值，也就是我们下一节要讲的东西。

### 1.3 Differentiable Image Sampling

如果$$(x^s_i, y^s_i)$$为一整数，那么输出Feature Map的$$(x^t_i, y^t_i)$$处的值便可以从输入Feature Map上直接映射过去。然而在的1.2节我们讲到，$$(s^s_i, y^s_i)$$往往不是整数，这时我们需要进行插值才能确定输出其值，在这个过程叫做一次插值，或者一次采样（Sampling）。插值过程可以用下式表示：


$$
V_{i}^c = \sum^H_n \sum^W_m U^c_{nm} k(x_i^s-m;\Phi_x) k(y_i^s -m; \Phi_y) 
,\quad where\quad \forall i\in[1,...,H'W'],\forall c\in[1,...,C]
$$


在上式中，函数$$f()$$表示插值函数，本文将以双线性插值为例进行解析，$$\Phi$$为$$f()$$中的参数，$$U^c_{nm}$$为输入Feature Map上点$$(n, m, c)$$处的值，$$V_i^c$$便是插值后输出Feature Map的$$(x^t_i, y^t_i)$$处的值。

$$H',W'$$分别为输出Feature Map的高和宽。当$$H'=H$$并且$$W'=W$$时，则STM是正常的仿射变换，当$$H'=H/2$$并且$$W'=W/2$$时, 此时STM可以起到和池化类似的降采样的功能。

以双线性插值为例，插值过程即为：


$$
V_{i}^c = \sum^H_n \sum^W_m U^c_{nm} max(0, 1 - |x_i^s-m|) max(0,1-|y_i^s -m|)
$$


上式可以这么理解：遍历整个输入Feature Map，如果遍历到的点$$(n,m)$$距离大于1，即$$|x_i^s-m|>1$$，那么$$max(0, 1 - |x_i^s-m|)=0$$（n处同理），即只有距离$$(s^s_i, y^s_i)$$最近的四个点参与计算。且距离与权重成反比，也就是距离越小，权值越大，也就是双线性插值的过程。

上式中的几个值都是可偏导的:


$$
\frac{\partial V_i^c}{\partial U_{nm}^c} = \sum^H_n \sum^W_m max(0, 1 - |x_i^s-m|) max(0,1-|y_i^s -m|)
$$



$$
\frac{\partial V_i^c}{\partial x_{i}^s} = \sum^H_n \sum^W_m U^c_{nm} max(0,1-|y_i^s -m|) \left\{
\begin{array}{}
0 & \text{if}\;|m-x_i^s|>1\\
1 & \text{if}\;m\geq x_i^s\\
-1 & \text{if}\;m< x_i^s
\end{array}
\right.
$$



$$
\frac{\partial V_i^c}{\partial y_{i}^s} = \sum^H_n \sum^W_m U^c_{nm} max(0,1-|x_i^s -n|) \left\{
\begin{array}{}
0 & \text{if}\;|n-y_i^s|>1\\
1 & \text{if}\;n\geq y_i^s\\
-1 & \text{if}\;n< y_i^s
\end{array}
\right.
$$


STM的可导带来的好处是其可以和整个卷积网络一起端到端的训练，能够以layer的形式直接插入到卷积网络中。

## 2. STN

1.3节中介绍过，将STM插入到卷积网络中便得到了STN，在插入STM的时候，需要注意以下几点：

1. 在输入图像之后接一个STM是最常见的操作，也是最容易理解的，即自动图像矫正；
2. 理论上讲STM是可以以任意数量插入到网络中的任意位置，但这时无疑增加了网络的深度，其带来的收益价值值得讨论；
3. STM虽然可以起到降采样的作用，但一般不这么使用，因为基于STM的降采样产生了对其的问题；
4. 可以在同一个卷积网络中并行使用多个STM，但是一般STM和图像中的对象是$$1:1$$的关系，因此并不是具有非常广泛的通用性。

## 3. STN的应用场景

3.1 并行STM


## Reference

\[1\] Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks\[C\]//Advances in neural information processing systems. 2015: 2017-2025.

## 附件A：仿射变换矩阵

仿射变换\(Affline Transformation\)是一种二维坐标到二维坐标的线性变化，其保持了二维图形的平直性（straightness）和平行性（parallelness）。

