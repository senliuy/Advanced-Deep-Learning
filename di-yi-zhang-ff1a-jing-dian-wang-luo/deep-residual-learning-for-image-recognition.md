# Deep Residual Learning for Image Recognition

## 前言

在VGG中，卷积网络达到了19层，在GoogLeNet中，网络史无前例的达到了22层。那么，网络的精度会随着网络的层数增多而增多吗？在深度学习中，网络层数增多一般会伴着下面几个问题

1. 计算资源的消耗
2. 模型容易过拟合
3. 梯度消失/梯度爆炸问题的产生

问题1可以通过GPU集群来解决，对于一个企业资源并不是很大的问题；问题2的过拟合通过采集海量数据，并配合Dropout正则化等方法也可以有效避免；问题3通过Batch Normalization也可以避免。貌似我们只要无脑的增加网络的层数，我们就能从此获益，但实验数据给了我们当头一棒。

作者发现，随着网络层数的增加，网络发生了退化（degradation）的现象：随着网络层数的增多，训练集loss逐渐下降，然后趋于饱和，当你再增加网络深度的话，训练集loss反而会增大。注意这并不是过拟合，因为在过拟合中训练loss是一直减小的。

当网络退化时，浅层网络能够达到比深层网络更好的训练效果，这时如果我们把低层的特征传到高层，那么效果应该至少不比浅层的网络效果差，或者说如果一个VGG-100网络在第98层使用的是和VGG-16第14层一模一样的特征，那么VGG-100的效果应该会和VGG-16的效果相同。所以，我们可以在VGG-100的98层和14层之间添加一条直接映射（Identity Mapping）来达到此效果。

基于这种使用直接映射来连接网络不同层直接的思想，残差网络应运而生。

## 1. 残差网络

### 1.1 残差块

残差网络是由一系列残差块组成的（图1）。一个残差块可以用表示为：

```
x_{l+1}= x_l+\mathcal{F}(x_l, {W_l})
```

残差块分成两部分直接映射部分和残差部分。h\(x\_l\)是直接映射，反应在图1中是左边的曲线；\mathcal{F}\(x\_l, {W\_l}\)是残差部分，一般由两个或者三个卷积操作构成，即图1中右侧包含卷积的部分。

###### 图1：残差块

\[ResNet\_1\]

图1中的'Weight‘在卷积网络中是指卷积操作，’addition‘是指单位加操作。

在卷积网络中，x\_l可能和x\_{l+1}的Feature Map的数量不一样，这时候就需要使用1\*1卷积进行升维或者降维（图2）。这时，残差块表示为：

```
x_{l+1}= h(x_l)+\mathcal{F}(x_l, {W_l})
```

其中h\(x\_l\) = W'\_lx。实验结果1\*1卷积对模型性能提升有限，所以一般是在升维或者降维时才会使用。

###### 图2：1\*1残差块

\[ResNet\_2\]

### 1.2 残差网络

残差网络的搭建分为两步：

1. 使用VGG公式搭建Plain VGG网络
2. 在Plain VGG的卷积网络之间插入Identity Mapping，注意需要升维或者降维的时候加入1\*1卷积。

在实现过程中，一般是直接stack残差块的方式。

## 2. 残差网络的背后原理

残差块一个更通用的表示方式是

```
y_l= h(x_l)+\mathcal{F}(x_l, {W_l})
```

```
x_{l+1} = f(y_l)
```

现在我们先不考虑升维或者降维的情况，那么在\[1\]中，h\(\cdot\)是直接映射，f\(\cdot\)是激活函数，一般使用ReLU。我们首先给出两个假设：

* 假设1：h\(\cdot\)是直接映射；
* 假设2：f\(\cdot\)是直接映射。

那么这时候残差块可以表示为：

```
x_{l+1} = x_l + \mathcal{F}(x_l, {W_l})
```

对于一个更深的层L，其与l层的关系可以表示为

```
x_L = x_l + \sum_{i=1}^{L-1}\mathcal{F}(x_i, {W_i})
```

这个公式反应了残差网络的两个属性：

1. L层可以表示为任意一个比它浅的l层和他们之间的残差部分之和；
2. x\_L= x\_0 + \sum\_{i=0}^{L-1}\mathcal{F}\(x\_i, {W\_i}\)，L是各个残差块特征的单位累和，而MLP是特征矩阵的累积。

根据BP中使用的导数的链式法则，损失函数\varepsilon关于x\_l的梯度可以表示为

```
\frac{\partial \varepsilon}{\partial x_l} = \frac{\partial \varepsilon}{\partial x_L}\frac{\partial x_L}{\partial x_l} = \frac{\partial \varepsilon}{\partial x_L}(1+\frac{\partial }{\partial x_l}\sum_{i=1}^{L-1}\mathcal{F}(x_i, {W_i})) = \frac{\partial \varepsilon}{\partial x_L}+\frac{\partial \varepsilon}{\partial x_L} \frac{\partial }{\partial x_l}\sum_{i=1}^{L-1}\mathcal{F}(x_i, {W_i})
```

上面公式反映了残差网络的两个属性：

1. 在整个训练过程中，\frac{\partial }{\partial x\_l}\sum\_{i=1}^{L-1}\mathcal{F}\(x\_i, {W\_i}\) 不可能一直为-1，也就是说在残差网络中不会出现梯度消失的问题。
2. \frac{\partial \varepsilon}{\partial x\_L}表示L层的梯度可以直接传递到任何一个比它浅的l层。

通过分析残差网络的正向和反向两个过程，我们发现，当残差块满足上面两个假设时，信息可以非常畅通的在高层和低层之间相互传导，说明这两个假设是让残差网络可以训练深度模型的充分条件。那么这两个假设是必要条件吗？

### 2.1 直接映射是最好的选择

对于假设1，我们采用反证法，假设h\(x\_l\) = \lambda\_l x\_l，那么这时候，残差块（图3.b）表示为

```
x_{l+1} = \lambda_lx_l + \mathcal{F}(x_l, {W_l})
```

对于更深的L层

```
x_{L} = (\prod_{i=l}^{L-1}\lambda_l)x_l + \sum_{i=l}^{L-1}((\prod_{i=l}^{L-1})\mathcal{F}(x_l, {W_l})
```

为了简化问题，我们只考虑公式的左半部分x'\_{L} = \(\prod\_{i=l}^{L-1}\lambda\_l\)x\_l，\varepsilon对x\_l求偏微分得

```
\frac{\partial\varepsilon}{\partial x_l} = \frac{\partial\varepsilon}{\partial x_L} (\prod_{i=l}^{L-1}\lambda_i)
```

上面公式反映了两个属性：

1. 当\lambda&gt;1时，很有可能发生梯度爆炸；
2. 当\lambda&lt;1时，梯度变成0，会阻碍残差网络信息的反向传递，从而影响残差网络的训练。

所以\lambda必须等1。同理，其他常见的激活函数都会产生和上面的例子类似的阻碍信息反向传播的问题。

对于其它不影响梯度的h\(\cdot\)，例如LSTM中的门机制（图3.c，图3.d）或者Dropout（图3.f）以及\[1\]中用于降维的1\*1卷积（图3.e）也许会有效果，作者采用了实验的方法进行验证，实验结果见图4

###### 图3：直接映射的变异模型

###### 图4：变异模型在Cifar10数据集上的表现



## Reference

\[1\] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

\[2\] He K, Zhang X, Ren S, et al. Identity mappings in deep residual networks\[C\]//European Conference on Computer Vision. Springer, Cham, 2016: 630-645.

