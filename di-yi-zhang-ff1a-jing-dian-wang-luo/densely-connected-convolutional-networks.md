# Densely Connected Convolutional Networks

## 前言

在残差网络的文章中，我们知道残差 \[2, 3\]网格能够应用在特别深的网络中的一个重要原因是，无论正向计算精度还是反向计算梯度，信息都能毫无损失的从一层传到另一层。如果我们的目的是保证信息毫无阻碍的传播，那么残差网络的stacking残差块的设计便不是信息流通最合适的结构。

基于信息流通的原理，一个最简单的思想便是在网络中的每个卷积操作中，将其低层的所有特征作为该网络的输入，也就是在一个层数为L的网络中加入\frac{L\(L+1\)}{2}个short-cut, 如图1。为了更好的保存低层网络的特征，DenseNet使用的是将不同层的输出拼接在一起，而在残差网络中使用的是单位加操作。以上便是DenseNet算法的动机。

###### 图1：DenseNet中一个Dense Block的设计

\[DenseNet\_1.png\]

## 1. DenseNet算法解析及源码实现

在DenseNet中，如果全部采用图1的结构的话，第L层的输入是之前所有的Feature Map拼接到一起。考虑到现今内存/显存空间的问题，该方法显然是无法应用到网络比较深的模型中的，故而DenseNet采用了图2所示的堆积Dense Block的形式，下面我们针对图2详细解析DenseNet算法。

###### 图2：DenseNet网络结构

\[DenseNet\_2.png\]

### 1.1 Dense Block

图1便是一个Dense Block，在Dense Block中，第l层的输入x\_l是这个块中前面所有层的输出:

```
x_l = [y_0, y_1, ..., y_{l-1}]
```

```
y_l = H_l(x_l)
```

其中，中括号\[y\_0, y\_1, ..., y\_{l-1}\]表示拼接操作，即按照Feature Map将l-1个输入拼接成一个Tensor。H\_l\(\cdot\)表示合成函数（Composite function）。在实现时，我使用了stored\_features存储每个合成函数的输出。

```py
def dense_block(x, depth=5, nb_filters = 2, growth_rate = 3):
    stored_features = x
    for i in range(depth):
        feature = composite_function(stored_features, nb_filters = growth_rate * i + nb_filters)
        if i > 0:
            stored_features = concatenate([stored_features, feature], axis=3)
        else:
            store_features = feature
    return stored_features
```

1.2 合成函数（Composite function）



## Reference

\[1\] Huang G, Liu Z, Weinberger K Q, et al. Densely connected convolutional networks\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017, 1\(2\): 3.

\[2\] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

\[3\] He K, Zhang X, Ren S, et al. Identity mappings in deep residual networks\[C\]//European Conference on Computer Vision. Springer, Cham, 2016: 630-645.

