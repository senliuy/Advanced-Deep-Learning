# Group Normalization

## 前言

Group Normalization（GN）是何恺明提出的一种归一化策略，它是介于Layer Normalization（LN）\[2\]和 Instance Normalization（IN）\[3\]之间的一种折中方案，图1最右。它通过将**通道**数据分成几组计算归一化统计量，因此GN也是和批量大小无关的算法，因此可以用在batchsize比较小的环境中。作者在论文中指出GN要比LN和IN的效果要好。

![](/assets/GN_1.png)

## 1. GN详解

### 1.1 GN算法

和之前所有介绍过的归一化算法相同，GN也是根据该层的输入数据计算均值和方差，然后使用这两个值更新输入数据：


$$
\mu_i = \frac{1}{m}\sum_{k \in \mathcal{S}_i} x_k
\qquad
\sigma_i = \sqrt{\frac{1}{m}\sum_{k \in \mathcal{S}_i}(x_k-\mu_i)^2 + \epsilon} 
\qquad
\hat{x}_i = \frac{1}{\sigma_i} (x_i-\mu_i)
$$


之前所介绍的所有归一化方法均可以使用上面式子进行概括，区别它们的是$$\mathcal{S}_i$$是如何取得的：

对于BN来说，它是取不同batch的同一个channel上的所有的值：


$$
\mathcal{S}_i = \{k | k_C = i_C\}
$$


而LN是从同一个batch的不同的channel上取所有的值：


$$
\mathcal{S}_i = \{k | k_N = i_N\}
$$


IN即不跨batch，也不跨channel：


$$
\mathcal{S}_i = \{k | k_N = i_N，k_C = i_C\}
$$


GN是将Channel分成若干组，只使用组内的数据计算均值和方差。通常组数$$G$$是一个超参数，TensorFlow中的默认值是32。


$$
\mathcal{S}_i = \{k | k_N = i_N, \lfloor \frac{k_C}{C/G}\rfloor = \lfloor \frac{i_C}{C/G}\rfloor\}
$$

我们可以看出，当GN的组数为1时，此时GN和LN等价；当GN的组数为通道数时，GN和IN等价。

GN和其它算法一样也可以添加参数$$\gamma$$和$$\beta$$来保证网络的容量。

### 1.2 GN的伪代码

论文中给出了基于TensorFlow的GN额源码：

```py
1 def GroupNorm(x, gamma, beta, G, eps=1e−5):
2     # x: input features with shape [N,C,H,W]
3     # gamma, beta: scale and offset, with shape [1,C,1,1]
4     # G: number of groups for GN
5     N, C, H, W = x.shape
6     x = tf.reshape(x, [N, G, C // G, H, W])
7     mean, var = tf.nn.moments(x, [2, 3, 4], keep dims=True) 
8     x = (x − mean) / tf.sqrt(var + eps)
9     x = tf.reshape(x, [N, C, H, W]) 
10    return x * gamma + beta
```

第6行代码将Tensor中添加一个’组‘的维度，形成一个五维张量。第7行的`axes`的值为[2,3,4]表明计算归一化统计量时即不会跨batch，也不会跨组。

### 1.2 GN的原理

在深度学习之前，传统的SIFT，HOG等算法均由按组统计特征的特性，它们一般将同一个种类的特征归为一组，然后在进行组归一化。在深度学习中，每个通道的Feature Map也可以看做结构化的特征向量。如果一个Feature Map的卷积数足够多，那么必然有一些通道的特征是类似的，因此我们可以将这些类似的特征进行归一化处理。

## Reference

\[1\] Wu Y, He K. Group normalization\[J\]. arXiv preprint arXiv:1803.08494, 2018.

\[2\] Ba J L, Kiros J R, Hinton G E. Layer normalization\[J\]. arXiv preprint arXiv:1607.06450, 2016.

\[3\] Vedaldi V L D U A. Instance Normalization: The Missing Ingredient for Fast Stylization\[J\]. arXiv preprint arXiv:1607.08022, 2016.

