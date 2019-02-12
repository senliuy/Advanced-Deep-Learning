# Instance Normalization

tags: Normalization

## 前言

对于我们之前介绍过的[图像风格迁移]()\[2\]这类的注重每个像素的任务来说，每个样本的每个像素点的信息都是非常重要的，于是像[BN]()\[3\]这种每个批量的所有样本都做归一化的算法就不太适用了，因为BN计算归一化统计量时考虑了一个批量中所有图片的内容，从而造成了每个样本独特细节的丢失。同理对于[LN]()\[4\]这类需要考虑一个样本所有通道的算法来说可能忽略了不同通道的差异，也不太适用于图像风格迁移这类应用。

所以这篇文章提出了Instance Normalization（IN），一种更适合对单个像素有更高要求的场景的归一化算法（IST，GAN等）。IN的算法非常简单，计算归一化统计量时考虑单个样本，单个通道的所有元素。IN（右）和BN（中）以及LN（左）的不同从图1中可以非常明显的看出。

<figure>
<img src="/assets/IN_1.png" alt="图1：LN（左），BN（中），IN（右）" />
<figcaption>图1：LN（左），BN（中），IN（右）</figcaption>
</figure>


## 1.IN详解

## 1.1 IST中的IN

在Gatys等人的IST算法中，他们提出的策略是通过L-BFGS算法优化生成图片，风格图片以及内容图片再VGG-19上生成的Feature Map的均方误差。这种策略由于Feature Map的像素点数量过于多导致了优化起来非常消耗时间以及内存。IN的作者Ulyanov等人同在2016年提出了Texture network\[5\]（图2），

<figure>
<img src="/assets/IN_2.png" alt="图2：Texture Networks的网络结构" />
<figcaption>图2：Texture Networks的网络结</figcaption>
</figure>

图2中的生成器网络（Generator Network）是一个由卷积操作构成的全卷积网络，在原始的Texture Network中，生成器使用的操作包括卷积，池化，上采样以及**BN**。但是作者发现当训练生成器网络网络时，使用的样本数越少（例如16个），得到的效果越好。但是我们知道BN并不适用于样本数非常少的环境中，因此作者提出了IN，一种不受限于批量大小的算法专门用于Texture Network中的生成器网络。

## 1.2 IN vs BN

BN的详细算法我们已经分析过，这里再重复一下它的计算方式：


$$
\mu_i = \frac{1}{HWT}\sum_{t=1}^T\sum_{l=1}^W\sum_{m=1}^H x_{tilm}
\qquad
\sigma_i^2 = \frac{1}{HWT}\sum_{t=1}^T\sum_{l=1}^W\sum_{m=1}^H (x_{tilm} -\mu_i)^2
\qquad
y_{tijk} = \frac{x_{tijk}-\mu_{i}}{\sqrt{\sigma_{i}^2+ \epsilon}}
$$


正如我们之前所分析的，IN在计算归一化统计量时并没有像BN那样跨样本、单通道，也没有像LN那样单样本、跨通道。它是取的单通道，单样本上的数据进行计算，如图1最右侧所示。所以对比BN的公式，它只需要它只需要去掉批量维的求和即可：


$$
\mu_{ti} = \frac{1}{HW}\sum_{l=1}^W\sum_{m=1}^H x_{tilm}
\qquad
\sigma_{ti}^2 = \frac{1}{HW}\sum_{l=1}^W\sum_{m=1}^H (x_{tilm} -\mu_{ti})^2
\qquad
y_{tijk} = \frac{x_{tijk}-\mu_{ti}}{\sqrt{\sigma_{ti}^2+ \epsilon}}
$$


对于是否使用BN中的可学习参数$$\beta$$和$$\gamma$$，从LN的TensorFlow中源码中我们可以看出这两个参数是要使用的。但是我们也可以通过将其值置为False来停用它们，这一点和其它归一化方法在TensorFlow中的实现是相同的。

## 1.3 TensorFlow 中的IN

IN在TensorFlow中的实现见[链接](https://github.com/tensorflow/tensorflow/blob/r1.12/tensorflow/contrib/layers/python/layers/normalization.py)，其函数声明如下：

```py
def instance_norm(inputs,
                  center=True,
                  scale=True,
                  epsilon=1e-6,
                  activation_fn=None,
                  param_initializers=None,
                  reuse=None,
                  variables_collections=None,
                  outputs_collections=None,
                  trainable=True,
                  data_format=DATA_FORMAT_NHWC,
                  scope=None)
```

其中的`center`和`scale`便是分别对应BN中的参数$$\beta$$和$$\gamma$$。

归一化统计量是通过`nn.moments`函数计算的，决定如何从inputs取值的是`axes`参数，对应源码中的`moments_axes`参数。

```py
    # Calculate the moments (instance activations).
    mean, variance = nn.moments(inputs, moments_axes, keep_dims=True)
```

下面我们提取源码中的核心部分，并通过注释的方法对齐进行解释（假设输入的Tensor是按NHWC排列的）：

```py
inputs_rank = inputs.shape.ndims # 取Tensor的维度数，这里值是4
reduction_axis = inputs_rank - 1 # 取Channel维的位置，值为3
moments_axes = list(range(inputs_rank)) # 初始化moments_axes链表，值为[0,1,2,3]
del moments_axes[reduction_axis] # 删除第3个值（Channel维），moments_axes变为[0,1,2]
del moments_axes[0] # 删除第一个值（Batch维），moments_axes变为[1,2]
```

## 总结

IN本身是一个非常简单的算法，尤其适用于批量较小且单独考虑每个像素点的场景中，因为其计算归一化统计量时没有混合批量和通道之间的数据，对于这种场景下的应用，我们可以考虑使用IN。

另外需要注意的一点是在图像这类应用中，每个通道上的值是比较大的，因此也能够取得比较合适的归一化统计量。但是有两个场景建议不要使用IN:

1. MLP或者RNN中：因为在MLP或者RNN中，每个通道上只有一个数据，这时会自然不能使用IN；
2. Feature Map比较小时：因为此时IN的采样数据非常少，得到的归一化统计量将不再具有代表性。

## Reference

\[1\] Vedaldi V L D U A. Instance Normalization: The Missing Ingredient for Fast Stylization\[J\]. arXiv preprint arXiv:1607.08022, 2016.

\[2\] Gatys L A, Ecker A S, Bethge M. Image style transfer using convolutional neural networks\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2414-2423.

\[3\] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift\[J\]. arXiv preprint arXiv:1502.03167, 2015.

\[4\] Ba J L, Kiros J R, Hinton G E. Layer normalization\[J\]. arXiv preprint arXiv:1607.06450, 2016.

\[5\] Ulyanov, D., Lebedev, V., Vedaldi, A., and Lempitsky, V. S. \(2016\). Texture networks: Feed-forward synthesis of textures and stylized images. In Proceedings of the 33nd International Conference on Machine Learning, ICML 2016, New York City, NY, USA, June 19-24, 2016, pages 1349–1357.

