# Image Style Transfer Using Convolutional Nerual Networks

tags: Nueral Style Transfer

## 前言

Leon A.Gatys是最早使用CNN做图像风格迁移的先驱之一，这篇文章还有另外一个版本\[2\]，应该是它投到CVPR之前的预印版，两篇文章内容基本相同。

我们知道在训练CNN分类器时，接近输入层的Feature Map包含更多的图像的纹理等细节信息，而接近输出层的Feature Map则包含更多的内容信息。这个特征的原理可以通过我们在[残差网络](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)中介绍的数据处理不等式（DPI）解释：越接近输入层的Feature Map经过的处理（卷积和池化）越少，则这时候损失的图像信息还不会很多。随着网络层数的加深，图像经过的处理也会增多，根据DPI中每次处理信息会减少的原理，靠后的Feature Map则包含的输入图像的信息是不会多余其之前的Feature Map的；同理当我们使用标签值进行参数更新时，越接近损失层的Feature Map则会包含越多的图像标签（内容）信息，越远则包含越少的内容信息。这篇论文正是利用了CNN的天然特征实现的图像风格迁移的。

具体的讲，当我们要在图片$$\vec{p}$$（content）的内容之上应用图片$$\vec{a}$$（style）的风格时，我们会使用梯度下降等算法更新目标图像$$\vec{x}$$（target）的内容，使其在较浅的层有和图片$$\vec{a}$$类似的响应值，同时在较深的层和$$\vec{p}$$也有类似的响应，这样就保证了$$\vec{x}$$和$$\vec{a}$$有类似的风格而且和$$\vec{p}$$有类似的内容，这样生成的图片$$\vec{x}$$就是我们要得到的风格迁移的图片。如图1所示。

在Keras官方源码中，作者提供了神经风格迁移的[源码](https://github.com/keras-team/keras/blob/fcf2ed7831185a282895dda193217c2a97e1e41d/examples/neural_style_transfer.py)，这里对算法的讲解将结合源码进行分析。

## 1. Image Style Transfer（IST）算法详解

### 1.1 s

IST的原理基于上面提到的网络的不同层会响应不同的类型特征的特点实现的。给定一个训练好的网络，源码中使用的是[VGG19](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/very-deep-convolutional-networks-for-large-scale-image-recognition.html) \[3\]，下面是源码第142-143行，因此在运行该源码时如果你之前没有下载过训练好的VGG19模型文件，第一次运行会有下载该文件的过程，文件名为'vgg19\_weights\_tf\_dim\_ordering\_tf\_kernels\_notop.h5'。

```py
142 model = vgg19.VGG19(input_tensor=input_tensor,
143                     weights='imagenet', include_top=False)
```

论文中有两点在源码中并没有体现，一个是对权值进行了归一化，使用的方法是我们之前介绍的[Weight Normalization](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-ba-zhang-ff1a-wang-luo-you-hua/weight-normalization.html)\[4\]，另外一个是使用平均池化代替最大池化，使用了这两点的话会有更快的收敛速度。

图2有三个部分，最左侧的输入是风格图片$$\vec{a}$$，将其输入到训练好的VGG19中，会得到一批它对应的Feature Map；最右侧则是内容图片$$\vec{p}$$，它也会输入到这个网络中得到它对应的Feature Map；中间是目标图片$$\vec{x}$$，它的初始值是白噪音图片，它的值会通过SGD进行更新，SGD的损失函数时通过$$\vec{x}$$在这个网络中得到的Feature Map和$$\vec{a}$$的Feature Map以及$$\vec{p}$$的Feature Map计算得到的。图2中所有的细节会在后面的章节中进行介绍。

![](/assets/IST_2.png)

传统的深度学习方法是根据输入数据更新网络的权值。而IST的算法是固定网络的参数，更新输入的数据。固定权值更新数据还有几个经典案例，例如材质学习\[5\]，卷积核可视化等。

### 1.2 内容表示

内容表示是图2中右侧的两个分支所示的过程。我们先看最右侧，$$\vec{p}$$输入VGG19中，我们提取其在第四个block中第二层的Feature Map，表示为conv4_2（源码中提取的是conv5\_2）。假设其层数为_$$l$$_，_$$N_l$$_是Feature Map的数量，也就是通道数，_$$M_l$$_是Feature Map的像素点的个数。那么我们得到Feature Map _$$F^l$$_可以表示为_$$F^l \in \mathcal{R}^{N_l \times M_l}$$_，$$F^l_{ij}$$则是第$$l$$层的第$$i$$个Feature Map在位置$$j$$处的像素点的值。根据同样的定义，我们可以得到$$\vec{x}$$在conv4_2处的Feature Map $$P^l$$。

如果$$\vec{x}$$的$$F_l$$和$$\vec{p}$$的$$P^l$$非常接近，那么我们可以认为$$\vec{x}$$和$$\vec{p}$$在内容上比较接近，因为越接近输出的层包含有越多的内容信息。这里我们可以定义IST的内容损失函数为：


$$
\mathcal{L}_{\text{content}}(\vec{p},\vec{x},l)=\frac{1}{2}\sum_{i,j}(F_{i,j}^l - P_{i,j}^l)^2
$$


下面我们来看一下源码，上面142行的`input_tensor`的是由$$\vec{p}, \vec{a}, \vec{x}$$一次拼接而成的，见136-138行。

```py
136 input_tensor = K.concatenate([base_image,
137                               style_reference_image,
138                               combination_image], axis=0)
```

通过对142行的`model`的遍历我们可以得到每一层的Feature Map的名字以及内容，然后将其保存在字典中，见147行。

```py
147 outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
```

这样我们可以根据关键字提取我们想要的Feature Map，例如我们提取两个图像在conv5\_2处的Feature Map $$P^l$$（源码中的`base_image_features`）和$$F^l$$源码中的`combination_features`），然后使用这两个Feature Map计算损失值，见208-212行：

```py
208 layer_features = outputs_dict['block5_conv2']
209 base_image_features = layer_features[0, :, :, :]
210 combination_features = layer_features[2, :, :, :]
211 loss += content_weight * content_loss(base_image_features,
212                                       combination_features)
```

上式中的```content_weight```是内容损失函数的比重，源码中给出的值是0.025，内容损失函数的定义见185-186行：

```py
185 def content_loss(base, combination):
186     return K.sum(K.square(combination - base))

```

有了损失函数的定义之后，我们便可以根据损失函数的值计算其关于$$F_{i,j}$$的梯度值，从而实现从后向前的梯度更新。

$$
\frac{\partial \mathcal{L}_{content}}{\partial F_{i,j^l}} = 
\left\{
\begin{array}{}
(F^l - P^l)_{i,j} & \text{if } F_{i,j} > 0\\
0 & \text{if } F_{i,j} < 0
\end{array}
\right.
$$

如果损失函数只包含内容损失，当模型收敛时，我们得到的$$\vec{x}’$$应该非常接近$$\vec{p}$$的内容。但是它很难还原到和$$\vec{p}$$一模一样，因为即使损失值为0时，我们得到的$$\vec{x}'$$值也有多种的形式。

为什么说$$\vec{x}’$$具有$$\vec{p}$$的内容呢，因为当$$\vec{x}’$$经过VGG19的处理后，它的conv5_2层的输出了$$\vec{p}$$几乎一样，而较深的层具有较高的内容信息，这也就说明了$$\vec{x}’$$和$$\vec{p}$$具有非常类似的内容信息。

### 1.3 样式表示


## Reference

\[1\] Gatys L A, Ecker A S, Bethge M. Image style transfer using convolutional neural networks\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2414-2423.

\[2\] Gatys L A, Ecker A S, Bethge M. A neural algorithm of artistic style\[J\]. arXiv preprint arXiv:1508.06576, 2015.

\[3\] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition\[J\]. arXiv preprint arXiv:1409.1556, 2014.

\[4\] Salimans T, Kingma D P. Weight normalization: A simple reparameterization to accelerate training of deep neural networks\[C\]//Advances in Neural Information Processing Systems. 2016: 901-909.

\[5\] Gatys L, Ecker A S, Bethge M. Texture synthesis using convolutional neural networks\[C\]//Advances in Neural Information Processing Systems. 2015: 262-270.

