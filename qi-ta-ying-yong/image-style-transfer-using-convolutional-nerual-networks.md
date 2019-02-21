# Image Style Transfer Using Convolutional Nerual Networks

tags: Nueral Style Transfer

## 前言

Leon A.Gatys是最早使用CNN做图像风格迁移的先驱之一，这篇文章还有另外一个版本\[2\]，应该是它投到CVPR之前的预印版，两篇文章内容基本相同。

我们知道在训练CNN分类器时，接近输入层的Feature Map包含更多的图像的纹理等细节信息，而接近输出层的Feature Map则包含更多的内容信息。这个特征的原理可以通过我们在[残差网络](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)中介绍的数据处理不等式（DPI）解释：越接近输入层的Feature Map经过的处理（卷积和池化）越少，则这时候损失的图像信息还不会很多。随着网络层数的加深，图像经过的处理也会增多，根据DPI中每次处理信息会减少的原理，靠后的Feature Map则包含的输入图像的信息是不会多余其之前的Feature Map的；同理当我们使用标签值进行参数更新时，越接近损失层的Feature Map则会包含越多的图像标签（内容）信息，越远则包含越少的内容信息。这篇论文正是利用了CNN的天然特征实现的图像风格迁移的。

具体的讲，当我们要在图片$$\vec{p}$$（content）的内容之上应用图片$$\vec{a}$$（style）的风格时，我们会使用梯度下降等算法更新目标图像$$\vec{x}$$（target）的内容，使其在较浅的层有和图片$$\vec{a}$$类似的响应值，同时在较深的层和$$\vec{p}$$也有类似的响应，这样就保证了$$\vec{x}$$和$$\vec{a}$$有类似的风格而且和$$\vec{p}$$有类似的内容，这样生成的图片$$\vec{x}$$就是我们要得到的风格迁移的图片。如图1所示。

在Keras官方源码中，作者提供了神经风格迁移的[源码](https://github.com/keras-team/keras/blob/fcf2ed7831185a282895dda193217c2a97e1e41d/examples/neural_style_transfer.py)，这里对算法的讲解将结合源码进行分析。

<figure>
<img src="/assets/IST_1.jpg" alt="图1：图像风格迁移效果图" />
<figcaption>图1：图像风格迁移效果图</figcaption>
</figure>

## 1. Image Style Transfer（IST）算法详解

### 1.1 算法概览

IST的原理基于上面提到的网络的不同层会响应不同的类型特征的特点实现的。给定一个训练好的网络，源码中使用的是[VGG19](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/very-deep-convolutional-networks-for-large-scale-image-recognition.html) \[3\]，下面是源码第142-143行，因此在运行该源码时如果你之前没有下载过训练好的VGG19模型文件，第一次运行会有下载该文件的过程，文件名为'vgg19\_weights\_tf\_dim\_ordering\_tf\_kernels\_notop.h5'。

```py
142 model = vgg19.VGG19(input_tensor=input_tensor,
143                     weights='imagenet', include_top=False)
```

论文中有两点在源码中并没有体现，一个是对权值进行了归一化，使用的方法是我们之前介绍的[Weight Normalization](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-ba-zhang-ff1a-wang-luo-you-hua/weight-normalization.html)\[4\]，另外一个是使用平均池化代替最大池化，使用了这两点的话会有更快的收敛速度。

图2有三个部分，最左侧的输入是风格图片$$\vec{a}$$，将其输入到训练好的VGG19中，会得到一批它对应的Feature Map；最右侧则是内容图片$$\vec{p}$$，它也会输入到这个网络中得到它对应的Feature Map；中间是目标图片$$\vec{x}$$，它的初始值是白噪音图片，它的值会通过SGD进行更新，SGD的损失函数时通过$$\vec{x}$$在这个网络中得到的Feature Map和$$\vec{a}$$的Feature Map以及$$\vec{p}$$的Feature Map计算得到的。图2中所有的细节会在后面的章节中进行介绍。

<figure>
<img src="/assets/IST_2.png" alt="图2：图像风格迁移算法流程图" />
<figcaption>图2：图像风格迁移算法流程图</figcaption>
</figure>

传统的深度学习方法是根据输入数据更新网络的权值。而IST的算法是固定网络的参数，更新输入的数据。固定权值更新数据还有几个经典案例，例如材质学习\[5\]，卷积核可视化等。

### 1.2 内容表示

内容表示是图2中右侧的两个分支所示的过程。我们先看最右侧，$$\vec{p}$$输入VGG19中，我们提取其在第四个block中第二层的Feature Map，表示为conv4_2（源码中提取的是conv5\_2）。假设其层数为$$l$$，$$N_l$$是Feature Map的数量，也就是通道数，$$M_l$$是Feature Map的像素点的个数。那么我们得到Feature Map $$F^l$$可以表示为$$F^l \in \mathcal{R}^{N_l \times M_l}$$，$$F^l_{ij}$$则是第$$l$$层的第$$i$$个Feature Map在位置$$j$$处的像素点的值。根据同样的定义，我们可以得到$$\vec{x}$$在conv4\_2处的Feature Map $$P^l$$。

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

上式中的`content_weight`是内容损失函数的比重，源码中给出的值是0.025，内容损失函数的定义见185-186行：

```py
185 def content_loss(base, combination):
186     return K.sum(K.square(combination - base))
```

有了损失函数的定义之后，我们便可以根据损失函数的值计算其关于$$F_{i,j}$$的梯度值，从而实现从后向前的梯度更新。


$$
\frac{\partial \mathcal{L}_{content}}{\partial F_{i,j}^l} = 
\left\{
\begin{array}{}
(F^l - P^l)_{i,j} & \text{if } F_{i,j} > 0\\
0 & \text{if } F_{i,j} < 0
\end{array}
\right.
$$


如果损失函数只包含内容损失，当模型收敛时，我们得到的$$\vec{x}'$$应该非常接近$$\vec{p}$$的内容。但是它很难还原到和$$\vec{p}$$一模一样，因为即使损失值为0时，我们得到的$$\vec{x}'$$值也有多种的形式。

为什么说$$\vec{x}'$$具有$$\vec{p}$$的内容呢，因为当$$\vec{x}'$$经过VGG19的处理后，它的conv5\_2层的输出了$$\vec{p}$$几乎一样，而较深的层具有较高的内容信息，这也就说明了$$\vec{x}'$$和$$\vec{p}$$具有非常类似的内容信息。

### 1.3 风格表示

风格表示的计算过程是图2的左侧和中间两个分支。和计算$$F^l$$相同，我们将$$\vec{a}$$输入到模型中便可得到它对应的Feature Map $$S^l$$。不同于内容表示的直接运算，风格表示使用的是Feature Map展开成1维向量的Gram矩阵的形式。使用Gram矩阵的原因是因为考虑到纹理特征是和图像的具体位置没有关系的，所以通过打乱纹理的位置信息来保证这个特征，Gram矩阵的定义如下：


$$
G_{i,j}^l = \sum_k F_{i,k}^l F_{j,k}^l
$$


另外一点和内容表示不同的是，风格表示使用了每个block的第一个卷积来计算损失函数，作者认为这种方式得到的纹理特征更为光滑，因为仅仅使用底层Feature Map得到的图像较为精细但是比较粗糙，而高层得到的图像则含有更多的内容信息，损失了一些纹理信息，但他的材质更为光滑。所以，综合了所有层的样式表示的损失函数为：


$$
L_{style} = \sum_l w_l E_l
$$


其中$$E_l$$是$$S^l$$的Gram矩阵$$A^l$$和$$F^l$$的Gram矩阵$$G^l$$的均方误差：


$$
E_l = \frac{1}{4N_l^2M_l^2}\sum_{i,j}(G_{i,j}^l - A_{i,j}^l)^2
$$


它关于$$F_{i,j}^l$$的梯度的计算方式为：


$$
\frac{\partial E_l}{\partial F_{i,j}^l} =
 \left\{
 \begin{array}{}
 \frac{1}{N_l^2M_l^2}((F^l)^T(G^l-A^l))_{ji} & \text{if } F_{i,j}^l > 0 \\
 0 & \text{if } F_{i,j}^l < 0
 \end{array}
 \right.
$$


上面的更新同样使用SGD。

下面我们继续来学习源码，从源码的214-223行我们可以看出样式表示使用了5个block的Feature Map：

```py
214 feature_layers = ['block1_conv1', 'block2_conv1',
215                   'block3_conv1', 'block4_conv1',
216                   'block5_conv1']
217 for layer_name in feature_layers:
218     layer_features = outputs_dict[layer_name]
219     style_reference_features = layer_features[1, :, :, :]
220     combination_features = layer_features[2, :, :, :]
221     sl = style_loss(style_reference_features, combination_features)
222     loss += (style_weight / len(feature_layers)) * sl
223 loss += total_variation_weight * total_variation_loss(combination_image)
```

从上面的代码中我们可以看出，样式表示使用了`feature_layers`中所包含的Feature Map，并且最后loss的计算把它们进行了相加。第221行的`style_loss`的定义见源码的171-178行：

```py
171 def style_loss(style, combination):
172     assert K.ndim(style) == 3
173     assert K.ndim(combination) == 3
174     S = gram_matrix(style)
175     C = gram_matrix(combination)
176     channels = 3
177     size = img_nrows * img_ncols
178     return K.sum(K.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))
```

从174-175行我们可以看出损失函数的计算使用的是两个Feature Map的Gram矩阵，Gram矩阵的定义见155-162行：

```py
155 def gram_matrix(x):
156     assert K.ndim(x) == 3
157     if K.image_data_format() == 'channels_first':
158         features = K.batch_flatten(x)
159     else:
160         features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
161     gram = K.dot(features, K.transpose(features))
162     return gram
```

第158或者160行的`batch_flatten`验证了Feature Map要先展开成向量，第161行则是Gram矩阵的计算公式。

还有一些超餐在配置文件中进行了指定，`style_weight`和`total_variation_weight`的默认值都是1。

### 1.4 风格迁移

明白了如何计算内容损失函数$$\mathcal{L}_{content}$$和风格损失函数$$\mathcal{L}_{style}$$之后，整个风格迁移任务的损失函数就是两个损失值得加权和：


$$
\mathcal{L}_{\text{total}}(\vec{p},\vec{a},\vec{x}) = \alpha \mathcal{L}_{\text{content}}(\vec{p}, \vec{x}) + \beta \mathcal{L}_{\text{style}}(\vec{a}, \vec{x})
$$


其中$$\alpha$$和$$\beta$$就是我们在1.2节和1.3节介绍的`content_weight`和`total_variation_weight`。通过调整这两个超参数的值我们可以设置生成的图像更偏向于$$\vec{p}$$的内容还是$$\vec{a}$$的风格。$$\frac{\partial \mathcal{L}_{total}}{\partial \vec{x}}$$的值用来更新输入图像$$\vec{x}$$的内容，作者推荐使用L-BFGS更新梯度。

另外对于$$\vec{x}$$的初始化，论文中推荐使用白噪音进行初始化，这样虽然计算的时间要更长一些，但是得到的图像的样式具有更强的随机性。而论文使用的是使用$$\vec{p}$$初始化$$\vec{x}$$，这样得到的生成图像更加稳定。

下面继续学习这一部分的源码。在第287-288行的`fmin_l_bfgs_b`说明了计算梯度使用了L-BFGS算法，它是scipy提供：

```py
287 x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
288                                  fprime=evaluator.grads, maxfun=20)
```

`fmin_l_bfgs_b`是scipy包中一个函数。第一个参数是定义的损失函数，第二个参数是输入数据，`fprime`通常用于计算第一个损失函数的梯度，`maxfun`是函数执行的次数。它的第一个返回值是更新之后的x的值，这里使用了递归的方式反复更新x，第二个返回值是损失值。

其中`x`的初始化使用的是内容图片$$\vec{p}$$:

```py
282 x = preprocess_image(base_image_path)
```

287行的损失函数定义在264-269行：

```py
264 def loss(self, x):
265     assert self.loss_value is None
266     loss_value, grad_values = eval_loss_and_grads(x)
267     self.loss_value = loss_value
268     self.grad_values = grad_values
269     return self.loss_value
```

其中最重要的函数是`eval_loss_and_grads()`函数，它定义在了237-248行：

```py
237 def eval_loss_and_grads(x):
238     if K.image_data_format() == 'channels_first':
239         x = x.reshape((1, 3, img_nrows, img_ncols))
240     else:
241         x = x.reshape((1, img_nrows, img_ncols, 3))
242     outs = f_outputs([x])
243     loss_value = outs[0]
244     if len(outs[1:]) == 1:
245         grad_values = outs[1].flatten().astype('float64')
246     else:
247         grad_values = np.array(outs[1:]).flatten().astype('float64')
248     return loss_value, grad_values
```

其中`f_outputs()`是实例化的Keras函数，作用是使用梯度更新$$\vec{x}$$的内容，见226-234行：

```py
226 grads = K.gradients(loss, combination_image)
227 
228 outputs = [loss]
229 if isinstance(grads, (list, tuple)):
230     outputs += grads
231 else:
232     outputs.append(grads)
233 
234 f_outputs = K.function([combination_image], outputs)
```

## 2. 总结

图像风格迁移是一个非常好玩但是无法对齐效果量化的算法，我们可以得到和一些著名画家风格看起来非常类似的画作，但是很难从数学的角度去衡量一个画作的风格，得出的结论是非常主观的。但是算法的设计动机是出于CNN的**底层Feature Map接近图像纹理而高层Feature Map接近图像内容**的天然特性，也是对神经网络这个黑盒子从另外一个角度给与了解释。IST产生的结果非常有趣，由此诞生了一批商用的软件，例如Prisma等。

IST如果能迁移到音频领域也许会有帮助，例如在TTS中如果可以将合成的语音的内容应用到真实人类语音的风格上，这样也许可以得到更为平滑的语音。或者如果我们将音频内容应用到某个人说话的风格中，也许我们可以得到和这个人说话风格非常类似的音频输出。

算法另外一个缺点是对噪音比较敏感，尤其是当参与合成的风格图片和内容图片都是真实照片的时候。


## Reference

\[1\] Gatys L A, Ecker A S, Bethge M. Image style transfer using convolutional neural networks\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2414-2423.

\[2\] Gatys L A, Ecker A S, Bethge M. A neural algorithm of artistic style\[J\]. arXiv preprint arXiv:1508.06576, 2015.

\[3\] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition\[J\]. arXiv preprint arXiv:1409.1556, 2014.

\[4\] Salimans T, Kingma D P. Weight normalization: A simple reparameterization to accelerate training of deep neural networks\[C\]//Advances in Neural Information Processing Systems. 2016: 901-909.

\[5\] Gatys L, Ecker A S, Bethge M. Texture synthesis using convolutional neural networks\[C\]//Advances in Neural Information Processing Systems. 2015: 262-270.

