# Image Style Transfer Using Convolutional Nerual Networks

tags: Nueral Style Transfer

## 前言

Leon A.Gatys是最早使用CNN做图像风格迁移的先驱之一，这篇文章还有另外一个版本[2]，应该是它投到CVPR之前的预印版，两篇文章内容基本相同。

我们知道在训练CNN分类器时，接近输入层的Feature Map包含更多的图像的纹理等细节信息，而接近输出层的Feature Map则包含更多的内容信息。这个特征的原理可以通过我们在[残差网络](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)中介绍的数据处理不等式（DPI）解释：越接近输入层的Feature Map经过的处理（卷积和池化）越少，则这时候损失的图像信息还不会很多。随着网络层数的加深，图像经过的处理也会增多，根据DPI中每次处理信息会减少的原理，靠后的Feature Map则包含的输入图像的信息是不会多余其之前的Feature Map的；同理当我们使用标签值进行参数更新时，越接近损失层的Feature Map则会包含越多的图像标签（内容）信息，越远则包含越少的内容信息。这篇论文正是利用了CNN的天然特征实现的图像风格迁移的。

具体的讲，当我们要在图片C（content）的内容之上应用图片S（style）的风格时，我们会使用梯度下降等算法更新目标图像T（target）的内容，使其在较浅的层有和图片S类似的响应值，同时在较深的层和C也有类似的响应，这样就保证了T和S有类似的风格而且和C有类似的内容，这样生成的图片T就是我们要得到的风格迁移的图片。如图1所示。

在Keras官方源码中，作者提供了神经风格迁移的[源码](https://github.com/keras-team/keras/blob/fcf2ed7831185a282895dda193217c2a97e1e41d/examples/neural_style_transfer.py)，这里对算法的讲解将结合源码进行分析。

## 1. Image Style Transfer算法详解



## Reference

[1] Gatys L A, Ecker A S, Bethge M. Image style transfer using convolutional neural networks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2414-2423.

[2] Gatys L A, Ecker A S, Bethge M. A neural algorithm of artistic style[J]. arXiv preprint arXiv:1508.06576, 2015.

[3]