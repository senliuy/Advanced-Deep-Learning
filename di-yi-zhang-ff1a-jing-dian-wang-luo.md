# 第一章：经典网络

物体分类是深度学习中最经典也是目前研究的最为透彻的一个领域，该领域的开创者也是深度学习的名人堂级别的人物，例如Geoffrey Hinton, Yoshua Bengio等。物体分类常见的数据集由数字数据集MNIST，物体数据集CIFAR-10和类别更多的CIFAR-100，以及任何state-of-the-art的网络实验都规避不了的超大数据集ImageNet。ImageNet是李飞飞教授主办的ILSVRC比赛中使用的数据集，ILSVRC的每年比赛中产生的网络也指引了卷积网络的发展方向。

2012年是ILSVRC的第三届比赛，这次比赛的冠军作品是Hinton团队的[AlexNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/imagenet-classification-with-deep-convolutional-neural-networks.html)\[1\]（图1），他们将2011年的top-5错误率从25.8%降低到16.4%。他们的最大贡献在于验证了卷积操作在大数据集上的有效性，从此物体分类进入了深度学习时代。

![](/assets/AlexNet_3.png)

2013年的ILSVRC已由深度学习算法霸榜，其冠军网络是ZFNet\[2\]。ZFNet使用了更深的深度，并且在论文中给出了CNN的有效性的初步解释。

![](/assets/ZFNet_1.png)

2014年是深度学习领域经典算法最为井喷的一年，在物体检测方向也是如此。这一届比赛的冠军是谷歌团队提出的[GoogLeNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/going-deeper-with-convolutions.html)[3] (top5：7.3%)，亚军则是牛津大学的[VGG](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/very-deep-convolutional-networks-for-large-scale-image-recognition.html)[4] (top5：6.7%)，但是在分类任务中VGG则击败GoogLeNet成为冠军。

VGG（图3）提出了搭建卷积网络的几个思想在现在依旧非常具有指导性，例如按照降采样的分布对网络进行分块，使用小卷积核，每次降采样之后Feature Map的数量加倍等等。另外VGG使用了当初贾扬清提出的Caffe[5]作为深度学习框架并开源了其模型，再凭借其比GoogLeNet更高效的特性，使VGG很快占有了大量的市场，尤其是在物体检测领域。VGG也将卷积网络凭借增加深度来提升精度推上了最高峰。

![](/assets/VGG_1.png)

GoogLeNet（图4）则从特征多样性的角度研究了卷积网络，GoogLeNet的特征多样性是基于一种并行的使用了多个不同尺寸的卷积核的单元来完成的。GoogLeNet的最大贡献在于指出卷积网络精度的增加不仅仅可以依靠深度，增加网络的复杂性也是一种有效的策略。

![](/assets/GoogLeNet_1.png)

2015年的冠军网络是何恺明等人提出的[残差网络](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)[5]（图5，top5：3.57%）。他们指出卷积网络的精度并不会随着深度的增加而增加，导致问题的原因是网络的退化问题。残差网络的核心思想是企图通过向网络中添加直接映射（跳跃连接）的方式解决退化问题。由于残差网络的简单易用的特征使其成为了目前使用的最为广泛的网络结构之一。

![](/assets/ResNet_8.jpg)

2016年ILSVRC的前几名都是模型集成，卷积网络的开创性结构陷入了短暂的停滞。当年的冠军是商汤可以和港中文联合推出的CUImage，它是6个模型的模型集成，并无创新性，此处不再赘述。

2017年是ILSVRC比赛的最后一届，这一届的冠军团队由Face++获得，他们提出了基于注意力机制的[SENet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/squeeze-and-excitation-networks.html)[6]（图6，top5：2.21%），该方法通过自注意力（self-attention）机制为每个Feature Map学习一个权重。

![](/assets/SENet_1.png)

另外一个非常重要的网络是黄高团队于CVPR2017中提出的[DenseNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/densely-connected-convolutional-networks.html)[7]，本质上是各个单元都有连接的密集连接结构（图7）。

![](/assets/DenseNet_1.png)

除了ILSVRC的比赛中个冠军作品们之外，在提升网络精度中还有一些值得学习的算法。例如[Inception的几个变种](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/going-deeper-with-convolutions.html)[8, 9, 10]。基于多项式提出的[PolyNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/polynet-a-pursuit-of-structural-diversity-in-very-deep-networks.html)[11]，PolyNet采用了更加多样性的特征。

卷积网络的另外一个方向是轻量级的网络，即在不大程度损失模型精度的前提下，尽可能的压缩模型的大小，提升预测的速度。

轻量级网络的第一个尝试是[SqueezeNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/squeezenet-alexnet-level-accuracy-with-50x-fewer-parameters-and-05mb-model-size.html)[12]，SqueezeNet的策略是使用一部分$$1\times1$$卷积代替$$3\times3$$卷积，它对标的模型是AlexNet。

轻量级网络最经典的策略是深度可分离卷积的提出，经典算法包括[MobileNetv1](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/mobilenetxiang-jie.html)[13]和[Xception](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/xception-deep-learning-with-depthwise-separable-convolutions.html)[14]。深度可分离卷积由深度卷积和单位卷积组成，深度卷积一般是以通道为单位的$$3\times3$$卷积，在这个过程中不同通道之间没有消息交换。而信息交换则由单位卷积完成，单位卷积就是标准的$$1\times1$$卷积。深度可分离卷积的一个比较新的方法是[MobileNetv2](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/mobilenetxiang-jie.html)[15]，它将深度可分离卷积和残差结构进行了结合，并通过一些列理论分析和实验得出了一种更优的结合方式。

轻量级网络的另外一种策略是在传统卷积和深度可分离卷积中的一个折中方案，是由[ResNeXt](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/aggregated-residual-transformations-for-deep-neural-networks.html)[16]中提出的，所谓分组卷积是指在深度卷积中以几个通道为一组的普通卷积。[ShuffleNetv1](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/shuffnet-v1-and-shufflenet-v2.html)[17]提出了通道洗牌策略以加强不同通道之间的信息流通，[ShuffleNetv2](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/shuffnet-v1-and-shufflenet-v2.html)[18]则是通过分析整个测试时间，提出了对内存访问更高效的ShuffleNetv2。ShuffleNetv2得出的结构是一种和DenseNet非常近似的密集连接结构。黄高团队的[CondenseNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/condensenet-an-efficient-densenet-using-learned-group-convolutions.html)[22]则是通过为每个分组学习一个索引层的形式来完成通道直接的信息流通的。

目前在ImageNet上表现最好的是谷歌DeepMind团队提出的NAS[19,20,21]系列文章，他们的核心观点是使用强化学习来生成一个完整的网络或是一个网络节点。[NAS](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/neural-architecture-search-with-reinforecement-learning.html)[19]是该系列的第一篇文章，它使用了强化学习在CIFAR-10上学习到了一个类似于DenseNet的完整的密集连接的网络，如图8。

![](/assets/NAS_4.png)

[NASNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/learning-transferable-architectures-for-scalable-image-recognition.html)[20]解决了NAS不能应用在ImageNet上的问题，它学习的不再是一个完整的网络而是一个网络单元，见图9。这种单元的结构往往比NAS网络要简答得多，因此学习起来效率更高；而且通过堆叠更多NASNet单元的形式可以非常方便的将其迁移到其它任何数据集，包括权威的ImageNet。

![](/assets/NASNet_2.png)

[PNASNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/progressive-neural-architecture-search.html)[21]则是一个性能更高的强化学习方法，其比NASNet具有更小的搜索空间，而且使用了启发式搜索，策略函数等强化学习领域的方法又花了网络超参的学习过程，其得到的网络也是目前ImageNet数据集上效果最好的网络。网络结构见图10。


![](/assets/PNASNet_2.png)

## Reference

\[1\] Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks\[C\]//Advances in neural information processing systems. 2012: 1097-1105.

\[2\] Zeiler M D, Fergus R. Visualizing and understanding convolutional networks\[C\]//European conference on computer vision. Springer, Cham, 2014: 818-833.

[3] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.                                                                                       

[4] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

[5] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

[6] Hu J, Shen L, Sun G. Squeeze-and-excitation networks[J]. arXiv preprint arXiv:1709.01507, 2017, 7.

[7] Huang G, Liu Z, Weinberger K Q, et al. Densely connected convolutional networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017, 1(2): 3.

[8] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift[J]. arXiv preprint arXiv:1502.03167, 2015.

[9] Szegedy C, Vanhoucke V, Ioffe S, et al. Rethinking the inception architecture for computer vision[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 2818-2826.

[10] Szegedy C, Ioffe S, Vanhoucke V, et al. Inception-v4, inception-resnet and the impact of residual connections on learning[C]//AAAI. 2017, 4: 12.

[11] Zhang X, Li Z, Loy C C, et al. Polynet: A pursuit of structural diversity in very deep networks[C]//Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on. IEEE, 2017: 3900-3908.

[12] Iandola F N, Han S, Moskewicz M W, et al. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size[J]. arXiv preprint arXiv:1602.07360, 2016.

[13] Howard A G, Zhu M, Chen B, et al. Mobilenets: Efficient convolutional neural networks for mobile vision applications[J]. arXiv preprint arXiv:1704.04861, 2017.

[14] Chollet F. Xception: Deep learning with depthwise separable convolutions[J]. arXiv preprint, 2017: 1610.02357.

[15] Sandler M, Howard A, Zhu M, et al. MobileNetV2: Inverted Residuals and Linear Bottlenecks[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2018: 4510-4520.

[16] Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks[C]//Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on. IEEE, 2017: 5987-5995.

[17] Zhang, X., Zhou, X., Lin, M., Sun, J.: Shufflenet: An extremely efficient convolutional neural network for mobile devices. arXiv preprint arXiv:1707.01083 (2017)

[18] Ma N, Zhang X, Zheng H T, et al. Shufflenet v2: Practical guidelines for efficient cnn architecture design[J]. arXiv preprint arXiv:1807.11164, 2018.

[19] Liu C, Zoph B, Shlens J, et al. Progressive neural architecture search[J]. arXiv preprint arXiv:1712.00559, 2017.

[20] Zoph B, Le Q V. Neural architecture search with reinforcement learning[J]. arXiv preprint arXiv:1611.01578, 2016.

[21] Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition[J]. arXiv preprint arXiv:1707.07012, 2017, 2(6).
