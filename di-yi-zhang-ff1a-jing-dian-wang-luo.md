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

2016年的冠军是商汤可以和港中文联合推出的CUImage，它是6个模型的模型集成，并无创新性，此处不再赘述。



## Reference

\[1\] Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks\[C\]//Advances in neural information processing systems. 2012: 1097-1105.

\[2\] Zeiler M D, Fergus R. Visualizing and understanding convolutional networks\[C\]//European conference on computer vision. Springer, Cham, 2014: 818-833.

[3] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition[J]. arXiv preprint arXiv:1409.1556, 2014.                                                                                       

[4] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

