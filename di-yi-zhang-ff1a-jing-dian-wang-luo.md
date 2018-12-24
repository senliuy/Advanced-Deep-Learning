# 第一章：经典网络

物体分类是深度学习中最经典也是目前研究的最为透彻的一个领域，该领域的开创者也是深度学习的名人堂级别的人物，例如Geoffrey Hinton, Yoshua Bengio等。物体分类常见的数据集由数字数据集MNIST，物体数据集CIFAR-10和类别更多的CIFAR-100，以及任何state-of-the-art的网络实验都规避不了的超大数据集ImageNet。ImageNet是李飞飞教授主办的ILSVRC比赛中使用的数据集，ILSVRC的每年比赛中产生的网络也指引了卷积网络的发展方向。

2012年是ILSVRC的第三届比赛，这次比赛的冠军作品是Hinton团队的[AlexNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/imagenet-classification-with-deep-convolutional-neural-networks.html)\[1\]（图1），他们将2011年的top-5错误率从25.8%降低到16.4%。他们的最大贡献在于验证了卷积操作在大数据集上的有效性，从此物体分类进入了深度学习时代。

![](/assets/AlexNet_3.png)

2013年的ILSVRC已由深度学习算法霸榜，其冠军网络是ZFNet\[2\]。ZFNet使用了更深的深度，并且在论文中给出了CNN的有效性的初步解释。

![](/assets/ZFNet_1.png)

## Reference

\[1\] Krizhevsky A, Sutskever I, Hinton G E. Imagenet classification with deep convolutional neural networks\[C\]//Advances in neural information processing systems. 2012: 1097-1105.

\[2\] Zeiler M D, Fergus R. Visualizing and understanding convolutional networks\[C\]//European conference on computer vision. Springer, Cham, 2014: 818-833.

