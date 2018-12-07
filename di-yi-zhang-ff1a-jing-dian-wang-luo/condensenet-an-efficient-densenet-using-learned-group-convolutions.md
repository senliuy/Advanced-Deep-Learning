# CondenseNet: An Efficient DenseNet using Learned Group Convolutions

# 前言

CondenseNet是黄高团队对其[DenseNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/densely-connected-convolutional-networks.html)\[2\]的升级。DenseNet的密集连接其实是存在冗余的，其最大的影响便是影响网络的效率。首先，为了降低DenseNet的冗余问题，CondenseNet提出了在训练的过程中对不重要的权值进行剪枝，即学习一个稀疏的网络。但是测试的整个过程就是一个简单的卷积，因为网络已经在训练的时候优化完毕。其次，为了进一步提升效率，CondenseNet在$$1\times1$$卷积的时候使用了分组卷积，分组卷积在AlexNet中首先应用于双GPU架构，并在[ResNeXt](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/aggregated-residual-transformations-for-deep-neural-networks.html)\[3\]中作为性能提升的策略首次被提出。最后，CondenseNet中指出临近的特征重用更重要，因此采用了指数增长的成长率（Growth Rate），并在DenseNet的block之间也添加了short-cut。

DenseNet，CondenseNet的训练和测试阶段的示意图如图1。其中的细节我们会在后面的部分详细解析。

![](/assets/CondenseNet_1.png)

## 1. CondenseNet详解

### 1.1 分组卷积的问题

在[ShuffleNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/shuffnet-v1-and-shufflenet-v2.html)\[4\]中我们指出分组卷积存在通道之间的信息沟通不畅以及特征多样性不足的问题。CondenseNet提出的解决策略是在训练的过程中让模型选择更好的分组方式，理论上每个通道的Feature Map是可以和所有Feature Map沟通到的。传统的沟通不畅的分组卷积自然不可能被学习到。图2是普通卷积核分组卷积的示意图。

![](/assets/CondenseNet_2.png)

我们换一个角度来看分组卷积，它也可以别看做普通卷积的稀疏表示，只不过指着稀疏方式是由认为生硬的指定的。这种稀疏连接虽然高效，但是人为的毫无根据的指定那些连接重要，哪些连接需要被删除无疑非常不合理。CondenseNet指出的解决方案是使用训练数据学习卷积网络的稀疏表示，让识别精度决定哪些权值该被保留，这个过程叫做_learning group convolution_，即图1中间红色的'L-Conv'。

### 1.2 自学习分组卷积

如图3所示，自学习分组卷积（Learned Group Convolution）可以分成两个阶段：浓缩（Condensing）阶段和优化（Optimizing）阶段。其中浓缩阶段用于剪枝没用的特征，优化阶段用于优化剪枝之后的网络。

![](/assets/CondenseNet_3.png)

在图3中分组数$$G=3$$。压缩率$$C=3$$，即只保留原来1/3的特征。

浓缩阶段1（图3的最左侧）是普通的卷积网络，在训练该网络时使用了分组lasso正则项，这样学到的特征会呈现结构化稀疏分布，好处是在后面剪枝部分不会过分的影响精度。


## Reference

\[1\] Huang G, Liu S, van der Maaten L, et al. CondenseNet: An Efficient DenseNet using Learned Group Convolutions\[J\]. group, 2017, 3\(12\): 11.

\[2\] Huang G, Liu Z, Weinberger K Q, et al. Densely connected convolutional networks\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017, 1\(2\): 3.

\[3\] Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks\[C\]//Computer Vision and Pattern Recognition \(CVPR\), 2017 IEEE Conference on. IEEE, 2017: 5987-5995.

\[4\] Zhang, X., Zhou, X., Lin, M., Sun, J.: Shufflenet: An extremely efficient convolutional neural network for mobile devices. arXiv preprint arXiv:1707.01083 \(2017\)

