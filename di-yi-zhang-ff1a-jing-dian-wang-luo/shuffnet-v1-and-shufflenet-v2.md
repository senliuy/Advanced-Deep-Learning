# ShuffNet v1 and ShuffleNet v2

tags: ShuffNet v1, ShuffleNet v2



## 前言

在[ResNeXt](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/aggregated-residual-transformations-for-deep-neural-networks.html)[3]的文章中，分组卷积作为传统卷积核深度可分离卷积的一种折中方案被采用。这时大量的对于整个Feature Map的Pointwise卷积成为了ResNeXt的性能瓶颈。一种更高效的策略是在组内进行Pointwise卷积，但是这种组内Pointwise卷积的形式不利于通道之间的信息流通，为了解决这个问题，ShuffleNet v1中提出了通道洗牌（channel shuffle）操作。

在ShuffleNet v2的文章中作者指出现在普遍采用的FLOPs评估模型性能是非常不合理的，因为一批样本的训练时间除了看FLOPs，还有很多过程需要消耗时间，例如文件IO，内存读取，GPU执行效率等等。作者从内存消耗成本，GPU并行性两个方向分析了模型可能带来的非FLOPs的行动损耗，进而设计了更加高效的ShuffleNet v2。ShuffleNet v2的架构和[DenseNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/densely-connected-convolutional-networks.html)[4]有异曲同工之妙，而且其速度和精度都要优于DenseNet。

## Reference

[1] Zhang, X., Zhou, X., Lin, M., Sun, J.: Shufflenet: An extremely efficient convolu-tional neural network for mobile devices. arXiv preprint arXiv:1707.01083 (2017)

[2] Ma N, Zhang X, Zheng H T, et al. Shufflenet v2: Practical guidelines for efficient cnn architecture design[J]. arXiv preprint arXiv:1807.11164, 2018.

[3] Xie S, Girshick R, Dollár P, et al. Aggregated residual transformations for deep neural networks\[C\]//Computer Vision and Pattern Recognition \(CVPR\), 2017 IEEE Conference on. IEEE, 2017: 5987-5995.

[4] Huang G, Liu Z, Weinberger K Q, et al. Densely connected convolutional networks[C]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017, 1(2): 3.


