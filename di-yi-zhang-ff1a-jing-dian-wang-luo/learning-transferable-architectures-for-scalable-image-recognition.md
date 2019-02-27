# Learning Transferable Architectures for Scalable Image Recognition

tags: NAS, NASNet, AutoML

## 前言

在[NAS](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/neural-architecture-search-with-reinforecement-learning.html){{"zoph2016neural"|cite}}一文中我们介绍了如何使用强化学习学习一个完整的CNN网络或是一个独立的RNN单元，这种dataset interest的网络的效果也是目前最优的。但是NAS提出的网络的计算代价是相当昂贵的，仅仅在CIFAR-10上学习一个网络就需要500台GPU运行28天才能找到最优结构。这使得NAS很难迁移到大数据集上，更不要提ImageNet这样几百G的数据规模了。而在目前的行内规则上，如果不能在ImageNet上取得令人信服的结果，你的网络结构很难令人信服的。

为了将NAS迁移到大数据集乃至ImageNet上，这篇文章提出了在小数据（CIFAR-10）上学习一个网络单元（Cell），然后通过堆叠更多的这些网络单元的形式将网络迁移到更复杂，尺寸更大的数据集上面。因此这篇文章的最大贡献便是介绍了如何使用强化学习学习这些网络单元。作者将用于ImageNet的NAS简称为NASNet{{"zoph2018learning"|cite}}，文本依旧采用NASNet的简称来称呼这个算法。实验数据也证明了NASNet的有效性，其在ImageNet的top-1精度和top-5精度均取得了当时最优的效果。

阅读本文前，强烈建议移步到我的《[Neural Architecture Search with Reinforecement Learning](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/neural-architecture-search-with-reinforecement-learning.html)》介绍文章中，因为本文并不会涉及强化学习部分，只会介绍控制器是如何学习一个NASNet网络块的。

## 1. NASNet详解

### 1.1 NASNet 控制器

在NASNet中，完整的网络的结构还是需要手动设计的，NASNet学习的是完整网络中被堆叠、被重复使用的网络单元。为了便于将网络迁移到不同的数据集上，我们需要学习两种类型的网络块：（1）_Normal Cell_：输出Feature Map和输入Feature Map的尺寸相同；（2）_Reduction Cell_：输出Feature Map对输入Feature Map进行了一次降采样，在Reduction Cell中，对使用Input Feature作为输入的操作（卷积或者池化）会默认步长为2。

NASNet的控制器的结构如图1所示，每个网络单元由$$B$$的网络块（block）组成，在实验中$$B=5$$。每个块的具体形式如图1右侧部分，每个块有并行的两个卷积组成，它们会由控制器决定选择哪些Feature Map作为输入（灰色部分）以及使用哪些运算（黄色部分）来计算输入的Feature Map。最后它们会由控制器决定如何合并这两个Feature Map。

<figure>
<img src="/assets/NASNet_1.png" alt="图1：NASNet控制器结构示意图"/>
<figcaption>图1：NASNet控制器结构示意图</figcaption>
</figure>


更精确的讲，NASNet网络单元的计算分为5步：

1. 从第$$h_{i-1}$$个Feature Map或者第$$h_i$$个Feature Map或者之前已经生成的网络块中选择一个Feature Map作为hidden layer A的输入，图2是学习到的网络单元，从中可以看到三种不同输入Feature Map的情况；
2. 采用和1类似的方法为Hidden Layer B选择一个输入；
3. 为1的Feature Map选择一个运算；
4. 为2的Feature Map选择一个元素；
5. 选择一个合并3，4得到的Feature Map的运算。

<figure>
<img src="/assets/NASNet_2.png" alt="图2：NASNet生成的CNN单元。(左)：Normal Cell，（右）Reduction Cell"/>
<figcaption>图2：NASNet生成的CNN单元。(左)：Normal Cell，（右）Reduction Cell</figcaption>
</figure>


在3，4中我们可以选择的操作有：

* 直接映射
* $$1\times1$$卷积；
* $$3\times3$$卷积；
* $$3\times3$$深度可分离卷积；
* $$3\times3$$空洞卷积；
* $$3\times3$$平均池化；
* $$3\times3$$最大池化；
* $$1\times3$$卷积 + $$3\times1$$卷积；
* $$5\times5$$深度可分离卷积；
* $$5\times5$$最大池化；
* $$7\times7$$深度可分离卷积；
* $$7\times7$$最大池化；
* $$1\times7$$卷积 + $$7\times1$$卷积；

在5中可以选择的合并操作有（1）单位加；（2）拼接。

最后所有生成的Feature Map通过拼接操作合成一个完整的Feature Map。

为了能让控制器同时预测Normal Cell和Reduction Cell，RNN会有$$2\times5\times B$$个输出，其中前$$5\times B$$个输出预测Normal Cell的$$B$$个块（如图1每个块有5个输出），后$$5\times B$$个输出预测Reduction Cell的$$B$$个块。RNN使用的是单层100个隐层节点的[LSTM](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/about-long-short-term-memory.html)。

### 1.2 NASNet的强化学习

NASNet的强化学习思路和NAS相同，有几个技术细节这里说明一下：

1. NASNet进行迁移学习时使用的优化策略是Proximal Policy Optimization（PPO）{{"zoph2018learning"|cite}}；
2. 作者尝试了均匀分布的搜索策略，效果略差于策略搜索。

### 1.3 Scheduled Drop Path

在优化类似于Inception的多分支结构时，以一定概率随机丢弃掉部分分支是避免过拟合的一种非常有效的策略，例如DropPath\[4\]。但是DropPath对NASNet不是非常有效。在NASNet的Scheduled Drop Path中，丢弃的概率会随着训练时间的增加线性增加。这么做的动机很好理解：训练的次数越多，模型越容易过拟合，DropPath的避免过拟合的作用才能发挥的越有效。

### 1.4 其它超参

在NASNet中，强化学习的搜索空间大大减小，很多超参数已经由算法写死或者人为调整。这里介绍一下NASNet需要人为设定的超参数。

1. 激活函数统一使用ReLU，实验结果表明ELU nonlinearity[5]效果略优于ReLU；
2. 全部使用Valid卷积，padding值由卷积核大小决定；
3. Reduction Cell的Feature Map的数量需要乘以2，Normal Cell数量不变。初始数量人为设定，一般来说数量越多，计算越慢，效果越好；
4. Normal Cell的重复次数（图3中的$$N$$）人为设定；
5. 深度可分离卷积在深度卷积和单位卷积中间不使用BN或ReLU;
6. 使用深度可分离卷积时，该算法执行两次；
7. 所有卷积遵循ReLU->卷积->BN的计算顺序；
8. 为了保持Feature Map的数量的一致性，必要的时候添加$$1\times1$$卷积。

堆叠Cell得到的CIFAR_10和ImageNet的实验结果如图3所示。

<figure>
<img src="/assets/NASNet_3.png" alt="图3：NASNet的CIFAR10和ImageNet的网络结构"/>
<figcaption>图3：NASNet的CIFAR10和ImageNet的网络结构</figcaption>
</figure>

## 2. 总结

NASNet最大的贡献是解决了NAS无法应用到大数据集上的问题，它使用的策略是先在小数据集上学一个网络单元，然后在大数据集上堆叠更多的单元的形式来完成模型迁移的。

NASNet已经不再是一个dataset interest的网络了，因为其中大量的参数都是人为设定的，网络的搜索空间更倾向于**密集连接的方式**。这种人为设定参数的一个正面影响就是减小了强化学习的搜索空间，从而提高运算速度，在相同的硬件环境下，NASNet的速度要比NAS快7倍。

NASNet的网络单元本质上是一个更复杂的Inception，可以通过堆叠
网络单元的形式将其迁移到任意分类任务，乃至任意类型的任务中。论文中使用NASNet进行的物体检测也要优于其它网络。

本文使用CIFAR-10得到的网络单元其实并不是非常具有代表性，理想的数据集应该是ImageNet。但是现在由于硬件的计算能力受限，无法在ImageNet上完成网络单元的学习，随着硬件性能提升，基于ImageNet的NASNet一定会出现。或者我们也可以期待某个土豪团队多费电电费帮我们训练出这样一个架构来。

## Reference

\[1\] Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition\[J\]. arXiv preprint arXiv:1707.07012, 2017, 2\(6\).

\[2\] Zoph B, Le Q V. Neural architecture search with reinforcement learning\[J\]. arXiv preprint arXiv:1611.01578, 2016.

\[3\] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

\[4\] G. Larsson, M. Maire, and G. Shakhnarovich. Fractalnet: Ultra-deep neural networks without residuals. arXiv preprint arXiv:1605.07648, 2016.

[5] D.-A. Clevert, T. Unterthiner, and S. Hochreiter. Fast and accurate deep network learning by exponential linear units (elus). In International Conference on Learning Representa-tions, 2016.

