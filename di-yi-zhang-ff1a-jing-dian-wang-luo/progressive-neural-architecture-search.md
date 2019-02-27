# Progressive Neural Architecture Search

tags: NAS, NASNet, PNASNet, AutoML

## 前言

在[NAS](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/neural-architecture-search-with-reinforecement-learning.html){{"zoph2016neural"|cite}}和[NASNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/learning-transferable-architectures-for-scalable-image-recognition.html){{"zoph2018learning"|cite}}中我们介绍了如何使用强化学习训练卷积网络的超参。NAS是该系列的第一篇，提出了使用强化学习训练一个控制器（RNN），该控制器的输出是卷积网络的超参，可以生成一个完整的卷积网络。NASNet提出学习网络的一个单元比直接整个网络效率更高且更容易迁移到其它数据集，并在ImageNet上取得了当时最优的效果。

本文是约翰霍普金斯在读博士刘晨曦在Google实习的一篇文章，基于NASNet提出了PNASNet{{"liu2018progressive"|cite}}，其训练时间降为NASNet的1/8并且取得了比ImageNet上更优的效果。其主要的优化策略为：

1. 更小的搜索空间；
2. Sequential model-based optimization\(SMBO\)：一种启发式搜索的策略，训练的模型从简单到复杂，从剪枝的空间中进行搜索；
3. 代理函数：使用代理函数预测模型的精度，省去了耗时的训练过程。

在阅读本文之前，确保你已经读懂了[NAS](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/neural-architecture-search-with-reinforecement-learning.html)和[NASNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/learning-transferable-architectures-for-scalable-image-recognition.html)两篇文章。

## 1. PNASNet详解

### 1.1 更小的搜索空间

回顾NASNet的控制器策略，它是一个有$$2\times B \times 5$$个输出的LSTM，其中2表示分别学习Normal Cell和Reduction Cell。$$B$$表示每个网络单元有$$B$$个网络块。$$5$$表示网络块有5个需要学习的超参，记做$$(I_1, I_2, O_1, O_2, C)$$。$$I_1, I_2 \in \mathcal{I}_b$$用于预测网络块两个隐层状态的输入（Input），它会从之前一个，之前两个，或者已经计算的网络块中选择一个。$$O_1, O_2 \in \mathcal{O}$$用于预测对两个隐层状态的输入的操作（Operation，共有13个，具体见NASNet。$$C\in \mathcal{C}$$表示$$O_1, O_2$$的合并方式，有单位加和合并两种操作。因此它的搜索空间的大小为：


$$
(2^2\times13^2 \times 3^2\times13^2 \times4^2\times13^2 \times5^2\times13^2 \times6^2\times13^2 \times 2)^2 \approx 2.0\times 10^{34}
$$


PNASNet的控制器的运作方式和NASNet类似，但也有几点不同。

**只有Normal Cell**：PNASNet只学习了Normal Cell，是否进行降采样用户自己设置。当使用降采样时，它使用和Normal Cell完全相同的架构，只是要把Feature Map的数量乘2。这种操作使控制器的输出节点数变为$$B \times 5$$。

**更小的**$$\mathcal{O}$$：在观察NASNet的实验结果是，我们发现有5个操作是从未被使用过的，因此我们将它们从搜索空间中删去，保留的操作剩下了8个：

* 直接映射
* $$3\times3$$深度可分离卷积；
* $$3\times3$$空洞卷积；
* $$3\times3$$平均池化；
* $$3\times3$$最大池化；
* $$5\times5$$深度可分离卷积；
* $$7\times7$$深度可分离卷积；
* $$1\times7$$卷积 + $$7\times1$$卷积；

**合并**$$\mathcal{C}$$：通过观察NASNet的实验结果，作者发现拼接操作也从未被使用，因此我们也可以将这种情况从搜索空间中删掉。因此PASNet的超参数是四个值的集合$$(I_1, I_2, O_1, O_2)$$。

因此PNASNet的搜索空间的大小是：


$$
2^2\times8^2 \times 3^2\times8^2 \times4^2\times8^2 \times5^2\times8^2 \times6^2\times8^2 \approx 5.6\times 10^{14}
$$


我们可以写一些规则来排除掉两个隐层状态的对称的情况，但即使排除掉对称的情况后，NASNet的搜索空间的大小仍然为$$10^{28}$$，PNASNet的搜索空间仍然为$$10^{12}$$。这两个值的具体计算比较复杂，且和本文主要要讲解的内容关系不大，感兴趣的读者自行推算。

### 1.2 SMBO

尽管已经将优化搜索空间优化到了$$10^{12}$$的数量级，但是这个规模依然十分庞大，在其中进行搜索依旧非常耗时。这篇文章的核心便是提出了Sequential model-based optimization\(SMBO\)，它在模型的搜索空间中进行优化时会剪枝掉一些分支从而缩小模型的搜索空间。具体的讲SMBO的搜索是一种递进（Progressive）的形式，它的网络块的数目会从1个开始逐渐增加到$$B$$个。

当网络块数$$b=1$$时，它的搜索空间为$$2^2\times8^2 = 256$$（不考虑对称情况），也就是可以生成256个不同的网络块（$$\mathcal{B}_1$$），计构成网络的超参数为$$\mathcal{S}_1$$。这个搜索空间并不大，我们可以枚举出所有情况并训练由它们组成的网络（$$\mathcal{M}_1$$）。接着我们训练所有的$$\mathcal{M}_1$$个网络，接着得到训练后的模型（$$\mathcal{C}_1$$）。通过使用验证集我们可以得到每个模型的精度（$$\mathcal{A}_1$$）。有了网络超参数$$\mathcal{S}_1$$和它们对应的精度$$\mathcal{A}_1$$，我们希望有一个代理函数$$\pi$$能够计算参数（特征）和精度（标签）额关系，这样我们就可以省去非常耗时的模型训练的过程了。代理函数的细节我们会在1.3节详细分析，在这你只需要把它看做从网络超参$$\mathcal{S}_1$$到它对应的精度$$\mathcal{A}_1$$的映射即可。

当网络块数$$b=2$$时，它的搜索空间为$$2^2\times8^2\times3^2\times8^2=147,456$$，它的实际意义是在b=1的基础上再扩展一个网络块，表示为$$\mathcal{S}_2'$$。使用b=1时得到的代理函数$$\pi$$可以非常快速的为每个扩展模型非常快速的预测一个精度，表示为$$\mathcal{A}_2'$$，这里可以称作代理精度。代理精度并不非常准确，我们需要得到真正的精度，它的作用是为我们剪枝搜索空间。具体的讲，我们会根据代理精度选取top-K个扩展模型（$$\mathcal{S}_2$$），一般K的值远小于搜索空间。仿照上段的过程，我们会依次使用$$\mathcal{S}_2$$搭建卷积网络$$\mathcal{C}_2$$，使用$$\mathcal{C}_2$$得到模型在验证集上的精度$$\mathcal{A}_2$$，最后我们使用得到的$$(\mathcal{S}_2,\mathcal{A}_2)$$更新代理函数$$\pi$$。

仿照上一段的过程，我们可以使用$$b\geq2$$更新的代理函数$$\pi$$得到$$b+1$$的top-K的扩展结构并更新得到新的代理函数$$\pi$$。以此类推直到$$b=B$$，如Algorithm1和图1。

![](/assets/PNASNet_a1.png)

<figure>
<img src="/assets/PNASNet_1.png" alt="图1：SMBO流程图（B=3）"/>
<figcaption>图1：SMBO流程图（B=3）</figcaption>
</figure>

SMBO像极了我们在[CTC](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/connectionist-temporal-classification-labelling-unsegmented-sequence-data-with-recurrent-neural-networks.html)中介绍的宽度为K的Beam Search。

### 1.3 代理函数

1.2节中介绍SMBO时，代理函数$$\pi$$在其中发挥了至关重要的作用，从上面的过程中我们知道代理函数必须有下面3条特征：

1. _处理变长数据_：在SMBO中我们会使用$$b$$的数据更新模型并在$$b+1$$的扩展模型上预测精度；
2. _正相关_：因为代理精度$$\mathcal{A}_b'$$的作用是用来选取top-K个扩展模型，所以其预测的精度不一定准确，但选取的top-K个扩展模型要尽可能的准确。所以保证代理函数预测的精度至少和实际精度是正相关的；
3. _样本有效_：在SMBO中我们的用于训练模型的样本数量是K，为了效率K的值一般会很小，所以我们希望代理函数在小数据集上也能表现出好的结果。

处理变长数据的一个非常经典的模型便是RNN，因为它可以将输入数据按照网络块切分成时间片。具体的讲，LSTM的输入是尺寸为$$4\times b$$超参数$$\mathcal{S}_b$$，其中4指的是超参数的四个元素$$(I_1, I_2, O_1, O_2)$$。输入LSTM之前，$$(I_1,I_2)$$经过one-hot编码后会通过一个共享的嵌入层进行编码，$$(O_1,O_2)$$也会先one-hot编码再通过另外一个共享的嵌入层进行编码。最后的隐层节点经过一个激活函数为sigmoid的全连接得到最后的预测精度。损失函数使用L1损失。

作者也采用了一组MLP作为对照试验，编码方式是将每个超参数转换成一个D维的特征向量，四个超参数拼接之后会得到一个4-D的特征向量。如果网络块数b&gt;1，我们则取这b个特征向量的均值作为输入，这样不管几个网络块，MLP的输入的数据维度都是4-D。损失函数同样使用L1损失。

由于样本数非常少，作者使用的是五个模型组成的模型集成。

为了验证代理函数在边长数据上的表示能力，作者在LSTM和MLP上做了一组排序相关性的对照试验。分析出的结论是在相同网络块下，LSTM优于MLP，但是在预测网络块多一个的模型上MLP要优于LSTM。原因可能是LSTM过拟合了。

## 2. PNASNet的实验结果

### 2.1 增进式的结构

根据1.2节介绍的SMBO的搜索过程，PNASNet可以非常容易得得出网络块数小于等于$$B$$的所有模型，其结果如图2所示。

<figure>
<img src="/assets/PNASNet_2.png" alt="图2：PNASNet得出的B=1,2,3,4,5的几个网络单元，推荐使用B=5"/>
<figcaption>图2：PNASNet得出的B=1,2,3,4,5的几个网络单元，推荐使用B=5</figcaption>
</figure>

作者也尝试了$$B>5$$的情况，发现这时候模型的精度会下降，推测原因是因为搜索空间过去庞大了。

### 2.2 迁移到ImageNet

NAS中提倡学习dataset interest的网络结构，但是NASNet和PNASNet在CIFAR-10上学习到的结构迁移到ImageNet上也可以取得非常好的效果。作者通过一组不同网络单元在CIFAR-10和ImageNet上的实验验证了CIFAR-10和ImageNet在网络结构上的强相关性，实验结果见图3。

<figure>
<img src="/assets/PNASNet_3.png" alt="图3：CIFAR10和ImageNet对网络单元的强相关性"/>
<figcaption>图3：CIFAR10和ImageNet对网络单元的强相关性</figcaption>
</figure>

## 3. 总结

这篇PNASNet是之前NAS和NASNet的第三个系列，其着重点放在了优化强化学习的搜索空间的优化，几个优化的策略也是以此为目的。更少的参数是为了减小搜索空间的大小，SMBO是为了使用剪枝策略来优化强化学习探索的区域大小，而代理函数则提供了比随机采样更有效的采样策略。

本文使用的剪枝搜索和策略函数是强化学习最常见的技巧，例如AlphaGo。作为一个强化学习届的小白，对此无法给下一个特别准确地总结，只能期待大牛们努力推出更高效，精度更高，以及能够以更小代价得出模型的方法。

