# Layer Normalization

## 前言

在上一篇的文章中我们介绍了BN\[2\]的计算方法并且讲解了BN如何应用在MLP以及CNN中如何使用BN。在文章的最后，我们指出BN并不适用于RNN等动态网络和batchsize较小的时候效果不好。Layer Normalization（LN）\[1\]的提出有效的解决BN的这两个问题。LN和BN不同点是归一化的维度是互相垂直的，如图1所示。在图1中$$N$$表示样本轴，C表示通道轴，F是每个通道的特征数量。BN如右侧所示，它是取不同样本的同一个通道的特征做归一化；LN则是如左侧所示，它取的是同一个样本的不同通道做归一化。

![](/assets/LN_1.png)

## 1. BN的问题

### 1.1 BN与Batch Size

如图1右侧部分，BN是按照样本数计算归一化统计量的，当样本数很少时，比如说只有4个。这四个样本的均值和方差便不能反映全局的统计分布息，所以基于少量样本的BN的效果会变得很差。在一些场景中，比如说硬件资源受限，在线学习等场景，BN是非常不适用的。

### 1.2 BN与RNN

RNN可以展开成一个隐藏层共享参数的MLP，随着时间片的增多，展开后的MLP的层数也在增多，最终层数由输入数据的时间片的数量决定，所以RNN是一个动态的网络。

在一个batch中，通常各个样本的长度都是不同的，当统计到比较靠后的时间片时，例如图2中$$t>4$$时，这时只有一个样本还有数据，基于这个样本的统计信息不能反映全局分布，所以这时BN的效果并不好。

另外如果在测试时我们遇到了长度大于任何一个训练样本的测试样本，我们无法找到保存的归一化统计量，所以BN无法运行。

![](/assets/LN_2.png)

## 2. LN详解

### 2.1 MLP中的LN

通过第一节的分析，我们知道BN的两个缺点的产生原因均是因为计算归一化统计量时计算的样本数太少。LN是一个独立于batch size的算法，所以无论样本数多少都不会影响参与LN计算的数据量，从而解决BN的两个问题。LN的做法如图1左侧所示：根据样本的特征数做归一化。

先看MLP中的LN。设$$H$$是一层中隐层节点的数量，$$l$$是MLP的层数，我们可以计算LN的归一化统计量$$\mu$$和$$\sigma$$：

$$
\mu^l = \frac{1}{H}\sum_{i=1}^H a_i^l
\qquad
\sigma^l = \sqrt{\frac{1}{H} \sum_{i=1}^H(a_i^l
 - \mu^l)^2}
$$

注意上面统计量的计算是和样本数量没有关系的，它的数量只取决于隐层节点的数量，所以只要隐层节点的数量足够多，我们就能保证LN的归一化统计量足够具有代表性。通过$$\mu^l$$和$$\sigma^l$$可以得到归一化后的值$$\hat{a}^l$$：

$$
\hat{\mathbf{a}}^l = \frac{\mathbf{a}^l - \mu^l}{\sqrt{{\sigma^l}^2 + \epsilon}}
$$

其中$$\epsilon$$是一个很小的小数，防止除0（论文中忽略了这个参数）。

在LN中我们也需要一组参数来保证归一化操作不会破坏之前的信息，在LN中这组参数叫做增益（gain）$$g$$和偏置（bias）$$b$$（等同于BN中的$$\gamma$$和$$\beta$$）。假设激活函数为$$f$$，最终LN的输出为：

$$
\mathbf{h}^l = f(\mathbf{g}^l \odot \hat{\mathbf{a}}^l + \mathbf{b}^l)
$$

合并公式(2)，(3)并忽略参数$$l$$，我们有：

$$
\mathbf{h} = f(\frac{\mathbf{g}}{\sqrt{{\sigma}^2 + \epsilon}} \odot (\mathbf{a} - \mu)+ \mathbf{b})
$$

### 2.2 RNN中的LN

在RNN中，我们可以非常简单的在每个时间片中使用LN，而且在任何时间片我们都能保证归一化统计量统计的是$$H$$个节点的信息。对于RNN时刻$$t$$时的节点，其输入是$$t-1$$时刻的隐层状态$$h^{t-1}$$和$$t$$时刻的输入数据$$\mathbf{x}_t$$，可以表示为：

$$
\mathbf{a}^t = W_{hh}h^{t-1} + W_{xh}\mathbf{x}^t
$$

接着我们便可以在$$\mathbf{a}^t$$上采取和1.1节中完全相同的归一化过程：

$$
\mathbf{h}^t = f(\frac{\mathbf{g}}{\sqrt{{\sigma^t}^2 + \epsilon}} \odot (\mathbf{a}^t - \mu^t)+ \mathbf{b})
\qquad
\mu^t = \frac{1}{H}\sum_{i=1}^H a^t_i
\qquad
\sigma^t = \sqrt{\frac{1}{H} \sum_{i=1}^H(a_i^t
 - \mu^t)^2}
$$

### 2.3 LN与ICS和损失平面平滑

LN能减轻ICS吗？当然可以，至少LN将每个训练样本都归一化到了相同的分布上。而在BN的文章中介绍过几乎所有的归一化方法都能起到平滑损失平面的作用。所以从原理上讲，LN能加速收敛速度的。

## 3. 对照实验

这里我们设置了一组对照试验来对比普通网络，BN以及LN在MLP和RNN上的表现。这里使用的框架是Keras，代码见：

### 3.1 MNIST上的实验




## Reference

\[1\] Ba J L, Kiros J R, Hinton G E. Layer normalization\[J\]. arXiv preprint arXiv:1607.06450, 2016.

\[2\] Ioffe S, Szegedy C. Batch normalization: Accelerating deep network training by reducing internal covariate shift\[J\]. arXiv preprint arXiv:1502.03167, 2015.

