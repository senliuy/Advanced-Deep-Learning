# Neural Architecture Search with Reinforecement Learning

tags: Reinforcement Learning, CNN, RNN

## 前言

CNN和RNN是目前主流的CNN框架，这些网络均是由人为手动设计，然而这些设计是非常困难以及依靠经验的。作者在这篇文章中提出了使用强化学习（Reinforcement Learning）学习一个CNN（后面简称NAS-CNN）或者一个RNN cell（后面简称NAS-RNN），并通过最大化网络在验证集上的精度期望来优化网络，在CIFAR-10数据集上，NAS-CNN的错误率已经逼近当时最好的[DenseNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/densely-connected-convolutional-networks.html)\[2\]，在TreeBank数据集上，NAS-RNN要优于LSTM。

## 1. 背景介绍

文章提出了Neural Architecture Search（NAS），算法的主要目的是使用强化学习寻找最优网络，包括一个图像分类网络的卷积部分（表示层）和RNN的一个类似于LSTM的cell。由于现在的神经网络一般采用堆叠block的方式搭建而成，这种堆叠的超参数可以通过一个序列来表示。而这种序列的表示方式正是RNN所擅长的工作。

所以，NAS会使用一个RNN构成的控制器（controller）以概率$$p$$随机采样一个网络结构$$A$$，接着在CIFAR-10上训练这个网络并得到其在验证集上的精度$$R$$，然后在使用$$R$$更新控制器的参数，如此循环执行直到模型收敛，如图1所示。

<figure>
<img src="/assets/NAS_1.png" alt="图1：NAS的算法流程图"/>
<figcaption>图1：NAS的算法流程图</figcaption>
</figure>



## 2. NAS详细介绍

### 2.1 NAS-CNN

首先我们考虑最简单的CNN，即只有卷积层构成。那么这种类型的网络是很容易用控制器来表示的。即将控制器分成$$N$$段，每一段由若干个输出，每个输出表示CNN的一个超参数，例如Filter的高，Filter的宽，横向步长，纵向步长以及Filter的数量，如图2所示。

<figure>
<img src="/assets/NAS_2.png" alt="图2：NAS-CNN的控制器结构图"/>
<figcaption>图2：NAS-CNN的控制器结构图</figcaption>
</figure>


了解了控制器的结构以及控制器如何生成一个卷积网络，唯一剩下的也是最终要的便是如何更新控制器的参数$$\theta_c$$。

控制器每生成一个网络可以看做一个action，记做$$a_{1:T}$$，其中$$T$$是要预测的超参数的数量。当模型收敛时其在验证集上的精度是$$R$$。我们使用$$R$$来作为强化学习的奖励信号，也就是说通过调整参数$$\theta_c$$来最大化R的期望，表示为：


$$
J(\theta_c) = E_{P(a_{1:T};\theta_c)}[R]
$$


由于$$R$$是不可导的，所以我们需要一种可以更新$$\theta_c$$的策略，NAS中采用的是Williams等人提出的REINFORCE rule\[3\]：


$$
\nabla_{\theta_c} J(\theta_c) = \sum_{t=1}^T E_{P(a_{1:T};\theta_c)}[\nabla_{\theta_c}logP(a_t|a_{(t-1):1};\theta_c)R]
$$


上式近似等价于：


$$
\frac{1}{m}\sum_{k=1}^m \sum_{t=1}^T \nabla_{\theta_c} logP(a_t|a_{(t-1):1};\theta_c)R_k
$$


其中$$m$$是每个batch中网络的数量。

上式是梯度的无偏估计，但是往往方差比较大，为了减小方差算法中使用的是下面的更新值：


$$
\frac{1}{m}\sum_{k=1}^m \sum_{t=1}^T \nabla_{\theta_c} logP(a_t|a_{(t-1):1};\theta_c)(R_k-b)
$$


基线b是以前架构精度的指数移动平均值。

上面得到的控制器的搜索空间是不包含跳跃连接（skip connection）的，所以不能产生类似于[ResNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)或者[Inception](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/going-deeper-with-convolutions.html)之类的网络。NAS-CNN是通过在上面的控制器中添加[注意力机制](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/neural-machine-translation-by-jointly-learning-to-align-and-translate.html)\[4\]来添加跳跃连接的，如图3。

<figure>
<img src="/assets/NAS_3.png" alt="图3：NAS-CNN中加入跳跃连接的控制器结构图"/>
<figcaption>图3：NAS-CNN中加入跳跃连接的控制器结构图</figcaption>
</figure>


在第$$N$$层，我们添加$$N-1$$个anchor来确定是否需要在该层和之前的某一层添加跳跃连接，这个anchor是通过两层的隐节点状态和sigmoid激活函数来完成判断的，具体的讲：


$$
P(\text{Layer j is an input to layer i}) = \text{sigmoid}(v^T \text{tanh}(W_{prev} * h_j + W_{curr} * h_i))
$$


其中$$h_j$$是第$$j$$层隐层节点的状态，$$j\in[0,N-1]$$。$$W_{prev}$$，$$W_{curr}$$和$$v^T$$是可学习的参数，跳跃连接的添加并不会影响更新策略。

由于添加了跳跃连接，而由训练得到的参数可能会产生许多问题，例如某个层和其它所有层都没有产生连接等等，所以有几个问题我们需要注意：

1. 如果一个层和其之前的所有层都没有跳跃连接，那么这层将作为输入层；
2. 如果一个层和其之后的所有层都没有跳跃连接，那么这层将作为输出层，并和所有输出层拼接之后作为分类器的输入；
3. 如果输入层拼接了多个尺寸的输入，则通过将小尺寸输入加值为0的padding的方式进行尺寸统一。

除了卷积和跳跃连接，例如池化，BN，Dropout等策略也可以通过相同的方式添加到控制器中，只不过这时候需要引入更多的策略相关参数了。

经过训练之后，在CIFAR-10上得到的卷积网络如图4所示。

<figure>
<img src="/assets/NAS_4.png" alt="图4：NAS-CNN生成的密集连接的网络结构"/>
<figcaption>图4：NAS-CNN生成的密集连接的网络结构</figcaption>
</figure>


从图4我们可以发现NAS-CNN和DenseNet有很多相通的地方：

1. 都是密集连接；
2. Feature Map的个数都比较少；
3. Feature Map之间都是采用拼接的方式进行连接。

在生成NAS-CNN的实验中，使用的是CIFAR-10数据集。网络中加入了BN和跳跃连接。卷积核的高的范围是$$[1,3,5,7]$$，宽的范围也是$$[1,3,5,7]$$，个数的范围是$$[24,36,48,64]$$。步长分为固定为1和在$$[1,2,3]$$中两种情况。控制器使用的是含有35个隐层节点的LSTM。

### 2.2 NAS-RNN

在这篇文章中，作者采用强化学习的方法同样生成了RNN中类似于[LSTM](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/about-long-short-term-memory.html)或者[GRU](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/learning-phrase-representations-using-rnn-encoder-decoder-for-statistical-machine-translation.html)的一个Cell。控制器的参数更新方法和1.2节类似，这里我们主要介绍如何使用一个RNN控制器来描述一个RNN cell。

传统RNN的的输入是$$x_t$$和$$h_{t-1}$$，输出是$$h_t$$，计算方式是$$h_t = tanh(W_1 x_t + W_2 h_{t-1})$$。LSTM的输入是$$x_t$$，$$h_{t-1}$$以及单元状态$$c_t-1$$，输出是$$h_t$$和$$c_t$$，LSTM的处理可以看做一个将$$x_t$$，$$h_{t-1}$$和$$c_t-1$$作为叶子节点的树结构，如图5所示。

<figure>
<img src="/assets/NAS_5.png" alt="图5：LSTM的计算图"/>
<figcaption>图5：LSTM的计算图</figcaption>
</figure>

和LSTM一样，NAS-RNN也需要输入一个$$c_{t-1}$$并输出一个$$c_t$$，并在控制器的最后两个单元中控制如何使用$$c_{t-1}$$以及如何计算$$c_t$$。

如图6所示，在这个树结构中有两个叶子节点和一个中间节点，这种两个叶子节点的情况简称为base2，而图4的LSTM则是base4。叶子节点的索引是0，1，中间节点的索引是2，如图6左侧部分。也就是说控制器需要预测3个block，每个block包含一个操作（加，点乘等）和一个激活函数（ReLU，sigmoid，tanh等）。在3个block之后接的是一个Cell inject，用于控制$$c_{t-1}$$的使用，最后是一个Cell indices，确定哪些树用于计算$$c_t$$。

<figure>
<img src="/assets/NAS_6.png" alt="图6：NAS-RNN的控制器生成RNN节点示例图"/>
<figcaption>图6：NAS-RNN的控制器生成RNN节点示例图</figcaption>
</figure>

详细分析一下图6：

1. 控制器为索引为0的树预测的操作和激活函数分别是_Add_和_tanh_，意味着$$a_0 = tanh(W_1 * x_t + W_2 * h_{t-1})$$；
2. 控制器为索引为1的树预测的操作和激活函数分别是_ElemMult_和_ReLU_，意味着$$a_1 = ReLU((W_3 * x_t)\odot(W_4*h_{t-1}))$$；
3. 控制器为Cell Indices的第二个元素的预测值为0，Cell Inject的预测值是_add_和_ReLU_，意味着$$a_0$$值需要更新为$$a_0^{new}=ReLU(a_0 + c_{t-1})$$，注意这里不需要额外的参数。
4. 控制器为索引为2的树预测的操作和激活函数分别是_ElemMult_和_Sigmoid_，意味着$$a_2 = sigmoid(a_0^{new}\odot a_1)$$，因为a\_2是最大的树的索引，所以$$h_t = a_2$$；
5. 控制器为Cell Indices的第一个元素的预测值是1，意思是$$c_t$$要使用索引为1的树在使用激活函数的值，即$$c_t = (W_3*x_t) \odot (W_4*h_{t-1})$$。

上面例子是使用“base 2”的超参作为例子进行讲解的，在实际中使用的是base 8，得到图7两个RNN单元。左侧是不包含_max_和_sin_的搜索空间，右侧是包含_max_和_sin_的搜索空间（控制器并没有选择sin）。

<figure>
<img src="/assets/NAS_7.png" alt="图7：NAS-CNN生辰的网络节点的计算图"/>
<figcaption>图7：NAS-CNN生辰的网络节点的计算图</figcaption>
</figure>

在生成NAS-RNN的实验中，使用的是Penn TreeBank数据集。操作的范围是\[_add_, _elem\_mult_\]，激活函数的范围是\[_identity_,_tanh_,_sigmoid_,_relu_\]。

## 2. 总结

在如何使用强化学习方面，谷歌一直是领头羊，除了他们具有很多机构难以匹敌的硬件资源之外，更重要的是他们拥有扎实的技术积累。本文开创性的使用了强化学习进行模型结构的探索，提出了NAS-CNN和NAS-RNN两个架构，两个算法的共同点都是使用一个RNN作为控制器来描述生成的网络架构，并使用生成架构在验证集上的表现并结合强化学习算法来训练控制器的参数。

本文的创新性可以打满分，不止是其算法足够新颖，更重要的是他们开辟的使用强化学习来学习网络架构可能在未来几年引网络模型自动生成的方向，尤其是在硬件资源不再那么昂贵的时候。文章的探讨还比较基础，留下了大量的待开发空间为科研工作者所探索，期待未来几年出现更高效，更精确的模型的提出。

## Reference

\[1\] Zoph B, Le Q V. Neural architecture search with reinforcement learning\[J\]. arXiv preprint arXiv:1611.01578, 2016.

\[2\] Huang G, Liu Z, Weinberger K Q, et al. Densely connected convolutional networks\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017, 1\(2\): 3.

\[3\] Williams R J. Simple statistical gradient-following algorithms for connectionist reinforcement learning\[J\]. Machine learning, 1992, 8\(3-4\): 229-256.

\[4\] Bahdanau D, Cho K, Bengio Y. Neural machine translation by jointly learning to align and translate\[J\]. arXiv preprint arXiv:1409.0473, 2014.

