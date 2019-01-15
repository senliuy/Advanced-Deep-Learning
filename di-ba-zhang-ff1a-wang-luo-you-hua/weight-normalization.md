# Weight Normalization

## 前言

之前介绍的BN和LN都是在数据的层面上做的归一化，而这篇文章介绍的Weight Normalization（WN\)是在权值的维度上做的归一化。WN的做法是将权值向量$$W$$在其欧氏范数和其方向上解耦成了参数向量$$\mathbf{v}$$和参数标量$$g$$后使用SGD分别优化这两个参数。

WN也是和样本量无关的，所以可以应用在batchsize较小以及RNN等动态网络中；另外BN使用的基于mini-batch的归一化统计量代替全局统计量，相当于在梯度计算中引入了噪声。而WN则没有这个问题，所以在生成模型，强化学习等噪声敏感的环境中WN的效果也要优于BN。

WN没有一如额外参数，这样更节约显存。同时WN的计算效率也要优于要计算归一化统计量的BN。

## 1. WN详解

神经网络的一个节点计算可以表示为：


$$
y = \phi(\mathbf{w}\cdot\mathbf{x}+b)
$$


其中$$\mathbf{w}$$是一个$$k$$-维的特征向量，$$y$$是该神经节点的输出，所以是一个标量。在得到损失值后，我们会根据损失函数的值使用SGD等优化策略更新$$\mathbf{w}$$和$$b$$。WN提出的归一化策略是将$$\mathbf{w}$$分解为一个参数向量$$\mathbf{v}$$和一个参数标量$$g$$，分解方法为


$$
\mathbf{w} = \frac{g}{||\mathbf{v}||} \mathbf{v}
$$


上式中$$||\mathbf{v}||$$表示$$\mathbf{v}$$的欧氏范数。当$$\mathbf{v}=\mathbf{w}$$且$$g = ||\mathbf{w}||$$时，WN还原为普通的计算方法，所以WN的网络容量是要大于普通神经网络的。

当我们将$$g$$固定为$$||\mathbf{w}||$$时，我们只优化$$\mathbf{v}$$，这时候相当于只优化$$\mathbf{w}$$的方向而保留其范数。当$$\mathbf{v}$$固定为$$\mathbf{w}$$时，这时候相当于只优化$$\mathbf{w}$$的范数，而保留其方向，这样为我们优化权值提供了更多可以选择的空间，且解耦方向与范数的策略也能加速其收敛。

在优化$$g$$时，我们一般通过优化$$g$$的log级参数$$s$$来完成，即$$g = e^s$$。

$$\mathbf{v}$$和$$g$$的更新值可以通过SGD计算得到：


$$
\nabla_g L = \frac{\nabla_{\mathbf w}L \cdot \mathbf{v}}{||\mathbf{v}||}
\qquad
\nabla_{\mathbf{v}} L = \frac{g}{||\mathbf{v}||} \nabla_{\mathbf{w}} L - \frac{g\nabla_g L}{||\mathbf{v}||^2} \mathbf{v}
$$


其中$$L$$为损失函数，$$\nabla_{\mathbf{w}}L$$为$$\mathbf{w}$$在$$L$$下的梯度值，上式也可以写作：


$$
\nabla_{\mathbf{v}} L = \frac{g}{||\mathbf{v}||} M_{\mathbf{w}} \nabla_{\mathbf w}L
\quad
\text{with}
\quad
M_{\mathbf{w}} = I - \frac{\mathbf{w}\mathbf{w}'}{||\mathbf{w}||^2}
$$


推导方式如下：

![](/assets/WN_1.png)

倒数第二步的推导是因为$$\mathbf{v}$$是$$\mathbf{w}$$的方向向量。

## Reference

[1] Salimans T, Kingma D P. Weight normalization: A simple reparameterization to accelerate training of deep neural networks[C]//Advances in Neural Information Processing Systems. 2016: 901-909.


