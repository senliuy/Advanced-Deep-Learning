# Bidirectional Scene Text Recognition with a Single Decoder

## 前言

Transformer被提出以来，几乎刷榜了NLP的所有任务，自然而然的大伙会想到使用Transformer来做场景文字的识别。而Transformer的特点使其具有了识别弧形文字，二维文字的天然优势。这篇文章提出的Bi-STR算法便是在卷积特征之后加入了Transformer同时作为编码器和解码器，同时加入了位置编码和方向编码来作为额外的特征。Bi-STR大幅刷新了ASTER的识别准确率，尤其是长文本的识别准确率。

## 1. Bi-STR算法详解

![](/assets/Bi-STR_1.png)

Bi-STR的网络结构如图1所示，从图1左侧的高层架构我们可以看出它是由3个主要部分组成：

1. 残差网络：用作图像的像素编码，用于计算其Feature Map;
2. 由$$N$$个编码层组成的编码器；
3. $$N$$个解码层组成的解码器，解码器有两个输出，分别是从左向右（ltr）以及从右想左（rtl）。

### 1.1 ResNet

ResNet被广泛的用于文字识别的骨干网络，这里采用了一个45层的残差网络。通过这个网络，我们可以得到输入图像的特征表示，这里表示为$$\mathcal{Q} \in \mathbb{R} ^ {W\times C \times H}$$，或者表示为长度为$$w$$的特征序列，即$$\mathbf{v}_1, ..., \mathbf{v}_W$$, 其中 $$\mathbf{v}_i \in \mathbb{R} ^ {C\times H}$$。

### 1.2 编码层

从图1的中间部分我们可以看出，ResNet得到的特征层$$\mathcal{Q}$$加上位置编码信息直接给到由自注意力机制组成的Transformer编码层，这里使用的是多头的Self-Attention。关于Transformer的详细讲解，可以看我的另外一篇文章，这里只对网络流程做一下梳理。

 1. 使用不同的3个特征矩阵乘以图像特征$$\mathcal{Q}$$，我们得到3个不同的向量，他们分别是Query向量（$$\mathbf{Q}$$），Key向量（$$\mathbf{K}$$）和Value向量（$$\mathbf{V}$$）。
 2. 根据$$\mathbf{Q}, \mathbf{K}, \mathbf{V}$$我们可以得到Self-Attention的矩阵表示：

$$
Attention(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}.
$$

 3. 多头自注意力机制

Multi-Head Self Attention是由多个Single-Head Self拼接而成的，表示为

$$
head_{i} = \text{Attention}(\mathbf{QW}^Q_i, \mathbf{KW}^K_i, \mathbf{VW}^V_i)
$$

多头Self-Attention表示为

$$
MultiHeadSelfAttention = Concat(head_1, ..., head_h)\mathbf{W}^O
$$

其中$$\mathbf{W}^Q_i \in \mathbb{R} ^ {d_\text{model} \times d_k}, \mathbf{W}^K_i \in \mathbb{R} ^ {d_\text{model} \times d_k}, \mathbf{W}^V_i \in \mathbb{R} ^ {d_\text{model} \times d_v}$$ 以及$$\mathbf{W} ^ O \in \mathbb{R} ^ {hd_v \times d_\text{model}}$$是参数矩阵。

 4. 与Feature Map一起提供给编码器的位置向量，它的编码方式和Transformer论文提供的方式相同。
 
### 1.3 解码层

图1右侧的是Bi-STR的解码层，它的输入一个是编码器的输出，还有三个分别是方向嵌入，词位置嵌入以及词嵌入。

#### 1.3.1 方向嵌入
 
首先方向嵌入是为了模拟双向RNN结构引入的编码器，作用是告诉模型是从左向右编码还是从右向左编码，关于方向编码具体的实现细节，论文中没有给出。

#### 1.3.2 
