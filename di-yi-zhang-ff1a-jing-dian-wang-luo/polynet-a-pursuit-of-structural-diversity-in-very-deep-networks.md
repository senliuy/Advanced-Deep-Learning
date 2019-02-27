# PolyNet: A Pursuit of Structural Diversity in Very Deep Networks

tags: PolyNet, Inception, ResNet

## 前言

在[Inception v4](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/going-deeper-with-convolutions.html){{"szegedy2017inception"|cite}}中，[Inception](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/going-deeper-with-convolutions.html){{"szegedy2015going"|cite}}和[残差网络](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html){{"he2016deep"|cite}}首次得以共同使用，后面简称IR。这篇文章提出的PolyNet可以看做是IR的进一步扩展，它从多项式的角度推出了更加复杂且效果更好的混合模型，并通过实验得出了这些复杂模型的最优混合形式，命名为_Very Deep PolyNet_{{"zhang2017polynet"|cite}}。

本文试图从**结构多样性**上说明PolyNet的提出动机，但还是没有摆脱通过堆积模型结构（Inception，ResNet）来得到更好效果的牢笼，模型创新性上有所欠缺。其主要贡献是虽然增加网络的深度和宽度能提升性能，但是其收益会很快变少，这时候如果从结构多样性的角度出发优化模型，带来的效益也许会由于增加深度带来的效益，为我们优化网络结构提供了一个新的方向。

## 1. PolyNet详解

### 1.1 结构多样性

当前增大网络表达能力的一个最常见的策略是通过增加网络深度，但是如图1所示，随着网络的深度增加，网络的收益变得越来越小。另一个模型优化的策略是增加网络的宽度，例如增加Feature Map的数量。但是增加网络的宽度是非常不经济的，因为每增加$$k$$个参数，其计算复杂度和显存占用都要增加$$k^2$$。

<figure>
<img src="/assets/PolyNet_1.png" alt="图1：网络深度和精度的关系"/>
<figcaption>图1：网络深度和精度的关系</figcaption>
</figure>

因此作者效仿IR的思想，希望通过更复杂的block结构来获得比增加深度更大的效益，这种策略在真实场景中还是非常有用的，即如何在有限的硬件资源条件下最大化模型的精度。

### 1.2 多项式模型

本文是从多项式的角度推导block结构的。首先一个经典的残差block可以表示为：


$$
(I+F)\cdot \mathbf{x} = \mathbf{x} + F \cdot \mathbf{x}  := \mathbf{x} + F(\mathbf{x} )
$$


其中$$\mathbf{x}$$是输入，$$I$$是单位映射，‘$$+$$’单位加操作，表示在残差网络中$$F$$是两个连续的卷积操作。如果$$F$$是Inception的话，上式便是IR的表达式，如图2所示。

<figure>
<img src="/assets/PolyNet_2.png" alt="图2：（左）经典残差网络，（右）：Inception v4"/>
<figcaption>图2：（左）经典残差网络，（右）：Inception v4</figcaption>
</figure>


下面我们将$$F$$一直看做Inception，然后通过将上式表示为更复杂的多项式的形式来推导出几个更复杂的结构。

* _poly-2_：$$I+F+F^2$$。在这个形式中，网络有三个分支，左侧路径是一个直接映射，中间路径是一个Inception结构，右侧路径是两个连续的Inception，如图3\(a\)所示。在这个网络中，所有Inception的参数是共享的，所以不会引入额外的参数。由于参数共享，我们可以推出它的的等价形式，如图3\(b\)。因为$$I+F+F^2=I+(I+F)F$$，而且这种形式的网络计算量少了1/3。
* _mpoly-2_：$$I+F+GF$$。这个block的结构和图3\(b\)相同，不同之处是两个Inception的参数不共享。其也可以表示为$$I+(I+G)F$$，如图3\(c\)所示。它具有更强的表达能力，但是参数数量也加倍了。
* _2-way_：$$I+F+G$$。即向网络中添加一个额外且参数不共享的残差块，思想和Multi-Residual Networks{{"abdi2016multi"|cite}}相同，如图3\(d\)。

<figure>
<img src="/assets/PolyNet_3.png" alt="图3：PolyNet的几种block"/>
<figcaption>图3：PolyNet的几种block</figcaption>
</figure>


结合上文提出的多项式的思想，几乎可以衍生出无限的网络模型，出于对计算性能的考虑，我们仅考虑下面三个结构:

* _poly-3_：$$I+F+F^2+F^3$$。
* _mploy-3_：$$I+F+GF+HGF$$。
* _3-way_：$$I+F+G+H$$。

如果你看过[DenseNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/densely-connected-convolutional-networks.html){{"huang2017densely"|cite}}的话，你会发现DenseNet本质上也是一个多项式模型，对于一个含有$$n$$个卷积的block块，可以用数学公式表示为：


$$
I \oplus C \oplus C^2 \oplus ... \oplus C^n
$$


其中$$C$$表示一个普通的卷积操作，$$\oplus$$表示特征拼接。

### 1.3 对照试验

如图4所示，我们把IR分成A，B，C共3个阶段，它们处理的Feature Map尺寸分别是$$35\times35$$，$$17\times17$$，$$8\times8$$。如果将A，B，C分别替换为1.2中提出的6个模型，我们可以得到共18个不同的网络结构，给与它们相同的超参数，我们得到的实验结果如图5。

<figure>
<img src="/assets/PolyNet_4.png" alt="图4：Inception的三个阶段和PolyNet提供的替换方式"/>
<figcaption>图4：Inception的三个阶段和PolyNet提供的替换方式</figcaption>
</figure>

<figure>
<img src="/assets/PolyNet_5.png" alt="图5：PolyNet精度图，上面比较的训练时间和精度的关系，下面比较的是参数数量和精度的关系"/>
<figcaption>图5：PolyNet精度图，上面比较的训练时间和精度的关系，下面比较的是参数数量和精度的关系</figcaption>
</figure>

通过图5我们可以抽取到下面几条的重要信息：

1. Stage-B的替换最有效；
2. Stage-B中使用_mpoly-3_最有效，_poly-3_次之，但是_poly-3_的参数数量要少于_mpoly-3_；
3. Stage-A和Stage-C均是使用_3-way_替换最有效，但是引入的参数也最多；
4. 3路Inception的结构一般要优于2路Inception。

另外一种策略是混合的使用3路Inception的混合模型，在论文中使用的是4组_3-way_$$\rightarrow$$_mpoly-3_$$\rightarrow$$_\_poly-3_替换IR中的阶段B，实验结果表明这种替换的效果要优于任何形式的非混合模型。

注意上面的几组实验

### 1.4 Very Deep PolyNet

基于上面提出的几个模型，作者提出了state-of-the-art的Very Deep PolyNet，结构如下：

* stageA：包含10个2-way的基础模块
* stageB：包含10个poly-3，2-way混合的基础模块（即20个基础模型）
* stageC：包含5个poly-3，2-way混合的基础模块（即10个基础模块）

**初始化**：作者发现如果先搭好网络再使用随机初始化的策略非常容易导致模型不稳定，论文使用了两个策略：

1. _initialization by insertion_\(插入初始化\)：策略是先使用迁移学习训练一个IR模型，再通过向其中插入一个Inception块构成_poly-2_；
2. _interleaved_（交叉插入）： 当加倍网络的深度时，将新加入的随机初始化的模型交叉的插入到迁移学习的模型中效果更好。

初始化如图6所示。

<figure>
<img src="/assets/PolyNet_6.png" alt="图6：（左）nitialization by insertion， （右）interleaved"/>
<figcaption>图6：（左）nitialization by insertion， （右）interleaved</figcaption>
</figure>

**随机路径**：收到Dropout的启发，PolyNet在训练的时候会随机丢掉多项式block中的一项或几项，如图7所示。

<figure>
<img src="/assets/PolyNet_7.png" alt="图7：随机路径。（左）：I+F+GF+HGF to I+GF；（右）I+F+G+H to I+G+H"/>
<figcaption>图7：随机路径。（左）：I+F+GF+HGF to I+GF；（右）I+F+G+H to I+G+H</figcaption>
</figure>

**加权路径**：简单版本的多项式结构容易导致模型不稳定，Very Deep PolyNet提出的策略是为Inception部分乘以权值$$\beta$$，例如_2-way_的表达式将由$$I+F+G$$变成$$I+\beta F+\beta G$$，论文给出的$$\beta$$的参考值是0.3。

## 2. 总结

PolyNet从多项式的角度提出了更多由Inception和残差块组合而成的网络结构，模型并没有创新性。最大的优点在于从多项式的角度出发，并且我们从这个角度发现了PolyNet和DenseNet有异曲同工之妙。

论文中最优结构的选择是通过实验得出的，如果能结合数学推导得出前因后果本文将上升到另一个水平。结合上面的Block，作者提出了混合模型Very Deep PolyNet并在ImageNet取得了目前的效果。

最后训练的时候使用的初始化策略和基于集成思想的随机路径反而非常有实用价值。

## Reference

\[1\] Zhang X, Li Z, Loy C C, et al. Polynet: A pursuit of structural diversity in very deep networks\[C\]//Computer Vision and Pattern Recognition \(CVPR\), 2017 IEEE Conference on. IEEE, 2017: 3900-3908.

\[2\]Szegedy C, Ioffe S, Vanhoucke V, et al. Inception-v4, inception-resnet and the impact of residual connections on learning\[C\]//AAAI. 2017, 4: 12.

\[3\] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich. Going deeper with convolutions. In CVPR, 2015.

\[4\] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 770-778.

\[5\] M. Abdi and S. Nahavandi. Multi-residual networks. arXiv preprint arXiv:1609.05672, 2016. 1, 2, 3, 4

\[6\] Huang G, Liu Z, Weinberger K Q, et al. Densely connected convolutional networks\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2017, 1\(2\): 3.

