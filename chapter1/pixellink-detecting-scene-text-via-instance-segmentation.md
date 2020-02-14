# PixelLink: Detecting Scene Text via Instance Segmentation

## 前言

在前面的文章中，我们介绍了文本框回归的算法[DeepText](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/deeptext-a-unified-framework-for-text-proposal-generation-and-text-detection-in-natural-images.html){{"zhong2016deeptext"|cite}}, [CTPN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/detecting-text-in-natural-image-with-connectionist-text-proposal-network.html){{"tian2016detecting"|cite}}以及[RRPN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/arbitrary-oriented-scene-text-detection-via-rotation-proposals.html){{"ma2018arbitrary"|cite}}；也介绍了以及实例分割的[HMCP](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/scene-text-detection-via-holistic-multi-channel-prediction.html){{"yao2016scene"|cite}}，在这里我们介绍一下另外一个基于实例分割的文字检测算法：PixelLink{{"deng2018pixellink"|cite}}。根据PixelLink的算法名字我们也可以推测到，它有两个重点，一个是Pixel（像素），一个是Link（像素点之间的连接），这两个重点也是构成PixelLink的网络的输出层和损失函数优化的目标值。下面我们来看一下PixelLink的详细内容。

## 1. PixelLink详解

### 1.1 骨干网络

PixelLink是一个基于实例分割的算法，它的核心思想有两点：

1. 判断图中的点是否为文本区域；
2. 该点是否和其附近8个点（正左，左上，正上，右上，正右，右下，正下，左下）是同一个实例；

根据上面的分析，我们可以得到PixelLink的两个输出（图1右上角）。一个是用于文字/非文字区域的预测（$$ 1\times2 = 2 $$），另外一个是用于连接的预测 ($$8\times2=16$$)，由于它们都是基于像素点的预测，所以它们的输出是2+16=18个。由于PixelLink要分别预测正连接（Positive Link）和负连接（Negative Link），所以它要对8个方向乘2。

有的同学可能会对产生这么一个疑问：既然文本区域和非文本区域以及正连接和负连接是互斥的，那么为什么要使用正负两个功能重叠的头的？正如在SegLink中所介绍的，正连接用于表示两个像素是否属于同一个实例，而负连接是用来判断两个像素是否为不同的连接。

<figure>
<img src="/assets/PixelLink_1.png" alt="图1：PixelLink网络结构图" />
<figcaption>图1：PixelLink网络结构图</figcaption>
</figure>

如图1的左侧部分所示，PixelLink的左侧是VGG-16的网络结构。有比较大变化的是VGG-16的最后一个Block，也就是图1网络的左下角，它的改变有两点：

1. pool5的步长是1（是否还有存在的必要？）
2. 为了保证像素上下左右之间的顺序，fc6和fc7两个全连接换成了卷积操作。

PixelLink右侧是一个上采样的过程，采用的上采样方法是双线性插值。PixelLink采用了广泛使用的U-Net架构进行两侧特征的融合，两侧融合使用的是单位加的操作。PixelLink提供了两种融合方式，图1所示的融合的是_{conv2_2, conv3_3, conv4_3, conv5_3, fc_7}_层，论文中管这种结构叫做**PixelLink+VGG16 2s**, 其中2s表示的是融合之后的尺寸是输入图像的$$\frac12$$, PixelLink的另外一种融合方式融合的是_{conv3_3, conv4_3, conv5_3, fc_7}_层，它的尺寸是输入图像的$$\frac14$$，所以它被叫做**PixelLink+VGG16 4s**。

### 1.2 PixelLink的Ground Truth

不同于基于边界框回归的检测算法，PixelLink有其特有的GroundTruth，我们应当将Pascal VOC或者是COCO数据集转化成PixelLink所需要的格式。

如果该像素位于文本标注框之内，则该像素的文本区域标注为Positive，当有覆盖存在时，不重叠的区域标注为正，剩余的所有像素均标注为负样本。关于连接正负的判断，论文中讲解的不够详细，我们分析代码之后才能弄明白：
1. 如果一个像素在文本区域中，则需要判断这个点与其8个邻居的正负关系，其实我们只需要判断文本区域边界的点即可，因为非边界像素的8个邻居的连接肯定为正。
2. 判断一个点与其邻居的连接的正负时，只需要判断它的邻居是否也在该像素点的文本框内，如果在的话，则它们的连接为正，否则为负。

### 1.3 PixelLink的损失函数

如前面所介绍的，PixelLink是由文本区域和非文本区域构成的双任务模型，所以它的损失函数由两个部分组成：

$$
L = \lambda L_{\text{pixel}} + L _ {\text{link}}
$$

上式中的$$\lambda$$是多任务的权值参数，作者发现像素损失更为重要，所以在论文中$$\lambda=2$$.

#### 1.3.1 像素损失$$L_{\text{pixel}}$$

像素损失主要是要解决小尺度文本区域的准确率问题。对于一个文本区域尺寸变化非常大的图片来说，如果我们为每一个像素值都分配一个相同的权值，那么大尺寸会有远大于小尺寸的内容参与损失函数的计算，这对小物体的检测是非常不利的，因此我们需要设计一个权值和尺寸成反比的损失函数来优化模型。作者将之命名为_Instance-Balanced Cross-Entropy Loss_，对于一个输入图像的所有文本区域，首先计算一个对每个区域都相等的值$$B$$（不明白为什么所有的区域的值都相等，论文中依然为其添加下标）。$$S_i$$是这个文本框实例的面积，该文本框中像素的权值与该文本框的面积成反比，即$$ w_i = \frac {B}{S_i} $$，也就是小面积的文本区域的像素点会得到更大的权值。

$$
B = \frac {S} {N}, S = \sum_{i} ^ {N} S_i, \forall i \in \{1, ..., N\}
$$

在上式中，$$S$$即为文本区域的总面积。PixelLink采用OHEM的策略来采样负样本（非文本区域），其中$$r\times S$$个损失值最大的负样本被采样用作PixelLink的负样本来优化。所有的正负样本的像素的权值构成矩阵$$W$$。像素损失$$L_{\text{pixel}}$$表示为：

$$
L_{\text{pixel}} = \frac{1}{(1+r)S} W L_{\text{pixel}\_\text{CE}}

$$

其中$$L_{\text{pixel_CE}}$$表示局域文本/非文本区域的交叉熵损失函数。

#### 1.3.2 连接损失$$L_\text{link}$$

连接损失由正连接损失和负连接损失组成，分别表示为

$$
L_\text{link_pos} = W_\text{pos_link} L_\text{link_CE}
$$

$$
L_\text{link_neg} = W_\text{neg_link} L_{\text{link}\_\text{CE}}
$$

其中$$L_\text{link_CE}$$是连接的交叉熵损失，$$W_\text{pos_link}$$和$$W_\text{neg_link}$$是两个权值，他是跟素损失的权值矩阵$$W$$的计算得到：

$$
W_\text{pos_link} = W(i,j) * (Y_\text{link}(i,j,k)==1)
$$

$$
W_\text{neg_link} = W(i,j) * (Y_\text{link}(i,j,k)==0)
$$

那么关于连接的类别平衡交叉熵损失函数表示为：

$$
L_\text{link} = \frac{L_\text{link_pos}}{rsum(W_\text{pos_link})} + \frac{L_\text{link_neg}}{rsum(W_\text{neg_link})} 
$$

其中$$rsum$$表示reduce-sum操作。

### 1.3 后处理

#### 1.3.1 像素合并

当得到网络的输出结果后，PixelLink需要将其转化为文本框，整个流程的关键环节有三步：
1. 当两个像素点都是正像素且它们之间至少有一个连接是正的时候，那么这两个像素点构成一个连通域；
2. 使用并查集的方式确定所有的连接；
3. 使用OpenCV的minAreaRect方式确定文字区域，它可以使矩形，也可以是多边形。

#### 1.3.2 后处理

由于PixelLink是基于实例分割的算法，所以他会产生很多小的区域，我们可以根据数据集的特征对这些误检进行过滤。

## 2. 总结

与传统的基于回归框的文字检测算法对比，这种基于实例分割的算法带来了两个优点：

1. 更擅长小物体的检测：因为它使用的是像素连通域拼接的方式形成的文本框，这种从底到上的方法使得PixelLink非常善于检测小物体；

2. 对训练集的数量依赖更小：基于像素的方法保证了PixelLink的样本数量，以及对复杂背景更鲁棒的抵抗能力，这使得PixelLink的训练数据的数量依赖更小。

但是这种方式也给PixelLink带来了一些缺点：

1. PixelLink的自底向上的方法使得大尺度的物体的检测的困难程度远大于小物体，可能会带来大尺寸物体的检测不准确；

2. 这种只看该像素与其周围邻居而忽略了更多的上下文信息的方式可能会使PixelLink产生一些误检；

3. 过分依赖后处理的操作来去掉误检，有些人工干预的痕迹，在有些场景中这些值很难确定，进而产生一些误检和漏检。



















