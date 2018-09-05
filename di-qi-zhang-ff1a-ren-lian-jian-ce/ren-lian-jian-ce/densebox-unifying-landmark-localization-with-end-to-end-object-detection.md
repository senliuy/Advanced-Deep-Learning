# DenseBox: Unifying Landmark Localization with End to End Object Detection

## 前言

DenseBox百度IDL的作品，提出的最初动机是为了解决普适的物体检测问题。其在2015年初就被提出来了，甚至比Fast R-CNN还要早，但是由于论文发表的比较晚，虽然算法上非常有创新点，但是依旧阻挡不了Fast R-CNN一统江山。

DenseBox的主要贡献如下：

1. 使用全卷积网络，任务类型类似于语义分割，并且实现了端到端的训练和识别，而R-CNN系列算法是从Faster R-CNN中使用了RPN代替了Selective Search才开始实现端到端训练的，而和语义分割的结合更是等到了2017年的Mask R-CNN才开始；
2. 多尺度特征，而R-CNN系列直到FPN才开始使用多尺度融合的特征；
3. 结合关键点的多任务系统，DenseBox的实验是在人脸检测数据集（MALF）上完成的，结合数据集中的人脸关键点可以使算法的检测精度进一步提升。

## 1. DenseBox详解

### 1.1 训练标签

DenseBox没有使用整幅图作为输入，因为作者考虑到一张图上的背景区域太多，计算时间会严重浪费在对没用的背景区域的卷积上。而且使用扭曲或者裁剪将不同比例的图像压缩到相同尺寸会造成信息的丢失。作者提出的策略是从训练图片中裁剪出包含人脸的patch，这些patch包含的背景区域足够完成模型的训练，详细过程如下：

1. 根据Ground Truth从训练数据集中裁剪出大小是人脸的区域的高的4.8倍的正方形作为一个patch，且人脸在这个patch的中心；
2. 将这个patch resize到$$240\times240$$大小。

举例说明：一张训练图片中包含一个$$60\times80$$的人脸，那么第一步会裁剪出大小是$$384\times384$$的一个patch。在第二步中将这个patch resize到$$240\times240$$。这张图片便是训练样本的输入。

训练集的标签是一个$$60\times60\times5$$的热图（图1），$$60\times60$$表示热图的尺寸，从这个尺寸我们也可以看出训练样本经过了两次降采样。$$5$$表示热图的通道数，组成方式如下：

<figure>
<img src="/assets/DenseBox_1.png" alt="图1：DenseBox的Ground Truth" />
<figcaption>图1：DenseBox的Ground Truth</figcaption>
</figure>

1. 图1中最前面的热图用于标注人脸区域置信度，前景为1，背景为0。DenseBox并没有使用左图的人脸矩形区域而是使用半径（$$r_c$$）为Ground Truth的高的0.3倍的圆作为标签值，而圆形的中心就是热图的中心，即有图中的白色圆形部分；

2. 图1中后面的四个热图表示像素点到最近的Ground Truth的四个边界的距离，如图2所示，Ground Truth为蓝色矩形，表示为$$d^* = (d^*_{x^t},d^*_{x^b},d^*_{y^t},d^*_{y^b})$$, 绿色为预测的矩形，表示为$$\hat{d} = (\hat{d}_{x^t},\hat{d}_{x^b},\hat{d}_{y^t},\hat{d}_{y^b})$$。(论文中符号的使用混乱且有错误，这里没有采用和论文完全相同的符号)。

<figure>
<img src="/assets/DenseBox_2.jpeg" alt="图2：DenseBox中距离热图示意图" width="600"/>
<figcaption>图2：DenseBox中距离热图示意图</figcaption>
</figure>

如果训练样本中的人脸比较密集，一个patch中可能出现多个人脸，如果某个人脸和中心点处的人脸的高的比例在$$[0.8,1.25]$$之间，则认为该样本为正样本。

作者认为DenseBox的标签设计是和感受野密切相关的，具体的讲，结合1.2节要分析的网络结构我们可以计算得到热图中每个像素的感受野是$$48\times48$$，这和我们每个patch中每个人脸的尺寸是非常接近的。在DenseBox中，每个像素点有5个预测值，而这5个预测值便可以确定一个检测框，所以DenseBox本质上也是一个密集采样，每个图片的采样个数是$$60\times60=3600$$个。

### 1.2 网络结构

DenseBox使用了16层的VGG-19作为骨干网络，但是只使用了其前12层，如图3所示。

<figure>
<img src="/assets/DenseBox_3.png" alt="图3：DenseBox中的网络结构" width="600"/>
<figcaption>图3：DenseBox中的网络结构</figcaption>
</figure>

首先需要注意的是在网络的Conv3_4和Conv4_4之间发生了一次特征融合，融合的方式是Conv4_4层的双线性插值上采样，因此得到的Feature Map和Conv3_4是相同的，即为$$60\times60$$，通过计算我们可以得知Conv3_4层的感受野的尺寸是$$48\times48$$，该层的尺寸和标签中的人脸尺寸接近，用于捕捉人脸区域的关键特征；Conv4_4层的感受野的大小是$$118\times118$$，用于捕捉人脸的上下文特征。

上采样之后网络有两个分支，分别用于计算检测损失和Bounding Box的回归损失，分支由$$1\times1$$卷积核Dropout组成。从这里我们看出DenseBox也是一个多任务模型，下面我们开始介绍这个多任务模型。

网络在VGG-19部分的初始化使用的是ImageNet上得到的迁移学习的参数，其余部分使用的是Xavier初始化。

### 1.3 多任务模型

DenseBox的第一个损失为分类损失（检测损失）。在1.1节中我们知道，分类标签$$y^* \in \{0,1\}$$，设$$\hat{y}$$为模型的预测值。作者使用的是l2损失:

$$
\mathcal{L}_{cls}(\hat{y}, y^*) = ||\hat{y} - y^*||^2
$$

第二个分支是bounding box回归损失，即计算图2中像素点分别到Ground Truth和到预测值之间的l2损失：

$$
\mathcal{L}_{loc}(\hat{d}, d^*) = \sum_{i \in \{x^t, y^t, x^b, y^b\}}||\hat{d}_i - d^*_i||^2
$$

### 1.4 平衡采样

在1.1节的最后我们讲到，DenseBox可被视为3600个样本的密集采样，其中每个像素点都可以看做是一个样本。在算法中并不是所有样本都会参与到训练中，且为了平衡正负样本，提高模型精度，DenseBox采用了以下策略。

#### 1.4.1 忽略灰色区域

所谓灰色区域，是指正负样本边界部分的像素点，因为在这些区域由于标注的样本是很难区分的，让其参与训练反而会降低模型的精度，因此这一部分不会参与训练，在论文中，长度小于2的边界部分视为灰色区域。DenseBox使用$$f_{ign}$$对灰色样本进行标注，$$f_{ign}=1$$表示为灰色区域样本。

### 1.4.2 Hard Negative Mining

DenseBox使用的Hard Negative Mining的策略和SVM类似，具体策略是：

1. 计算整个patch的3600个所有样本点，并根据loss进行排序；
2. 取其中的1%，也就是36个作为hard-negative样本；
3. 随机采样36个负样本和hard-negative构成72个负样本；
4. 随机采样72个正样本。

使用上面策略得到的144个样本参与训练，DensoBox使用掩码$$f_{sel}$$对参与训练的样本点进行标注：样本被选中$$f_{sel}=1$$，否则$$f_{sel}=0$$。

### 1.4.3 使用掩码的损失函数

损失函数使用掩码来控制哪些样本参与训练，其掩码$$M(\hat{t}_i)$$是由1.4.1节的$$f_{ign}$$和1.4.2节的$$f_{sel}$$共同决定的：

$$
M(\hat{t}_i) = 
\left\{
\begin{array}{}
0 \quad f_{ign}^i = 1\text{ or } f_{sel}^i = 0\\
1 \quad \text{otherwise}
\end{array}
\right.
$$

使用掩码后，得到的损失函数如下

$$
\mathcal{L}_{det}(\theta) = \sum_i (M(\hat{t}_i)\mathcal{L}_{cls}(\hat{y}_i, y^*_i) + \lambda_{loc}[y^*_i>0]M(\hat{t}_i)\mathcal{L}_{cls}(\hat{d}_i, d^*_i)
$$

其中$$\theta$$为卷积网络的参数，$$[y^*_i>0]$$表示只有正样本参与bounding box的训练，其数学表达式为

$$
[y^*_i>0] = 
\left\{
\begin{array}{}
 0 \quad y^*_i>0\\
 1 \quad \text{otherwise}
\end{array}
\right.
$$

$$\lambda_{loc}$$是平衡两个任务的参数，论文中值为3。位置$$d_i$$使用的是归一化的值。
## Reference

\[1\] Qin H, Yan J, Li X, et al. Joint training of cascaded cnn for face detection\[C\]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2016: 3456-3465._