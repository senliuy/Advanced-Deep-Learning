# YOLO9000: Better, Faster, Stronger

## 前言

在这篇像极了奥运格言（Faster，Higher，Stronger）的论文 {{"redmon2017yolo9000"|cite}}中，作者提出了YOLOv2和YOLO9000两个模型。

其中YOLOv2采用了若干技巧对[YOLOv1](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/you-only-look-once-unified-real-time-object-detection.html) {{"redmon2016you"|cite}} 的速度和精度进行了提升。其中比较有趣的有以下几点：

1. 使用聚类产生的锚点代替[Faster R-CNN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.html){{"ren2015faster"|cite}} 和[SSD](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/ssd-single-shot-multibox-detector.html){{"liu2016ssd"|cite}} 手工设计的锚点；
2. 在高分辨率图像上进行迁移学习，提升网络对高分辨图像的响应能力；
3. 训练过程图像的尺寸不再固定，提升网络对不同训练数据的泛化能力。

除了以上三点，YOLO还使用了残差网络的直接映射的思想，R-CNN系列的预测相对位移的思想，Batch Normalization，全卷积等思想。YOLOv2将算法的速度和精度均提升到了一个新的高度。正是所谓的速度更快（Faster），精度更高（Better/Higher）

论文中提出的另外一个模型YOLO9000非常巧妙的使用了WordNet {{"miller1990introduction"|cite}}的方式将检测数据集COCO和分类数据集ImageNet整理成一个多叉树，再通过提出的联合训练方法高效的训练多叉树对应的损失函数。YOLO9000是一个非常强大（Stronger）且有趣的模型，非常具有研究前景。

在下面的章节中，我们将论文分成YOLOv2和YOLO9000两个部分并结合论文和源码对算法进行详细解析。

## 1. YOLOv2: Better, Faster

YOLOv2使用的是和YOLOv1相同的思路，算法流程参考[YOLOv1](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/you-only-look-once-unified-real-time-object-detection.html)的介绍，但看这篇文章肯定会感到一头雾水，因为论文并没有详细介绍YOLOv2的详细流程。而且我也不打算介绍，因为这只是对YOLOv1无畏的重复，强烈推荐读者能搞懂YOLOv1之后再来读这篇文章。

### 1.1. 更好（Better）

YOLOv1之后，一系列算法和技巧的提出极大的提高了深度学习在各个领域的泛化能力。作者总结了可能在物体检测中有用的方法和技巧（图1）并将它们结合成了我们要介绍的YOLOv2。所以YOLOv2并没有像SSD或者Faster R-CNN具有很大的难度，更多的是在YOLOv1基础上的技巧方向的提升。在下面的篇幅中，我们将采用和论文相同的结构并结合基于Keras的[源码](https://github.com/yhcc/yolo2)对YOLOv2中涉及的技巧进行讲解。

###### 图1：YOLOv2中使用的技巧及带来的性能提升

![](/assets/YOLOv2_1.png)

#### 1.1.1. BN替代Dropout

YOLOv2中作者舍弃了Dropout而使用Batch Normalization（BN）来减轻模型的过拟合问题，从图1中我们可以看出BN带来了2.4%的mAP的性能提升。

Batch Normalization和Dropout均有正则化的作用。但是Batch Normalization具有提升模型优化的作用，这点是Dropout不具备的。所以BN更适合用于数据量比较大的场景。

关于BN和Dropout的异同，可以参考Ian Goodfellow在Quora上的[讨论](https://www.quora.com/What-is-the-difference-between-dropout-and-batch-normalization#)。

#### 1.1.2. 高分辨率的迁移学习

之前的深度学习模型很多均是生搬在ImageNet上训练好的模型做迁移学习。由于迁移学习的模型是在尺寸为$$224\times224$$的输入图像上进行训练的，进而限制了检测图像的尺寸也是$$224\times224$$。在ImageNet上图像的尺寸一般在500左右，降采样到224的方案对检测任务的负面影响要远远大于分类任务。

为了提升模型对高分辨率图像的响应能力，作者先使用尺寸为$$448\times448$$的ImageNet图片训练了10个Epoch（并没有训练到收敛，可能考虑$$448\times448$$的图片的一个Epoch时间要远长于$$224\times224$$的图片），然后再在检测数据集上进行模型微调。图1显示该技巧带来了3.7%的性能提升

#### 1.1.3.  骨干网络Darknet-19

YOLOv2使用了DarkNet-19作为骨干网络（图2），在这里我们需要注意两点：

1. YOLOv2输入网络的图像尺寸并不是图2画的$$224\times224$$, 而是使用了$$416\times416$$的输入图像，原因我们随后会介绍；
2. 在$$3\times3$$卷积中间添加了$$1\times1$$卷积，Feature Map之间的一层非线性变化提升了模型的表现能力；
3. Darknet-19进行了5次降采样，但是在最后一层卷积并没有添加池化层，目的是为了获得更高分辨率的Feature Map；
4. Darknet-19中并不含有全连接，使用的是全局平均池化的方式产生长度固定的特征向量。

###### 图2：Darknet网络结构

![](/assets/YOLOv2_2.png)

首先，YOLOv2使用的是$$416\times416$$的输入图像，考虑到很多情况下待检测物体的中心点容易出现在图像的中央，所以使用$$416\times416$$经过5次降采样之后生成的Feature Map的尺寸是$$13\times13$$，这种奇数尺寸的Feature Map获得的中心点的特征向量更准确。其实这也和YOLOv1产生$$7\times7$$的理念是相同的。

其次，YOLOv2也学习了Faster R-CNN和RPN中的锚点机制，锚点也可以被叫做先验框，即给出一个检测框可能的形状，向先验框的收敛总是比向固定box的收敛要容易的多。不同于以上两种算法的是YOLOv2使用的是在训练集上使用了K-means聚类产生的候选框，而上面两种方法的候选框是人工设计的。

最后，关于全卷积的作用，$$1\times1$$ 卷积带来的非线性变化我们已经在之前的文章中多次提及，这里便不再说明。

#### 1.1.4. 锚点聚类

在1.1.3节中我们介绍到YOLOv2使用的k-means聚类产生的锚点，该方法提出的动机是考虑到人工设计的锚点具有太强的主观性，与其主管设计，不如根据训练集学习一组更具有代表性的锚点。

由于锚点的中心即是Grid的中心点（YOLOv1是$$7\times7$$ 的Grid，YOLOv2是$$13\times13$$ 的Grid），所以所需要聚类的只有锚点的宽（$$w$$）和高（$$h$$）。更形象的解释就是对训练集的Ground Truth的聚类，聚类的分类目标是Ground Truth不同的大小和不同的长宽比。

类别数目为$$k$$ 的k-means可以简单总结为四步：

1. 随机初始化$$k$$ 个中心点；
2. 根据样本和中心点间的欧式距离确定样本所属的类别；
3. 根据样本的类别更新样本的中心点；
4. 循环执行2，3直到中心点的位置不再变化。

YOLOv2并没有直接使用欧氏距离作为聚类标准，因为大尺寸的样本产生的误差要比小样本大。锚点作为候选框的先验，当然是希望锚点与Ground Truth的IoU越大越好，所以这里使用了IoU作为分类标准，即：


$$
d(box, centroid) = 1 - IOU(box, centroid)
$$


聚类的数目$$k$$是一个超参数，经过一系列的对比试验（图3），出于对速度和精度的折中考虑，YOLOv2使用的是$$k=5$$的值。

###### 图3：k-means的$$k$$和IoU均值的实验对比

![](/assets/YOLOv2_3.png)

遗憾的是我并没有在源码中找到k-means的实现，在darknet源码中找到了两组值：

```py
# coco
anchors =  0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828  

# voc
anchors =  1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071
```

上面的值应该是锚点在$$13\times13$$的Feature Map上的尺寸，可以等比例换算到原图中（图3右侧部分）。从上面的两组值我们也可以看出coco数据集的物体尺寸更偏小一些。

为了验证上面总结的思想，我用python实现了一份用于锚点聚类的k-means，源码见[链接](https://github.com/senliuy/Advanced-Deep-Learning/blob/master/assets/yolo2_kmeans.ipynb)。核心算法见代码片段1：

###### 代码片段1：用于锚点聚类的k-means

```py
def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    """
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    # initiate centroids
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters
```

跑了一下在pascal voc上的聚类，得到的$$k=5$$和$$k=9$$的实验结果如下:

```
# k=5
Accuracy: 60.07%
Boxes:
 [[4.446      6.48266667]
 [2.106      3.95502646]
 [1.17       1.8744186 ]
 [9.81066667 9.91591592]
 [0.494      0.90133333]]
Ratios:
 [0.53, 0.55, 0.62, 0.69, 0.99]

# k=9
Accuracy: 66.66%
Boxes:
 [[10.738      10.192     ]
 [ 1.56        1.456     ]
 [ 0.884       2.704     ]
 [ 3.562       3.12      ]
 [ 5.75612121  7.67      ]
 [ 0.494       0.69333333]
 [ 2.938       6.526     ]
 [ 0.572       1.42133333]
 [ 1.638       3.98666667]]
Ratios:
 [0.33, 0.4, 0.41, 0.45, 0.71, 0.75, 1.05, 1.07, 1.14]
```

虽然和源码提供的值不完全一样，但是取得的先验框和源码的差距很小，而且IoU也基本符合图3给出的实验结果。

#### 1.1.5. 直接位置预测

YOLOv2使用了和YOLOv1类似的损失函数，不同的是YOLOv2将分类任务从cell中解耦。因为在YOLOv1中，cell负责预测与之匹配的类别，bounding box负责位置精校，也就是预测位置。YOLOv1的输出层我们在[YOLOv1](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/you-only-look-once-unified-real-time-object-detection.html)的图3中进行了描述。但是在YOLOv2中使用了锚点机制，物体的类别和位置均是由锚点对应的特征向量决定的，如图4。

###### 图4：YOLOv2的输出层
![](/assets/YOLOv2_4.png)

在Keras源码中使用的是80类的COCO数据集，锚点数$$k=5$$，所以YOLOv2的每个cell的输出层有$$(80+5)\times 5 = 425$$个节点。

直接将锚点机制添加到YOLO中（也就是SSD）会产生模型不稳定的问题，尤其在早期迭代的时候，这些不稳定大部分是发生在预测$$(x,y)$$的时候。

回顾一下[SSD](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/ssd-single-shot-multibox-detector.html)的损失函数，相对位移$$(x,y)$$的计算方式为：

$$\hat{g}^{cx}_j = (g^{cx}_j - d^{cx}_i)/d^w_i$$  
$$\hat{g}^{cy}_j = (g^{cy}_j - d^{cy}_i)/d^h_i$$

即

$$g^{cx}_j = \hat{g}^{cx}_j * d^w_i + d^{cx}_i$$  
$$g^{cy}_j = \hat{g}^{cy}_j * d^h_i + d^{cy}_i$$

注意论文在这个地方是减号，应该是一个错误。

其中$$\hat{g}^{cx}_j$$和$$\hat{g}^{cy}_j$$是预测值，在模型训练初期，由于使用的是迁移学习和随机初始化得到的网络，此时网络必然收敛的不是特别好，这就造成了$$\hat{g}^{cx}_j$$和$$\hat{g}^{cy}_j$$的随机性。再加上并没有对$$\hat{g}$$的值加以限制，使得预测的检测卡可能出现在网络中的任何位置而且这些事和锚点如何初始化无关的。正确的做法应该是预测的检测框应该也是由该锚点负责的，也就预测的检测框的中心点应该落到该锚点的内部。

为了解决这个问题，YOLOv2采用了YOLOv1中使用的相对一个cell的位移，同时使用了logistic函数将预测值限制在了$$[0,1]$$ 的范围内。YOLOv2的输出层会产生5个值，$$(t_x, t_y, t_w, t_j, t_o)$$，该输出层对应cell的左上角为$$(c_x, c_y)$$, 宽和高分别为$$(p_w, p_h)$$, 那么对应的预测的相对位移为：

$$b_x = \sigma(t_x) + c_x$$  
$$b_x = \sigma(t_y) + c_y$$  
$$b_w = p_w e^{t_w}$$  
$$b_h = p_h e^{t_h}$$  
$$Pr(object)\times IoU(b, object) = \sigma(t_o)$$

上式表示的几何关系见图5。

###### 图5：YOLOv2的预测值和匹配cell的几何关系

![](/assets/YOLOv2_5.png)

YOLOv2的损失函数`./utils/loss_util.py`和YOLOv1的是相同的，均是由5个任务组成的多任务损失函数。源码中各个模型的权重也和YOLOv1中提到的权重一致。

从图1中我们可以看出，1.1.4节的Dimension Clusters加上1.1.5节的Direct Location Prediction非常有效的将mAP提高了4.8%。

#### 1.1.6. 细粒度特征

在特征在物体检测和语义分割中的研究中表明，使用多尺度的是非常有效的。YOLOv2的多尺度是通过将最后一个$$26\times26$$的Feature Map直接映射到最后一个$$13\times13$$的Feature Map上形成的。

这里面有两个技术细节需要详细说明一下：

1. $$26\times26\times512$$的Feature Map首先通过TensorFlow的`space_to_depth()`函数转换成$$13\times13\times2048$$的Feature Map（图6）然后再和后面的Feature Map进行映射的；
2. 论文中采用的是类似残差网络的映射方式，也就是将Feature Map执行单位加操作，但是源码中使用的是DenseNet的方式，也就是Merge成$$13\times13\times(2048+1024)$$的 Feature Map。

###### 图6：tf.space\_to\_depth\(\)

![](/assets/YOLOv2_6.png)

图1显示该方法带来了1%的性能提升。

#### 1.1.7. 多尺度训练

全卷积的使用是网络在单词测试环境中每张输入图像的尺寸可以不同。但是当使用mini-batch方式训练的时候，图像的尺寸要是相同的，因为TensorFLow等框架要求输入网络的是一个维度为$$N\times W\times H\times C$$ 的矩阵。其中$$N$$是批（batch）的大小，$$W$$, $$H$$, $$C$$分别是图片的宽，高和通道数。所以虽然每个batch之内图像的尺寸必须是相同的，但是不同的batch之间图像的尺寸是不受限于框架的。YOLOv2便是基于这点实现了其训练过程中的多尺度。

在YOLOv2z中，每隔10个batch便随机从$${320, 352, 384, ..., 608}$$选择一个新的尺度作为输入图像的尺寸，多尺度训练将mAP提高了1.4%。

### 1.2. 更快（Faster）

YOLOv2用于提速的技术我们已经在1.1节中介绍过，这里仅列出技术和提速的关系：

1. 使用全卷积网络代替全连接，网络具有更少的参数，速度更快；
2. 使用Batch Normalization代替Dropout的正则化的功能，使用BN训练的模型更稳定；
3. DarkNet-19将VGG-16运算数量从306.9亿降低到55.8亿。

文至此处，一个更快，更好的YOLOv2已介绍完毕，虽然不像SSD对YOLOv1的提升在技术上那么惊艳，但其使用的若干技术确实是非常有效。在下一部分我们将开始介绍YOLO9000，一个无论在技术，还是再创新点上都非常惊艳的模型。

## 2. YOLO9000: 更强（Stronger）

在80类的COCO数据集中，物体的[类别](https://github.com/yhcc/yolo2/blob/master/model_data/coco_classes.txt)都是比较抽象的，例如类别‘dog’并没有精确到具体狗的品种（哈士奇或者柯基等）。而ImageNet中包含的类别则更具体，不仅包含‘dog’类，还包括‘poodle’和‘Corgi’类。我们将COCO数据集的狗的图片放到训练好的ImageNet模型中理论上是能判断出狗的品种的，同理我们将ImageNet中的狗的图片（不管是贵宾犬，还是柯基犬）放任在COCO训练好的检测模型中，理论上是能够检测出来的。但是生硬的使用两个模型是非常愚蠢且低效的。YOLO9000的提出便是巧妙地利用了COCO数据集提供的检测标签和ImageNet强大的分类标签，使得训练出来的模型具有强大的检测和分类的能力。

遗憾的是并没有找打YOLO9000的TensorFlow或是Keras源码，暂且用基于DarkNet的[源码](https://github.com/pjreddie/darknet)分析之：

### 2.1. 分层分类

ImageNet的数据集的标签是通过WordNet\[5\]的方式组织的，WordNet反应了物体类别之间的语义关系，例如‘dog’类既是‘canine’的子类，也是‘domestic animal’的子类，由于一个子节点有两个父节点，所以WordNet本质上是一个图模型。

在YOLOv2中，作者将WordNet简化成了一个分层的树结构，即WordTree。WordTree的生成方式也很简单，如果一个节点含有多个父节点，只需要保存到根节点路径最短的那条路径即可，生成的层次树模型见图7。在DarkNet的源码中，WordTree以二进制文件的形式保存在[./data/9k.tree](https://github.com/pjreddie/darknet/blob/master/data/9k.tree)文件中。在9k.tree中，第一列表示类别的标签，标签的类别可以在[./data/9k.names](https://github.com/pjreddie/darknet/blob/master/data/9k.names)通过行数（从0开始计数）对应上，9k.tree的第二列表示该节点的父节点，值为$$-1$$的话表示父节点为空。

###### 图7：YOLO9000的WordTree

![](/assets/YOLOv2_7.png)

例如从8888\(military officer\)行开始向上回溯到根节点，走过的路径依次是:

6920\(corgi\) -&gt; 6912\(dog\) -&gt; 6856\(canine\) -&gt; 6781\(carnivore\) -&gt; 6767\(placenal\) -&gt; 6522\(mammal\) -&gt; 6519\(vertebrate\) -&gt; 6468\(chordate\) -&gt; 5174\(animal\) -&gt; 5170\(worsted\) -&gt; 1042\(living thing\) -&gt; 865\(whole\) -&gt; 2\(object\)-&gt; -1

貌似问题不大。

现在问题是如果标签是以WordTree的形式组织的，我们如何确定检测的物体属于哪一类呢？在使用WordTree进行分类时，我们预测每个节点的条件概率，以得到同义词集合（synset）中每个同义词的下义词（hyponym）的概率，例如在‘dog’节点处我们要预测

$$Pr(poodle | dog)$$  
$$Pr(Corgi | dog)$$  
$$Pr(griffon | dog)$$  
$$...$$

当我们要预测一只狗是不是柯基时，$$Pr(Corgi)$$是一系列条件概率的乘积：

$$Pr(Corgi) = Pr(Corgi|dog) \times Pr(dog|canine) \times ... \times Pr(living thing|whole) \times Pr(whole|object) \times Pr(object)$$

其中$$Pr(object) = 1$$。Pr\(Corgi\|dog\)则是在‘dog’的所有下义词中为‘Corgi’的概率，由softmax激活函数求得，其它情况依次类推（图8）。

###### 图8：在ImageNet和在WordTree下的预测。

![](/assets/YOLOv2_8.png)

图8是作者为了验证其想法建立的WordTree 1k模型，在构建WordTree时添加了369个中间节点以便构成一个完整的WordTree。根据上面的分析，YOLO9000是一个多标签分类的模型，例如‘Corgi’则是一个含有1369个标签的的数据，其one-hot编码的形式为在第（6920，6912，6856，6781，6767，6522，6519，6468，5174，5170，1042，865，2）共13个位置处为1，其余的位置均为0。

在预测物体的类别时，我们遍历整个WordTree，在每个分割中采用最高的置信度路径，直到分类概率小于某个阈值（源码给的是0.6）时，然后预测结果。

### 2.2. 使用WordTree合并数据集

WordTree的类别数量非常大，基本可以囊括目前所有的检测数据集，只需要在WordTree中标明哪些节点是检测数据集上的即可，图7显示的是COCO数据集合并到WordTree的结果。其中蓝色节点表示COCO中可以检测的类别。

### 2.3. 检测和分类的联合训练

WordTree 1k的实验验证了作者的猜测，作者更大胆的将分类任务扩大到了整个ImageNet数据集。YOLO9000提取了ImageNet最常出现的top-9000个类别并在构建WordTree时将类别数扩大到了9418类。由于同时使用了ImageNet和COCO数据集进行训练，为了平衡分类和检测任务，YOLO9000将COCO上采样到和ImageNet的比例为$$1:4$$。

YOLO9000使用了YOLOv2的框架但是有以下改进：

1. 每个锚点的输出不再是85个，而是$$9418+5 = 9423$$ 个；
2. 为了减少每个cell的输出节点数，YOLO9000使用了3个锚点，但每个cell的输出也达到了$$3\times 9423=28269$$个；
3. 当运行分类任务时，要更新该节点和其所有父节点的权值，且不更新检测任务的权值，因为我们此时根本没有任何关于检测的信息；
4. 锚点和Ground Truth的IoU大于0.3时便被判为正锚点，而YOLOv2的阈值是0.5。

## 总结

YOLO9000这篇论文算是干货满满的一篇文章，首先YOLOv2通过一系列非常有效的Trick将物体检测的速度和精度刷新到了新的高度。这些Trick不仅在YOLOv2中非常有效，而且对我们的其它任务也很有参考价值，例如高分辨率迁移学习应用到语义分割，多尺度训练应用到图像分类任务等。

YOLO9000更是强大到令人发指，其对COCO的80类的子类和父类能进行检测并不让我感到意外，YOLO9000强大之处在于路径中没有COCO类别的156类中也取得了非常不错的效果。这种半监督学习是非常有研究和应用前景的一个方向，因为在我们的大部分场景中获得大量数据集难度非常大，但我们又可以多少搞到写数据，这时候就要发挥半监督学习的作用了。

