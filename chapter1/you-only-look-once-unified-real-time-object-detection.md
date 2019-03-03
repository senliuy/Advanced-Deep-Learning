# You Only Look Once: Unified, Real-Time Object Detection

## 前言

在R-CNN {{"girshick2014rich"|cite}}系列的论文中，目标检测被分成了候选区域提取和候选区域分类及精校两个阶段。不同于这些方法，YOLO将整个目标检测任务整合到一个回归网络中。对比Fast R-CNN {{"girshick2015fast"|cite}}提出的两步走的端到端方案，YOLO {{"redmon2016you"|cite}}的单阶段的使其是一个更彻底的端到端的算法（图1）。YOLO的检测过程分为三步：

1. 图像Resize到448\*448；
2. 将图片输入卷积网络；
3. NMS得到最终候选框。

###### 图1：YOLO算法框架

![](/assets/YOLOv1_1.png)

虽然在一些数据集上的表现不如Fast R-CNN及其后续算法，但是YOLO带来的最大提升便是检测速度的提升。在YOLO算法中，检测速度达到了45帧/秒，而一个更快速的Fast Yolo版本则达到了155帧/秒。另外在YOLO的背景检测错误率要低于Fast R-CNN。最后，YOLO算法具有更好的通用性，通过Pascal数据集训练得到的模型在艺术品问检测中得到了比Fast R-CNN更好的效果。

YOLO是可以用在Fast R-CNN中的，结合YOLO和Fast R-CNN两个算法，得到的效果比单Fast R-CNN要更好。

YOLO源码是使用DarkNet框架实现的，由于本人对DarkNet并不熟悉，所以这里我使用YOLO的[TensorFlow源码](https://github.com/nilboy/tensorflow-yolo)详细解析YOLO算法的每个技术细节和算发动机。

## YOL算法详解

YOLO检测速度远远超过R-CNN系列的重要原因是YOLO将整个物体检测统一成了一个回归问题。YOLO的输入是整张待检测图片，输出则是得到的检测结果，整个过程只经过一次卷积网络。Faster R-CNN {{"ren2015faster"|cite}}虽然使用全卷积的思想实现了候选区域的权值共享，但是每个候选区域的特征向量任然要单独的计算分类概率和bounding box。

YOLO实现统一检测的方法是增加网络的输出节点数量，其实也算是空间换时间的一种策略。在Faster R-CNN的Fast R-CNN部分，网络有分类和回归两个任务，网络输出节点个数是$$C+5$$，其中$$C$$是数据集的类别个数。而YOLO的输出层O节点个数达到了$$S\times S\times (C+B\times 5)$$，下面我们来讲解输出节点每个字符的含义。

### 1. YOLO输出层详解

#### 1.1 $$S\times S$$窗格

YOLO将输入图像分成$$S\times S$$的窗格（Grid），如果Ground Truth的中心落在某个单元（cell）内，则该单元责该物体的检测，如图2所示。

###### 图2：$$S\times S$$窗格

![](/assets/YOLOv1_2.png)

什么是某个单元负责落在该单元内的物体检测呢？举例说明一下，首先我们将输出层$$O_{S\times S \times (C+B \times 5)}$$看做一个三维矩阵，如果物体的中心落在第$$(i,j)$$个单元内，那么网络只优化一个$$C+B\times5$$维的向量，即向量$$O[i,j,:]$$。$$S$$是一个超参数，在源码中$$S=7$$，即配置文件`./yolo/config.py`的CELL\_SIZE变量。

```py
CELL_SIZE = 7
```

#### 1.2 Bounding Box

$$B$$是每个单元预测的bounding box的数量，$$B$$的个数同样是一个超参数。在`./yolo/config.py`文件中$$B=2$$，YOLO使用多个bounding box是为了每个cell计算top-B个可能的预测结果，这样做虽然牺牲了一些时间，但却提升了模型的检测精度。

```py
BOXES_PER_CELL = 2
```

注意不管YOLO使用了多少个bounding box，每个单元的bounding box均有相同的优化目标值。在`./yolo/yolo_net.py`中，Ground Truth的label值被复制了$$B$$次。每个bounding box要预测5个值：bounding box $$(x,y,w,h)$$以及置信度$$P$$。其中\(x,y\)是bounding box相对于每个cell中心的相对位置，$$(w,h)$$是物体相对于整幅图的尺寸。

###### 代码片段1：bounding box预处理

```py
boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
classes = labels[..., 5:]
offset = tf.reshape(
    tf.constant(self.offset, dtype=tf.float32),
    [1, self.cell_size, self.cell_size, self.boxes_per_cell])
offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
offset_tran = tf.transpose(offset, (0, 2, 1, 3))
...
boxes_tran = tf.stack(
    [boxes[..., 0] * self.cell_size - offset,
    boxes[..., 1] * self.cell_size - offset_tran,
    tf.sqrt(boxes[..., 2]),
    tf.sqrt(boxes[..., 3])], axis=-1)
```

labels需要往前追溯到Pascal voc文件的解析代码中，位于文件`./utils/pascal_voc.py`的139和145行

```py
boxes = [(x2 + x1) / 2.0, (y2 + y1) / 2.0, x2 - x1, y2 - y1]
...
label[y_ind, x_ind, 1:5] = boxes
```

置信度$$P$$表示bounding box中物体为待检测物体的概率以及bounding box对该物体的覆盖程度的乘积。所以$$P = Pr(Object) \times IOU_{pred}^{truth}$$。如果bounding box没有覆盖物体，$$P=0$$，否则$$P=IOU_{pred}^{truth}$$。

###### 代码片段2：bounding box的标签值处理

```py
predict_boxes = tf.reshape(
    predicts[:, self.boundary2:],
    [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

response = tf.reshape(labels[..., 0],[self.batch_size, self.cell_size, self.cell_size, 1])
...
predict_boxes_tran = tf.stack(
    [(predict_boxes[..., 0] + offset) / self.cell_size,
    (predict_boxes[..., 1] + offset_tran) / self.cell_size,
    tf.square(predict_boxes[..., 2]),
    tf.square(predict_boxes[..., 3])], axis=-1)

iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

# calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

# calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask
...
coord_mask = tf.expand_dims(object_mask, 4)
```

同时，YOLO也预测检测物体为某一类C的条件概率：$$Pr(Class_i|Object)$$. 对于每一个单元，YOLO值计算一个分类概率，而与$$B$$的值无关。在测试时，将条件概率$$Pr(Class_i|Object)$$乘以$$P$$便得到了每个为每一类的概率：

$$Pr(Class_i|Object)\times Pr(Object)\times IOU_{pred}^{truth}=Pr(Class_i) \times IOU_{pred}^{truth}$$

该部分代码在`./test.py`文件中：

```py
for i in range(self.boxes_per_cell):
    for j in range(self.num_class):
        probs[:, :, i, j] = np.multiply(class_probs[:, :, j], scales[:, :, i])
```

#### 1.3 类别C

不同于Faster R-CNN添加了背景类，YOLO仅使用了数据集提供的物体类别，在Pascal VOC中，待检测物体有20类，具体类别内容了列在了配置文件`./yolo/config.py`中

```py
CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']
```

对于输出层的两个超参数，$$S=7$$，$$B=2$$。则输出层的结构如图3所示。

###### 图3：YOLO的输出层

![](/assets/YOLOv1_3.png)

### 2. 输入层

YOLO作为一个统计检测算法，整幅图是直接输入网络的。因为检测需要更细粒度的图像特征，YOLO将图像Resize到了448\*448而不是物体分类中常用的224\*224的尺寸。resize在`./utils/pascal_voc.py`中，需要注意的是YOLO并没有采用VGG中先将图像等比例缩放再裁剪的形式，而是直接将图片非等比例resize。所以YOLO的输出图片的尺寸并不是标准比例的。

```py
image = cv2.resize(image, (self.image_size, self.image_size))
```

### 3. 骨干架构

YOLO使用了GoogLeNet作为骨干架构，但是使用了更少的参数，同时YOLO也不像GoogLeNet有3个输出层，图4。为了提高模型的精度，作者也使用了在ImageNet进行预训练的迁移学习策略。

###### 图4：YOLO的骨干架构

![](/assets/YOLOv1_4.png)

研究发现，在AlexNet中提出的ReLU存在Dead ReLU的问题，所谓Dead ReLU是指由于ReLU的x负数部分的导数永远为0，会导致一部分神经元永远不会被激活，从而一些参数永远不会被更新。

为了解决这个问题，Andrew NG团队提出了leaky ReLU，即在负数部分给与一个很小的梯度，leaky ReLU拥有ReLU的所有优点，但同时不会有Dead ReLU的问题。YOLO中的leaky ReLU（$$\phi(x)$$）表示为

$$\phi(x) = \begin{cases}
x & if x > 0 \\
0.1 \times x & otherwise
\end{cases}$$

```py
def leaky_relu(alpha):
    def op(inputs):
        return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
    return op
```

然而现在的一些文章指出leaky ReLU并不是那么理想，现在尝试网络超参数时ReLU依旧是首选。

### 4. 损失函数

YOLO的输出层包含标签种类决定了YOLO的损失函数必须是一个多任务的损失函数。根据1.2节的介绍我们已知YOLO的输出层包含分类信息，置信度$$P$$和bounding box的坐标信息$$(x,y,w,h)$$。我们先给出YOLO的损失函数的表达式再逐步解析损失函数这样设计的动机。

![](/assets/YOLOv1_6.png)

#### 4.1 noobj

根据图2和图4所示，YOLO的$$S\times S$$的窗格形式必然导致输出层含有大量的不包含物体的区域（也就是背景区域）。YOLO并不是直接将这一部分丢弃而是直接将其作为noobj一个分支进行优化。也是因为这个分支导致YOLO在检测背景时的错误率要比Fast R-CNN低近乎3倍。

#### 4.2 $$\lambda_{coord}$$和$$\lambda_{noobj}$$

YOLO并没有使用深度学习常用的均方误差（MSE）而是使用和方误差（SSE）作为损失函数，作者的解释是SSE更好优化。但是SSE作为损失函数时会使模型更倾向于优化输出向量长度更长的任务（也就是分类任务）。为了提升bounding box边界预测的优先级，该任务被赋予了一个超参数$$\lambda_{coord}$$，在论文中$$\lambda_{coord}=5$$。

作者在观察数据集时发现Pascal VOC中包含样本的单元要远远少于包含背景区域的单元，为了解决前/背景样本的样本不平衡的问题，作者给非样本区域的分类任务一个更小的权值$$\lambda_{noobj}$$，在论文中$$\lambda_{noobj}=0.5$$。

需要注意的是TF的源码（`./yolo/config.py`）使用的并不是论文和DarkNet源码中给出的超参数。对于损失函数的四个任务，坐标预测，前景预测，背景预测和分类预测的权值使用的权值分别是1，1，2，5。该值并不是非常重要，通常需要根据模型在验证集上的表现调整。

```py
OBJECT_SCALE = 1.0
NOOBJECT_SCALE = 1.0
CLASS_SCALE = 2.0
COORD_SCALE = 5.0
```

#### 4.3 $$I^{obj}_{i,j}$$，$$I^{obj}_{i}$$和$$I^{noobj}_{i,j}$$

根据1.1节的定义，当bounding box $$Box_{i,j}$$负责检测某个物体时，$$I^{obj}_{i,j}=1$$，否则$$I^{obj}_{i,j}=0$$。其中$$i$$用于遍历图像的单元，$$j$$用于遍历每个cell的bounding box。而$$I^{noobj}_{i,j}$$的定义则与$$I^{obj}_{i,j}$$相反。在1.2节中我们介绍分类是以单元为单位的而与bounding box无关，所以$$I^{obj}_{i}$$表示物体出现在$$Cell_{i}$$中。

$$I^{obj}_{i,j}$$，$$I^{obj}_{i}$$和$$I^{noobj}_{i,j}$$分别是代码片段2中的参数object\_mask，coord\_mask和noobject\_mask。由于使用了mask，当网络遇到一个正样本时，只有一个单元的权值被调整，这也就是1.1节说的”该单元负责该物体的检测“。

#### 4.4 $$\sqrt{\cdot}$$

最后还有一个小知识点，为了平衡短边和长边对损失函数的影响，YOLO使用了边长的平方根来减小长边的影响。

YOLO的损失函数见代码片段3

###### 代码片段3：YOLO的损失函数

```py
# class_loss
class_delta = response * (predict_classes - classes)
class_loss = tf.reduce_mean(
tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]),name='class_loss') * self.class_scale

# object_loss
object_delta = object_mask * (predict_scales - iou_predict_truth)
object_loss = tf.reduce_mean(
    tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), 
    name='object_loss') * self.object_scale

# noobject_loss
noobject_delta = noobject_mask * predict_scales
noobject_loss = tf.reduce_mean(
    tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), 
    name='noobject_loss') * self.noobject_scale

# coord_loss
coord_mask = tf.expand_dims(object_mask, 4)
boxes_delta = coord_mask * (predict_boxes - boxes_tran)
coord_loss = tf.reduce_mean(
    tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]),
    name='coord_loss') * self.coord_scale
```

### 5. 后处理

测试样本时，有些物体会被多个单元检测到，NMS用于解决这个问题。

## YOLO的优点

YOLO的快速我们已经一再重复，其性能的提升是因为YOLO统一检测框架的提出。同时YOLO在检测背景和通用性上表现也比Fast R-CNN要好。关于为什么YOLO比Fast R-CNN更擅长检测背景我们在4.1节进行了说明。从图5中我们可以看出YOLO的主要问题在于bounding box的精确检测。

* Correct: 正确分类且IoU&gt;0.5；
* Localization：正确分类且0.1&lt;IoU&lt;0.5
* Similar：类别近似且IoU&gt;0.1
* Other：分类错误且IoU&gt;0.1
* Background：IoU&lt;0.1的所有样本

###### 图5：Fast R-CNN vs YOLO

![](/assets/YOLOv1_5.png)

YOLO的论文中也指出YOLO的通用性更强，例如在人类画作的数据集上YOLO的表现要优于Fast R-CNN。但是为什么通用性更好至今我尚未想通，等待大神补充。

## YOLO的缺点

YOLO的缺点也非常明显，首先其精确检测的能力比不上Fast R-CNN更不要提和其更早提出的Faster R-CNN了。

YOLO的另外一个重要的问题是对小物体的检测，其为了提升速度而粗粒度划分单元而且每个单元的bounding box的功能过度重合导致模型的拟合能力有限，尤其是其很难覆盖到的小物体。YOLO检测小尺寸问题效果不好的另外一个原因是因为其只使用顶层的Feature Map，而顶层的Feature Map已经不会包含很多小尺寸的物体的特征了。

Faster R-CNN之后的算法均趋向于使用全卷积代替全连接层，但是YOLO依旧笨拙的使用了全连接不仅会使特征向量失去对于物体检测非常重要的位置信息，而且会产生大量的参数，影响算法的速度。

不过暂时不用着急，YOLO也在不断进化，YOLOv2，YOLOv3将会在速度和精度上进行优化，我们会在后续的文章中介绍。

