# SSD: Single Shot MultiBox Detector

## 前言

在YOLO\[2\]的文章中我们介绍到YOLO存在三个缺陷：

1. 两个bounding box功能的重复降低了模型的精度；
2. 全连接层的使用不仅使特征向量失去了位置信息，还产生了大量的参数，影响了算法的速度；
3. 只使用顶层的特征向量使算法对于小尺寸物体的检测效果很差。

为了解决这些问题，SSD应运而生。SSD的全称是Single Shot MultiBox Detector，Single Shot表示SSD是像YOLO一样的单次检测算法，MultiBox指SSD每次可以检测多个物体，Detector表示SSD是用来进行物体检测的。

针对YOLO的三个问题，SSD做出的改进如下：

1. 使用了类似Faster R-CNN中RPN网络提出的锚点（Anchor）机制，增加了bounding box的多样性；
2. 使用全卷积的网络结构，提升了SSD的速度；
3. 使用网络中多个阶段的Feature Map，提升了特征多样性。

SSD的算法如图1。

###### 图1：SSD算法流程

![](/assets/SSD_1.png)

从某个角度讲，SSD和RPN的相似度也非常高，网络结构都是全卷积，都是采用了锚点进行采样，不同之处有下面两点：

1. RPN只使用卷积网络的顶层特征，不过在FPN和Mask R-CNN中已经对这点进行了改进；
2. RPN是一个二分类任务（前/背景），而SSD是一个包含了物体类别的多分类任务。

在论文中作者说SSD的精度超过了Faster R-CNN，速度超过了YOLO。下面我们将结合基于Keras的[源码](https://github.com/pierluigiferrari/ssd_keras)和论文对SSD进行详细剖析。

## SSD详解

### 1. 算法流程

SSD的流程和YOLO是一样的，输入一张图片得到一系列候选区域，使用NMS得到最终的检测框。与YOLO不同的是，SSD使用了不同阶段的Feature Map用于检测，YOLO和SSD的对比如图2所示。

###### 图1：SSD vs YOLO

![](/assets/SSD_2.png)

在详解SSD之前，我先在代码片段1中列出SSD的超参数（`./models/keras_ssd300.py`），随后我们会在下面的章节中介绍这些超参数是如何使用的。

###### 代码片段1：SSD的超参数

```py
def ssd_300(image_size,
            n_classes,
            mode='training',
            l2_regularization=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            subtract_mean=[123, 117, 104],
            divide_by_stddev=None,
            swap_channels=[2, 1, 0],
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False)
```

#### 1.1 SSD的骨干网络

首先我们先看一下SSD的骨干网络的源码（`./models/keras_ssd300.py`），再结合源码和图2我们来剖析SSD的算法细节。

###### 代码片段2：SSD骨干网络源码。注意源码中的变量名称和图2不一样，我在代码中进行了更正。

```py
conv1_1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_1')(x1)
conv1_2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1_2')(conv1_1)
pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool1')(conv1_2)

conv2_1 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_1')(pool1)
conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2_2')(conv2_1)
pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool2')(conv2_2)

conv3_1 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_1')(pool2)
conv3_2 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_2')(conv3_1)
conv3_3 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3_3')(conv3_2)
pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool3')(conv3_3)

conv4_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_1')(pool3)
conv4_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_2')(conv4_1)
conv4_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3')(conv4_2)
pool4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name='pool4')(conv4_3)

conv5_1 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_1')(pool4)
conv5_2 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_2')(conv5_1)
conv5_3 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5_3')(conv5_2)
pool5 = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same', name='pool5')(conv5_3)

fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc6')(pool5)

fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7')(fc6)

conv8_1 = Conv2D(256, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_1')(fc7)
conv8_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv8_1)
conv8_2 = Conv2D(512, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6_2')(conv8_1)

conv9_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_1')(conv8_2)
conv9_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv9_1)
conv9_2 = Conv2D(256, (3, 3), strides=(2, 2), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7_2')(conv9_1)

conv10_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_1')(conv9_2)
conv10_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2')(conv10_1)

conv11_1 = Conv2D(128, (1, 1), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_1')(conv10_2)
conv11_2 = Conv2D(256, (3, 3), strides=(1, 1), activation='relu', padding='valid', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2')(conv11_1)
```

从图1中我们可以看出，SSD输入图片的尺寸是$$300\times 300$$，另外SSD也由一个输入图片尺寸是$$512\times 512$$的版本，这个版本的SSD虽然慢一些，但是是检测精度达到了76.9%。

SSD采用的是VGG-16的作为骨干网络，VGG的详细内容参考文章[Very Deep Convolutional NetWorks for Large-Scale Image Recognition](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/very-deep-convolutional-networks-for-large-scale-image-recognition.html)。使用标准网络的目的是为了使用训练好的模型进行迁移学习，SSD使用的是在ILSVRC CLS-LOC数据集上得到的模型进行的初始化。目的是在更高的采样率上计算Feature Map。

第一点不同的是在block5中，max\_pool2d的步长$$stride=1$$，此时图像将不会进行降采样，也就是说输入到block6的Feature Map的尺寸任然是$$38\times 38$$。

SSD的$$3\times 3$$的conv6和$$1\times 1$$的conv7的卷积核是通过预训练模型的fc6和fc7采样得到，这种从全连接层中采样卷积核的方法参考的是DeepLab-LargeFov \[4\]的方法。具体细节在DeepLab-LargeFov的论文中进行分析。

在VGG的卷积部分之后，全连接被换成了卷机操作，在block6的卷积含有一个参数`rate=6`。此时的卷积操作为空洞卷积（Dilation Convolution）\[3\]，在TensorFLow中使用`tf.nn.atrous_conv2d()`调用。

空洞卷积可以在不增加模型复杂度的同时扩大卷积操作的视野，通过在卷积核中插值0的形式完成的。如图3所示，\(a\)是膨胀率为1的卷积，也就是标准的卷积，其感受野的大小是$$3\times 3$$。\(b\)的膨胀率为2，卷积核变成了$$7\times 7$$的卷积核，其中只有9个红点处的值不为0，在不增加复杂度的同时感受野变成了$$7\times 7$$。\(c\)的膨胀率是4，感受野的大小变成了$$15\times 15$$。在设置感受野的膨胀率时要谨慎设计，否则如果卷积核大于Feature Map的尺寸之后程序会报错。

###### 图3：空洞卷积示例图

![](/assets/SSD_3.png)

fc7之后输出的Feature Map的大小是$$19\times 19$$，经过block8的一次padding和一次valid卷积之后（即相当于一次same卷积），再经过一次步长为2的降采样，输入到block 9的Feature Map的尺寸是$$10\times 10$$。block 9的操作和block 8相同，即输入到block 8的Feature Map的尺寸是$$5\times 5$$。block 10和block 11使用的是valid卷积，所以图像的尺寸分别是3和1。这样我们便得到了图2中Feature Map尺寸的变化过程。

#### 1.2 多尺度预测

在卷积网络中，不同深度的Feature Map趋向于响应不同程度的特征，SDD使用了骨干网络中的多个Feature Map用于预测检测框。通过图1和图2我们可以发现，SSD使用的是conv4\_3, fc7, conv8\_2, conv9\_2, conv10\_2, conv11\_2分别用于检测尺寸从小到大的物体,如代码片段3 （`./models/keras_ssd300.py`）。

###### 代码片段3：SSD使用全卷积预测检测框

```py
# Feed conv4_3 into the L2 normalization layer
conv4_3_norm = L2Normalization(gamma_init=20, name='conv4_3_norm')(conv4_3)

### Build the convolutional predictor layers on top of the base network

# We precidt `n_classes` confidence values for each box, hence the confidence predictors have depth `n_boxes * n_classes`
# Output shape of the confidence layers: `(batch, height, width, n_boxes * n_classes)`
conv4_3_norm_mbox_conf = Conv2D(n_boxes[0] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_conf')(conv4_3_norm)
fc7_mbox_conf = Conv2D(n_boxes[1] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_conf')(fc7)
conv8_2_mbox_conf = Conv2D(n_boxes[2] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_conf')(conv8_2)
conv9_2_mbox_conf = Conv2D(n_boxes[3] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_conf')(conv9_2)
conv10_2_mbox_conf = Conv2D(n_boxes[4] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv10_2_mbox_conf')(conv10_2)
conv11_2_mbox_conf = Conv2D(n_boxes[5] * n_classes, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_2_mbox_conf')(conv11_2)
# We predict 4 box coordinates for each box, hence the localization predictors have depth `n_boxes * 4`
# Output shape of the localization layers: `(batch, height, width, n_boxes * 4)`
conv4_3_norm_mbox_loc = Conv2D(n_boxes[0] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4_3_norm_mbox_loc')(conv4_3_norm)
fc7_mbox_loc = Conv2D(n_boxes[1] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='fc7_mbox_loc')(fc7)
conv8_2_mbox_loc = Conv2D(n_boxes[2] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8_2_mbox_loc')(conv8_2)
conv9_2_mbox_loc = Conv2D(n_boxes[3] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9_2_mbox_loc')(conv9_2)
conv10_2_mbox_loc = Conv2D(n_boxes[4] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv10_2_mbox_loc')(conv10_2)
conv11_2_mbox_loc = Conv2D(n_boxes[5] * 4, (3, 3), padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv11_2_mbox_loc')(conv11_2)
```

其中第二行的L2Normalization使用的是ParseNet \[5\]中提出的全局归一化。即对像素点的在通道维度上进行归一化，其中gamma是一个可训练的放缩变量。

SSD对于第i个Feature Map的每个像素点都会产生n\_boxes\[i\]个锚点进行分类和位置精校，其中n\_boxes的值为\[4,6,6,6,4,4\]，我们在1.3节会介绍n\_boxes值的计算方法。SSD相当于预测M个bounding box，其中：

$$
M = 38\times 38\times 4 + 19\times 19\times 6 + 10\times 10\times 6 + 5\times 5\times 6+ 3\times 3\times 4 +1\times 1\times 4=8732。
$$

上式便是图2中最右侧8732的计算方式。也就是对于一张300\*300的输入图片，SSD要预测8732个检测框，所以SSD本质上可以看做是密集采样。SSD的分类有C+1个值包括C类前景和1类背景，回归包括物体位置的四要素\(y,x,h,w\)。对于20类的Pascal VOC来说SSD是一个含有8732\*\(21+4\)的多任务模型。

通过代码片段3，我们可以看出SSD并没有使用全连接产生预测结果，而是使用的3\*3的卷机操作分别产生了分类和回归的预测结果。对于一个分类任务来说，Feature Map的数量是\(C+1\)\*n\_boxes\[i\]，而回归任务的Feature Map的数量是4\*n\_boxes\[i\]。

#### 1.3 SSD中的锚点

在1.2节中，我们介绍了SSD的n\_boxes=\[4,6,6,6,4,4\]，下面我们就来详细解析SSD锚点是什么样子的。

SSD使用多尺度的Feature Map的原因是使用不同层次的Feature Map检测不同尺寸的物体，所以onv4\_3, fc7, conv8\_2, conv9\_2, conv10\_2, conv11\_2的锚点的尺寸也是从小到大。论文中给出的值是从0.2到0.9间一个线性变化的值：

$$
s_k = s_{min} + \frac{s_{max} - s_{min}}{m-1}(k-1)，k\in[1,m]
$$

$$s\_{min}$$和$$s\_{max}$$是两个超参数，需要根据不同的数据集自行调整。论文中给出的例子是$$s\_{min}=0.2$$，$$s\_{max}=0.9$$，$$m=6$$。$$s\_k$$表示的是锚点大小相对于Feature Map的比例，通过上式得出的值依次是\[0.2, 0.34, 0.48, 0.62, 0.76, 0.9\]。

对于6组Feature Map，SSD分别产生\[4,6,6,6,4,4\]个不同比例的锚点。锚点的比例是超参数`aspect_ratios_per_layer`中给出的值加上一组比例为$$s'\_k=\sqrt{s\_k s\_{k+1}}$$的框，其中$$s\_{k+1} = s\_k + \(s\_k - s\_{k-1}\)$$。根据$$s\_k$$和长宽比$$a\_r$$我们便可以得到不同样式的锚点，其中锚点的宽$$w^a\_k = s\_k\sqrt{a\_r}$$，高$$h^a\_k = s\_k/\sqrt{a\_r}$$。$$a\_r \in {1,2,3,\frac{1}{2},\frac{1}{3}}$$。

$$a\_r$$的取值也是一个超参数，在源码中，定义在`aspect_ratios_per_layer`中。根据a`spect_ratios_per_layer`的变量个数，我们便可以得到n\_boxes的值。

举个例子，在conv4\_3中，要产生$$38\times 38\times 4$$个锚点，其中有三个锚点的尺度分别是（1, 2.0, 0.5），再加上一组1:1的尺度为$$s'\_k=\sqrt{0.2\*0.34} = 0.2608$$的锚点，得到四组锚点分别是$$\[\(0.2,0.2\), \(0.2608, 0.2608\), \(0.2828, 0.1414\), \(0.1414, 0.2828\)\]$$。等比例换算到原图中得到的锚点的大小（取整）为$$\[\(60, 60\), \(78, 78\), \(85, 42\), \(42, 85\)\]$$。

通过上面的介绍，我们得到了锚点四要素中的$$w$$和$$h$$，锚点的$$x$$, $$y$$通过下式得到

$$
(x,y) = (\frac{i+0.5}{|f_k|}, \frac{j+0.5}{|f_k|}), i,j\in [0, |f_k|]
$$

$$i,j$$即Feature Map像素点的坐标，$$f\_k$$是Feature Map的尺寸。图4便是在$$8\times 8$$和$$4\times 4$$的Feature Map上得到不同尺度的锚点的示例。

###### 图4：锚点示例，改图也展示了锚点对Ground Truth的响应。

\[SSD\_4.png\]

锚点如何设计是一种见仁见智的方案，例如源码中锚点的尺度便和论文不同，在源码中，尺度定义在jupyter notebook 文件`./ssd300_training.ipynb`中。关于具体如何定义这些锚点其实不必太过在意，这些锚点的作用是为检测框提供一个先验假设，网络最后输出的候选框还是要经过Ground Truth纠正的。

```py
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
```

除了锚点的尺度以外，源码中锚点的中心点的实现也和论文不同。源码使用预先计算好的步长加上位移进行预测的，即超参数中的变量`steps=[8, 16, 32, 64, 100, 300]`。conv4\_3经过了3次降采样，即Feature Map的一步相当于原图的8步。但是对于这种方案存在一个问题，即75降采样到38时是不能整除的，也就是最后一列并没有参加降采样，这样步长非精确的计算经过多次累积会被放大到很大。例如经过源码中步长为64的conv9\_2层的最后一行和最后一列的锚点的中心点将会取到图像之外，有兴趣的读者可以打印一下。

知道锚点的四要素（x,y,w,h）之后，我们需要确定锚点的分类标签。SSD与RPN相比，正锚点的选取规则则要简单很多，只要锚点和Ground Truth的IoU（即论文中的jaccard overlap）大于某个阈值时，该锚点便被视为正锚点。

源码中，锚点是在keras\_layers/keras\_layer\_AnchorBoxes中实现的，通过AnchorBoxes函数调用。网络中的6个Feature Map会产生6组共8732个先验box，如代码片段4所示。

###### 代码片段4：计算先验box

```py
# Output shape of anchors: `(batch, height, width, n_boxes, 8)`
conv4_3_norm_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                                         two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0], clip_boxes=clip_boxes,
                                         variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv4_3_norm_mbox_priorbox')(conv4_3_norm_mbox_loc)
fc7_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                                two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1], clip_boxes=clip_boxes,
                                variances=variances, coords=coords, normalize_coords=normalize_coords, name='fc7_mbox_priorbox')(fc7_mbox_loc)
conv6_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2], clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv6_2_mbox_priorbox')(conv6_2_mbox_loc)
conv7_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3], clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv7_2_mbox_priorbox')(conv7_2_mbox_loc)
conv8_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios[4],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[4], this_offsets=offsets[4], clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv8_2_mbox_priorbox')(conv8_2_mbox_loc)
conv9_2_mbox_priorbox = AnchorBoxes(img_height, img_width, this_scale=scales[5], next_scale=scales[6], aspect_ratios=aspect_ratios[5],
                                    two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[5], this_offsets=offsets[5], clip_boxes=clip_boxes,
                                    variances=variances, coords=coords, normalize_coords=normalize_coords, name='conv9_2_mbox_priorbox')(conv9_2_mbox_loc)
```

#### 1.4 SSD的匹配准则

从Feature Map得到锚点之后，我们要确定Ground Truth和哪个锚点匹配，与之匹配的锚点将负责该Ground Truth的预测。在YOLO中，Ground Truth的中心点落在哪个单元内，则该单元的bounding box负责预测其准确的边界。SSD的锚点匹配采用了‘bipartite’和‘multi’两种策略，匹配源码位于`./ssd_encoder_decoder/`目录下面。
在bipartite模式中，每个Ground Truth选择与其IoU（论文用的是Jaccard Overlap）最大的锚点进行匹配.如果一个锚点被多个Ground Truth匹配，那么该锚点只匹配与其IoU最大的Ground Truth，其它Ground Truth从剩下的锚点中选择Iou最大的那个进行匹配。bipartite可以保证每个Ground Truth都会有唯一的一个锚点进行匹配。bipartite的源码见代码片段5。

###### 代码片段5：bipartite匹配
```py
def match_bipartite_greedy(weight_matrix):
    '''
    Parameters:
        weight_matrix (array): A 2D Numpy array that represents the weight matrix
            for the matching process. If `(m,n)` is the shape of the weight matrix,
            it must be `m <= n`. The weights can be integers or floating point
            numbers. The matching process will maximize, i.e. larger weights are
            preferred over smaller weights.

    Returns:
        A 1D Numpy array of length `weight_matrix.shape[0]` that represents
        the matched index along the second axis of `weight_matrix` for each index
        along the first axis.
    '''
    weight_matrix = np.copy(weight_matrix)
    num_ground_truth_boxes = weight_matrix.shape[0]
    all_gt_indices = list(range(num_ground_truth_boxes)) 
    matches = np.zeros(num_ground_truth_boxes, dtype=np.int)
    for _ in range(num_ground_truth_boxes):
        # Find the maximal anchor-ground truth pair in two steps: First, reduce
        # over the anchor boxes and then reduce over the ground truth boxes.
        anchor_indices = np.argmax(weight_matrix, axis=1) # Reduce along the anchor box axis.
        overlaps = weight_matrix[all_gt_indices, anchor_indices]
        ground_truth_index = np.argmax(overlaps) # Reduce along the ground truth box axis.
        anchor_index = anchor_indices[ground_truth_index]
        matches[ground_truth_index] = anchor_index # Set the match.

        # Set the row of the matched ground truth box and the column of the matched
        # anchor box to all zeros. This ensures that those boxes will not be matched again,
        # because they will never be the best matches for any other boxes.
        weight_matrix[ground_truth_index] = 0
        weight_matrix[:,anchor_index] = 0

    return matches
```
在bipartite策略中被匹配的锚点数量是非常少的，这就造成了训练时的正负样本的不平衡。所以需要multi策略进行纠正，源码中也是使用的multi策略。mutli在bipartite策略的基础上增加了所有与Ground Truth的IoU大于阈值$$\theta$$（源码中$$\theta=0.5$$）的锚点作为匹配锚点。SSD中一个Ground Truth是可以有多个锚点与其匹配的，但是反过来是不行的，一个锚点只能与和它IoU最大的Ground Truth进行匹配。mutli策略的源码见代码片段6

###### 代码片段6：multi匹配
```py
def match_multi(weight_matrix, threshold):
    '''
    Returns:
        Two 1D Numpy arrays of equal length that represent the matched indices. The first
        array contains the indices along the first axis of `weight_matrix`, the second array
        contains the indices along the second axis.
    '''
    num_anchor_boxes = weight_matrix.shape[1]
    all_anchor_indices = list(range(num_anchor_boxes))
    # Find the best ground truth match for every anchor box.
    ground_truth_indices = np.argmax(weight_matrix, axis=0)
    overlaps = weight_matrix[ground_truth_indices, all_anchor_indices]

    # Filter out the matches with a weight below the threshold.
    anchor_indices_thresh_met = np.nonzero(overlaps >= threshold)[0]
    gt_indices_thresh_met = ground_truth_indices[anchor_indices_thresh_met]

    return gt_indices_thresh_met, anchor_indices_thresh_met
```

尽管通过multi匹配策略增加了正样本的数量，但是在8732个锚点中，正负样本的比例还是非常不均衡的。所以SSD使用了难分样本挖掘（Hard Negative Mining）的策略对负样本进行采样。即对负样本的置信度进行排序，在保证正负样本$$1:3$$的的前提下抽取top-k个负样本。

#### SSD的损失函数

由于SSD也是一个由分类任务和检测任务多任务模型，所以SSD的损失函数将由置信度误差$L_{conf}$和位置误差$L_{loc}$组成:

$$L(x,c,l,g) = \frac{1}{N} (L_{conf}(x, c) + \alpha L_{loc}(x,l,g))$$

其中$$N$$是正锚点的数量，$$\alpha$$是两个任务的侧重比重，经过交叉验证之后$$\alpha$$被设置成了1。$$x_{i,j}^p =\{0,1\}\in x$$用于指示该锚点是否和Ground Truth进行了匹配。

对于分类任务，SSD使用的是softmax多类别的损失函数，上式中的$$c$$表示分类置信度：

$$
L_{conf}(x,c) = - \sum^{N}_{i\in Pos} x^p_{i,j}log(\hat{c}^p_i) - \sum_{i\in Neg} log(\hat{c}^0_i), \hat{c}^p_i=\frac{exp(c^p_i)}{\sum_p exp(c^p_i)}
$$

对于回归任务，SSD预测的是正锚点和Ground Truth的相对位移，损失函数使用的是Smooth L1损失函数。$$l$$表示预测的锚点和Ground Truth的相对位移，而$$g$$表示实际的相对位移。其中l和g包含物体位置的四要素$$(\hat{g}^{cx}_j, \hat{g}^{cy}_j, \hat{g}^w_j, \hat{g}^h_j)$$。
$$\hat{g}^{cx}_j = (g^{cx}_j - d^{cx}_i)/d^w_i$$
$$\hat{g}^{cy}_j = (g^{cy}_j - d^{cy}_i)/d^h_i$$
$$\hat{g}^w_j = log(\frac{g^w_j}{d^w_i})$$
$$\hat{g}^h_j = log(\frac{g^h_j}{d^h_i})$$

损失函数表示为实际偏移和预测偏移的Smooth L1损失：

$$
L_{loc}(x,l,g) = - \sum^{N}_{i\in Pos} \sum_{m \in {cx,cy,w,h}} x^k_{i,j} smooth_{L1} (l^m_i - \hat{g}^m_j)
$$

## Reference

\[1\] Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector\[C\]//European conference on computer vision. Springer, Cham, 2016: 21-37.

\[2\] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

\[3\] Liu,W.,Rabinovich,A.,Berg,A.C.:ParseNet:Lookingwidertoseebetter.In:ILCR.\(2016\)Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions." arXiv preprint arXiv:1511.07122

\[4\] Chen, L.C., Papandreou, G., Kokkinos, I., Murphy, K., Yuille, A.L.: Semantic image segmentation with deep convolutional nets and fully connected crfs. In: ICLR. \(2015\)

\[5\] Liu,W.,Rabinovich,A.,Berg,A.C.:ParseNet:Looking wider to see better.In:ILCR.\(2016\)

