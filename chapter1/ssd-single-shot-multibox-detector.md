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

\[SSD\_1.png\]

从某个角度讲，SSD和RPN的相似度也非常高，网络结构都是全卷积，都是采用了锚点进行采样，不同之处有下面两点：

1. RPN只使用卷积网络的顶层特征，不过在FPN和Mask R-CNN中已经对这点进行了改进；
2. RPN是一个二分类任务（前/背景），而SSD是一个包含了物体类别的多分类任务。

在论文中作者说SSD的精度超过了Faster R-CNN，速度超过了YOLO。下面我们将结合基于TensorFlow的[源码](https://github.com/balancap/SSD-Tensorflow)和论文对SSD进行详细剖析。这里说明一下，这份源码使用了slim库，slim库是TensorFLow的一个高层封装，和keras的功能类似。

## SSD详解

### 1. 算法流程

SSD的流程和YOLO是一样的，输入一张图片得到一系列候选区域，使用NMS得到最终的检测框。与YOLO不同的是，SSD使用了不同阶段的Feature Map用于检测，YOLO和SSD的对比如图2所示。

###### 图1：SSD vs YOLO

\[SSD\_2.png\]

在详解SSD之前，我现在代码片段1中列出SSD的超参数，随后我们会在下面的章节中介绍这些超参数是如何使用的

代码片段1：SSD的超参数

```py
default_params = SSDParams(
    img_shape=(300, 300),
    num_classes=21,
    no_annotation_label=21,
    feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],
    feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],
    anchor_size_bounds=[0.15, 0.90],
    # anchor_size_bounds=[0.20, 0.90],
    anchor_sizes=[(21., 45.),
                  (45., 99.),
                  (99., 153.),
                  (153., 207.),
                  (207., 261.),
                  (261., 315.)],
    # anchor_sizes=[(30., 60.),
    #               (60., 111.),
    #               (111., 162.),
    #               (162., 213.),
    #               (213., 264.),
    #               (264., 315.)],
    anchor_ratios=[[2, .5],
                   [2, .5, 3, 1./3],
                   [2, .5, 3, 1./3],
                   [2, .5, 3, 1./3],
                   [2, .5],
                   [2, .5]],
    anchor_steps=[8, 16, 32, 64, 100, 300], ###???
    anchor_offset=0.5,
    normalizations=[20, -1, -1, -1, -1, -1],
    prior_scaling=[0.1, 0.1, 0.2, 0.2]
    )
```

#### 1.1 SSD的骨干架构

首先我们先看一下SSD的骨干网络的源码，再结合源码和图2我们来剖析SSD的算法细节。

###### 代码片段2：SSD骨干网络源码。

```py
with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse):
    # Original VGG-16 blocks.
    net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
    end_points['block1'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool1')
    # Block 2.
    net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
    end_points['block2'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool2')
    # Block 3.
    net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
    end_points['block3'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool3')
    # Block 4.
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
    end_points['block4'] = net
    net = slim.max_pool2d(net, [2, 2], scope='pool4')
    # Block 5.
    net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
    end_points['block5'] = net
    net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

    # Additional SSD blocks.
    # Block 6: let's dilate the hell out of it!
    net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6') #空洞卷积
    end_points['block6'] = net
    net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)
    # Block 7: 1x1 conv. Because the fuck.
    net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
    end_points['block7'] = net
    net = tf.layers.dropout(net, rate=dropout_keep_prob, training=is_training)

    # Block 8/9/10/11: 1x1 and 3x3 convolutions stride 2 (except lasts).
    end_point = 'block8'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
    end_points[end_point] = net
    end_point = 'block9'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = custom_layers.pad2d(net, pad=(1, 1))
        net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
    end_points[end_point] = net
    end_point = 'block10'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
    end_points[end_point] = net
    end_point = 'block11'
    with tf.variable_scope(end_point):
        net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
        net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
    end_points[end_point] = net

    # Prediction and localisations layers.
    predictions = []
    logits = []
    localisations = []
    for i, layer in enumerate(feat_layers):
        with tf.variable_scope(layer + '_box'):
            p, l = ssd_multibox_layer(end_points[layer],
                                      num_classes,
                                      anchor_sizes[i],
                                      anchor_ratios[i],
                                      normalizations[i])
        predictions.append(prediction_fn(p))
        logits.append(p)
        localisations.append(l)

    return predictions, localisations, logits, end_points
```

从图1中我们可以看出，SSD输入图片的尺寸是300\*300，另外SSD也由一个输入图片尺寸是512\*512的版本，这个版本的SSD虽然慢一些，但是是检测精度达到了76.9%。

SSD采用的是VGG-16的作为骨干网络，VGG的详细内容参考文章[Very Deep Convolutional NetWorks for Large-Scale Image Recognition](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/very-deep-convolutional-networks-for-large-scale-image-recognition.html)。在VGG的卷积部分之后，全连接被换成了卷机操作，在block6的卷积含有一个参数`rate=6`。此时的卷积操作为空洞卷积（Dilation Convolution）\[3\]，在TensorFLow中使用`tf.nn.atrous_conv2d()`调用。

空洞卷积可以在不增加模型复杂度的同时扩大卷积操作的视野，通过在卷积核中插值0的形式完成的。如图3所示，\(a\)是膨胀率为1的卷积，也就是标准的卷积，其感受野的大小是3\*3。\(b\)的膨胀率为2，卷积核变成了7\*7的卷积核，其中只有9个红点处的值不为0，在不增加复杂度的同时感受野变成了7\*7。\(c\)的膨胀率是4，感受野的大小变成了15\*15。在设置感受野的膨胀率时要谨慎设计，否则如果卷积核大于Feature Map的尺寸之后程序会报错。

## Reference

\[1\] Liu W, Anguelov D, Erhan D, et al. Ssd: Single shot multibox detector\[C\]//European conference on computer vision. Springer, Cham, 2016: 21-37.

\[2\] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection\[C\]//Proceedings of the IEEE conference on computer vision and pattern recognition. 2016: 779-788.

\[3\] Liu,W.,Rabinovich,A.,Berg,A.C.:ParseNet:Lookingwidertoseebetter.In:ILCR.\(2016\)Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions." arXiv preprint arXiv:1511.07122

