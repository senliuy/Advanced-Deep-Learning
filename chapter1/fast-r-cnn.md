# Fast R-CNN

## 简介

在之前介绍的R-CNN {{"girshick2014rich"|cite}}和SPP-net {{"he2015spatial"|cite}} 中，它们都是先通过卷积网络提取特征，然后根据特征训练SVM做分类和回归器用于位置矫正。这种多阶段的流程有两个问题

1. 保存中间变量需要使用大量的硬盘存储空间
2. 不能根据分类和矫正结果调整卷积网络权值，会一定程度的限制网络精度。

作者通过多任务的方式将R-CNN和SPP-net整合成一个流程，同时也带来分类和检测精度的提升吗，通过Softmax替代SVM的分类任务省去了中间变量的使用。Fast R-CNN的代码也是非常优秀的一份代码，强烈推荐参考学习：[https://github.com/rbgirshick/fast-rcnn。](https://github.com/rbgirshick/fast-rcnn。)

同SPP-net一样，Fast R-CNN {{"girshick2015fast"|cite}}将整张图像输入到卷积网络用语提取特征，将Selective Search选定的候选区域坐标映射到卷积层。使用ROI 池化层 \(单尺度的SPP层\)将不同尺寸的候选区域特征窗口映射成相同尺寸的特征向量。经过两层全连接后将得到的特征分支成两个输出层，一个N+1类的softmax用于分类，一个bbox 回归器用于位置精校。这两个任务的损失共同用于调整网络的参数。

和SPP-net对比，fast R-CNN最大的优点是多任务的引进，在优化训练过程的同时也避免了额外存储空间的使用并在一定程度上提升了精度。

## 算法详解

### 1. 数据准备

#### 1.1 候选区域选择

数据准备工作集中在`/lib/datasets` 目录下面，下面会着重介绍几个重要部分

通过Selective Search[^1]选取候选区域，在5.5中论文指出，随着候选区域的增多，mAP呈先上升后下降的趋势。在fast-rcnn的源码中，作者选取了2000个候选区域。

```py
# PASCAL specific config options
self.config = {'cleanup'  : True,
               'use_salt' : True,
               'top_k'    : 2000}
```

#### 1.2 输入图片尺度

作者在5.2中讨论了输入图片的尺寸问题，通过对比多尺度 {480，576，688，864，1200}和单尺度的精度，作者发现单尺度和多尺度的精度差距并不明显。这也从一个角度证明了深度卷积网络有能力直接学习到输入图片的尺寸不变性。

尺度选项可以在`lib/fast-rcnn/config.py`里面设计，如下面代码，scales可以为单值或多个值。

```
# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)
```

综合时间，精度等各种因素，作者在实验中使用了最小边长600，最大边长不超过1000的resize图像方法，通过下面函数实现。

```py
def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale
```

#### 1.3 图像扩充

数据扩充往往对深度卷积网络的训练能够起到正面作用，尤其是在数据量不足的情况下。在实验中，作者仅使用了反转图片的这种扩充方式，在`/lib/fast-rcnn-train`里面调用了反转图片的函数。

```py
if cfg.TRAIN.USE_FLIPPED:
    print 'Appending horizontally-flipped training examples...'
    imdb.append_flipped_images()模拟 
    print 'done'
```

### 2. 模型训练

#### 2.1 网络结构

Fast-RCNN选择了VGG-16的卷积网络结构，并将最后一层的max pooling换成了roi pooling。经过两层全连接和Dropout后，接了一个双任务的loss，分别用于分类和位置精校。

###### 图1： Fast-RCNN算法流程![](/assets/Fast-RCNN_1.png)

#### 2.2 ROI pooling 层

ROI Pooling是一层的SPP-net。由$$(r,c,w,h)$$定义，$$(r,c)$$表示候选区域的左上角，$$(w,h)$$表示候选区域的高和宽。假设我们要将特征层映射到$$H*W$$的矩阵。ROI pooling通过将特征层分成h/H \* w/W的窗格，每个窗格进行max pooling得到，在作者的实验中$$W=H=7$$。ROI pooling layer是在caffe源码中`src/caffe/layers/roi_pooling_layer.cpp`通过C++语言实现的。

既然ROI pooling层是自己定义的，当然我们也应向caffe中定义其它层一样，给出ROI pooling层的反向的计算过程。在Fast-RCNN中，ROI 使用的是固定grid数量的max pooling，在调参时，只对grid中选为最大值的像素点更新参数。可以表示为


$$
\frac{\partial L}{\partial x_i} = \sum_r \sum_j [i = i^*(r,j)]\frac{\partial L}{\partial y_{r,j}}
$$


selective search是在输入图像上完成的，由于所有的候选区域会共享计算，即对整张图进行卷积操作，然后将SS选择的候选区域映射到conv5特征层，最后在每一个候选区域上做ROI Pooling，如下图。

![](/assets/Fast-RCNN_2.png)

所以存在一个图像到conv5候选区域的映射过程，在Fast R-CNN源码中通过**卷积后，图像的相对位置不变这一特征**完成的。在Fast R-CNN使用的VGG网络中，通过max pooling做了4次stride=2的降采样，而VGG的卷积都是same卷积（卷积后图像的尺寸不变），所以特征图的尺寸变成了原来的1/16=0.625，在ROI pooling层中，spatial\_ratio便是记录的这个数据。

```
layer {
  name: "roi_pool5"
  type: "ROIPooling"
  bottom: "conv5_3"
  bottom: "rois"
  top: "pool5"
  roi_pooling_param {
    pooled_w: 7
    pooled_h: 7
    spatial_scale: 0.0625 # 1/16
  }
}
```

原图的候选区域$$(x_1,y_1,x_2,x_2)$$对应的特征图的区域$$(x_1', y_1', x_2', y_2')$$是:

$$x_1' = round(x_1 \times  spatial\_scale)$$

$$y_1' = round(y_1 \times  spatial\_scale)$$

$$x_2' = round(x_2 \times  spatial\_scale)$$

$$y_2' = round(y_2 \times spatial\_scale)$$

#### 2.3 多任务

Fast-RCNN最重要的贡献是多任务模型的提出，多任务将原来物体检测的多阶段训练简化成一个端到端（end-to-end）的模型。Fast-RCNN有两个任务组成，一个任务是用来对候选区域进行分类，另外一个回归器是用来矫正候选区域的位置。

##### 2.3.1 分类任务$$L_{cls}$$

设$$p=\{p_0, p_1, ..., p_n\}$$_是候选区域集合，则_$$L_{cls}$$是一个K+1类的分类任务，其中输入数据是经过卷积和全连接之后提取的特征向量，输出数据是候选区域的类别（$$u$$），包括K类物体\($$u\geq 1$$\)和1类背景\($$u=0$$\)。分类任务的损失函数是softmax损失。

2.3.2 位置精校任务$$L_{loc}$$

##### 对于候选区域所属的类别u，$$v=\{v_x, v_y, v_w, v_h\}$$表示候选区域的ground-truth, $$t^u = (t^u_x, t^u_y, t^u_w, t^u_h)$$表示对候选区域的类别u（$$u\geq1$$）预测的位置。损失函数是smooth L1损失，表示为


$$
L_{loc}(t^u, v) = \sum_{i \in {x,y,w,h}}smooth_{L_1}(t^u_i, v_i)
$$


其中


$$
s
mooth_{L_1}(x) = \begin{cases}
0.5 x^2 & if |x|<1 \\
|x| - 0.5 & otherwise
\end{cases}
$$


smooth L1的形状类似于二次曲线。

smooth L1同样以layer的形式定义在了caffe的源码中 `/src/caffe/layers/smooth_L1_loss_layer.cpp`

###### 图2：smooth L1 曲线

![](/assets/Fast-RCNN_3.png)

##### 2.3.3 多任务

多任务学习是由两个损失函数和权重$$\lambda$$组成，表示为


$$
L(p,u,t^u,v) = L_{cls}(p,u) + \lambda [u\leq 1]L_{loc}(t^u, v)
$$


在实验中，作者将$$\lambda$$统一设成了1。在模型文件中，定义了这两个损失

```
layer {
  name: "loss_cls"
  type: "SoftmaxWithLoss"
  bottom: "cls_score"
  bottom: "labels"
  top: "loss_cls"
  loss_weight: 1
}
layer {
  name: "loss_bbox"
  type: "SmoothL1Loss"
  bottom: "bbox_pred"
  bottom: "bbox_targets"
  bottom: "bbox_loss_weights"
  top: "loss_bbox"
  loss_weight: 1
}
```

**根据笔者的经验，训练Fast-RCNN时根据两个损失函数的收敛情况适当的调整权值能得到更好的结果。调整的经验是给收敛更慢的那个任务更大的比重。**

#### 2.4 SGD训练详解

##### 2.4.1 迁移学习

同Fast-RCNN一样，作者同样使用ImageNet的数据对模型进行了预训练。详细的讲，首先使用1000类的ImageNet训练一个1000类的分类器，如图2的虚线部分。然后提取模型中的特征层以及其以前的所有网络，使用Fast-RCNN的多任务模型训练网络，即图2所有的实线部分。

##### 2.4.2 Minibatch training

在Fast-RCNN中，设每个batch的大小是R。在抽样时，每次随机选择N个图片，每张图片中随机选择R/N个候选区域，在实验中N=2，R=128。对候选区域进行抽样时，选取25%的正样本（和ground truth的IoU大于0.5），75%的负样本。

### 3. 物体检测

使用selective search输入图像中提取2000个候选区域，按照同训练样本相同的resize方法调整候选区域的大小。将所有的候选区域输入到训练好的神经网络，得到每一类的后验概率p和相对偏移r。通过预测概率给每一类一个置信度，并使用NMS对每一类确定最终候选区域。Fast-RCNN使用了奇异值分解来提升矩阵乘法的运算速度。

[^1]: Selective Search 无法通过GPU执行，这是造成Fast R-CNN无法实时的一个重要性能瓶颈。在Faster-RCNN中，对这一瓶颈进行了优化

