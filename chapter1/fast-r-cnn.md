# Fast R-CNN

## 简介

在之前介绍的R-CNN\[1\]和SPP-net\[2\]中，它们都是先通过卷积网络提取特征，然后根据特征训练SVM做分类和回归器用于位置矫正。这种多阶段的流程有两个问题

1. 保存中间变量需要使用大量的硬盘存储空间
2. 不能根据分类和矫正结果调整卷积网络权值，会一定程度的限制网络精度。

作者通过多任务的方式将R-CNN和SPP-net整合成一个流程。这样避免和中间存储空间的使用，同时也带来分类和检测精度的提升。Fast R-CNN的代码也是非常优秀的一份代码，强烈推荐参考学习：[https://github.com/rbgirshick/fast-rcnn。](https://github.com/rbgirshick/fast-rcnn。)

同SPP-net一样，Fast R-CNN将整张图像输入到卷积网络用语提取特征，将Selective Search选定的候选区域坐标映射到卷积层。使用ROI 池化层 \(单尺度的SPP层\)将不同尺寸的候选区域特征窗口映射成相同尺寸的特征向量。经过两层全连接后将得到的特征分支成两个输出层，一个N+1类的softmax用于分类，一个bbox 回归器用于位置精校。这两个任务的损失共同用于调整网络的参数。

和SPP-net对比，fast R-CNN最大的优点是多任务的引进，在优化训练过程的同时也避免了额外存储空间的使用并在一定程度上提升了精度。

## 算法详解

### 1. 数据准备

#### 1.1 候选区域选择

数据准备工作集中在` /lib/datasets` 目录下面，下面会着重介绍几个重要部分

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

图2

#### 2.2 ROI pooling 层

ROI Pooling是一层的SPP-net。有\(r,c,w,h\)定义，\(r,c\)表示候选区域的左上角，\(w,h\)表示候选区域的高和宽。假设我们要将特征层映射到H\*W的矩阵。ROI pooling通过将特征层分成h/H \* w/W的窗格，每个窗格进行max pooling得到，在作者的实验中W=H=7。ROI pooling layer是在caffe源码中`src/caffe/layers/roi_pooling_layer.cpp`通过C++语言实现的。

既然ROI pooling层是自己定义的，当然我们也应向caffe中定义其它层一样，给出ROI pooling层的反向的计算过程。在Fast-RCNN中，ROI 使用的是固定grid数量的max pooling，在调参时，只对grid中选为最大值的像素点更新参数。可以表示为

```
\frac{\partial L}{\partial x_i} = \sum_r \sum_j [i = i^*(r,j)]\frac{\partial L}{\partial y_{r,j}}
```

#### 2.3 多任务

Fast-RCNN最重要的贡献是多任务模型的提出，多任务将原来物体检测的多阶段训练简化成一个端到端（end-to-end）的模型。Fast-RCNN有两个任务组成，一个任务是用来对候选区域进行分类，另外一个回归器是用来矫正候选区域的位置。

2.3.1 分类任务L\_{cls}

设p=\{p_0, p_1, ..., p\_n\}是候选区域集合，则L\_{cls}是一个K+1类的分类任务，其中输入数据是经过卷积和全连接之后提取的特征向量，输出数据是候选区域的类别（u），包括K类物体\(u&gt;=1\)和1类背景\(u=0\)。分类任务的损失函数是softmax损失。

2.3.2 位置精校任务L\_{loc}

对于候选区域所属的类别u，v={v_x, v_y, vw, vh}表示候选区域的ground-truth, tu = \(tux, tuy, tuw, tuh\)表示对候选区域的类别u（u&gt;=1）预测的位置。损失函数是smooth L1损失，表示为

$$smooth_{L_1}(x) = \begin{case}
\end{case}$$

#### 参考文献

\[1\] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in CVPR, 2014

\[2\] K. He, X. Zhang, S. Ren, and J. Sun. Spatial pyramid pooling in deep convolutional networks for visual recognition. In ECCV, 2014. 1, 2, 3, 4, 5, 6, 7

[^1]: Selective Search 无法通过GPU执行，这是造成Fast R-CNN无法实时的一个重要性能瓶颈。在Faster-RCNN中，对这一瓶颈进行了优化

