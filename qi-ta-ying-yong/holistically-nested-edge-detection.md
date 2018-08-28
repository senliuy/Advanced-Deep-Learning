# Holistically-Nested Edge Detection

# 前言

本文提出了一个新的网络结构用于边缘检测，即本文的题目Holistically-Nested Network（HED）。其中Holistically表示该算法试图训练一个image-to-image的网络；Nested则强调在生成的输出过程中通过不断的集成和学习得到更精确的边缘预测图的过程。从图1中HED和传统Canny算法进行边缘检测的效果对比图我们可以看到HED的效果要明显优于Canny算子的。

###### 图1：HED vs Canny

![](/assets/HED_1.png)

由于是HED是image-to-image的，所以该算法也很容易扩展到例如语义分割的其它领域。此外在OCR中的文字检测中，文字区域往往具有比较强的边缘特征，因此HED也可以扩展到场景文字检测中，著名的EAST \[2\]算法便得到了HED的启发。

下面我们结合HED的[Keras源码](https://github.com/lc82111/Keras_HED)对HED展开详细分析。

### 1.1 HED的骨干网络

HED创作于2015年，使用了当时state-of-the-art的VGG-16作为骨干网络，并且使用迁移学习初始化了网络权重。

HED使用了多尺度的特征，类似多尺度特征的思想还有Inception，SSD，FPN等方法，对比如图2。

* \(a\) Multi-stream learning: 使用不同结构，不同参数的网络训练同一副图片，类似的结构有Inception；
* \(b\) Skip-layer network learning: 该结构有一个主干网络，在主干网络中添加若干条到输出层的skip-layer，类似的结构有FPN；
* \(c\) Single model on multiple inputs: 该方法使用同一个网络，不同尺寸的输入图像得到不同尺度分Feature Map，YOLOv2采用了该方法；
* \(d\) Training independent network: 使用完全独立的网络训练同一张图片，得到多个尺度的结果，该方法类似于集成模型；
* \(e\) Holistically-Nested networks: HED采用的方法，下面详细介绍。

###### 图2：几种提取多尺度特征的算法的网络结构

![](/assets/HED_2.png)

### 1.2 Holistically-Nested networks

Holistically-Nested networks的结构如图3以及下面代码：

###### 图3：Holistically-Nested networks结构图

![](/assets/HED_3.png)

```py
# Input
img_input = Input(shape=(480,480,3), name='input')
# Block 1
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
b1= side_branch(x, 1) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block1_pool')(x) # 240 240 64
# Block 2
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
b2= side_branch(x, 2) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block2_pool')(x) # 120 120 128
# Block 3
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
b3= side_branch(x, 4) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block3_pool')(x) # 60 60 256
# Block 4
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
b4= side_branch(x, 8) # 480 480 1
x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='block4_pool')(x) # 30 30 512
# Block 5
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x) # 30 30 512
b5= side_branch(x, 16) # 480 480 1
# fuse
fuse = Concatenate(axis=-1)([b1, b2, b3, b4, b5])
fuse = Conv2D(1, (1,1), padding='same', use_bias=False, activation=None)(fuse) # 480 480 1
# outputs
o1    = Activation('sigmoid', name='o1')(b1)
o2    = Activation('sigmoid', name='o2')(b2)
o3    = Activation('sigmoid', name='o3')(b3)
o4    = Activation('sigmoid', name='o4')(b4)
o5    = Activation('sigmoid', name='o5')(b5)
ofuse = Activation('sigmoid', name='ofuse')(fuse)
# model
model = Model(inputs=[img_input], outputs=[o1, o2, o3, o4, o5, ofuse])
```

无论从图3还是源码，VGG-16的骨干架构是非常明显的。在VGG-16的5个block的Max Pooling降采样之前，HED通过side_branch函数产生了5个分支，side_branch的源码如下

```py
def side_branch(x, factor):
    x = Conv2D(1, (1, 1), activation=None, padding='same')(x)
    kernel_size = (2*factor, 2*factor)
    x = Conv2DTranspose(1, kernel_size, strides=factor, padding='same', use_bias=False, activation=None)(x)
    return x
```

其中Conv2DTranspose是反卷积操作，side_branch的输出特征向量的维度已反应在注释中。

HED的fuse branch层是由5个side_branch的输出通过Concatenate操作合并而成的。网络的5个side_branch和一个fuse branch通过sigmoid激活函数后共同作为网络的输出，每个输出的尺寸均和输入图像相同。

### 1.3 HED的损失函数



## Reference

\[1\] Xie S, Tu Z. Holistically-nested edge detection\[C\]//Proceedings of the IEEE international conference on computer vision. 2015: 1395-1403.

\[2\] Zhou X, Yao C, Wen H, et al. EAST: an efficient and accurate scene text detector\[C\]//Proc. CVPR. 2017: 2642-2651.

