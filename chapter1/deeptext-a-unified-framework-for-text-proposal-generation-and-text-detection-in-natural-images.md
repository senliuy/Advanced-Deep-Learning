# DeepText: A Unified Framework for Text Proposal Generation and Text Detection in Natural Images {#deeptext-a-unified-framework-for-text-proposal-generation-and-text-detection-in-natural-images}

## 前言 {#前言}

16年那段时间的文字检测的文章，多少都和当年火极一时的[Faster R-CNN]()\[2\]有关，DeepText（图1）也不例外，整体上依然是Faster R-CNN的框架，并在其基础上做了如下优化：

1. **Inception-RPN**：将RPN的$$3\times3$$卷积划窗换成了基于[Inception]()\[3\]的划窗。这点也是这篇文章的亮点；
2. **ATC**： 将类别扩展为‘文本区域’，‘模糊区域’与‘背景区域’;
3. **MLRP**：使用了多尺度的特征，ROI提供的按Grid的池化的方式正好融合不同尺寸的Feature Map。
4. **IBBV**：使用多个Iteration的bounding boxes的集合使用NMS

###### 图1：DeepText网络结构图

![](\assets\DeepText_1.png)

在阅读本文前，一定要先搞清楚Faster R-CNN，本文只会对DeepText对Faster R-CNN的改进进行说明，相同部分不再重复。

## 1. DeepText详解

DeepText的结构如Faster R-CNN如出一辙：首先特征层使用的是VGG-16，其次算法均由用于提取候选区域的RPN和用于物体检测的Fast R-CNN。

下面我们对DeepText优化的四点进行讲解。

### 1.1 Inception-RPN

首先DeepText使用了GoogLeNet提出的Inception结构代替Faster R-CNN中使用的$$3\times3$$卷积在Conv5_3上进行滑窗。Inception的作用参照GoogLeNet中的讲解。

DeepText的Inception由3路不同的卷积构成：

* padding=1的$$3\times3$$ 的Max Pooling后接128个用于降维的$$1\times1$$卷积；
* 384个padding=1的$$3\times3$$卷积；
* 128个padding=2的$$5\times5$$卷积。

由于上面的Inception的3路卷积并不会改变Feature Map的尺寸，经过Concatnate操作后，Feature Map的个数变成了$$128+384+128 = 640$$。

针对场景文字检测中Ground Truth的特点，DeepText使用了和Faster R-CNN不同的锚点：$$(32, 48, 64, 80)$$四个尺寸及$$(0.2, 0.5, 0.8, 1.0, 1.2, 1.5)$$六种比例共$$4\times6=24$$个锚点。

DeepText的采样阈值也和Faster R-CNN不同————当$$\text{IoU} > 0.5$$时，锚点为正；$$\text{IoU} < 0.3$$，锚点为负。

Inception-RPN使用了阈值为0.7的NMS过滤锚点，最终得到的候选区域是top-2000的样本。

### Ambiguous Text Classification（ATC） 

DeepText将样本分成3类：
* Text: $$\text{IoU} > 0.5$$;
* Ambiguous: $$0.2 < \text{IoU} < 0.5$$; 
* Background: $$\text{IoU} < 0.2$$.

  


