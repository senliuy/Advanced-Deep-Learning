# DeepText: A Unified Framework for Text Proposal Generation and Text Detection in Natural Images {#deeptext-a-unified-framework-for-text-proposal-generation-and-text-detection-in-natural-images}

## 前言 {#前言}

16年那段时间的文字检测的文章，多少都和当年火极一时的[Faster R-CNN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks.html)\[2\]有关，DeepText（图1）也不例外，整体上依然是Faster R-CNN的框架，并在其基础上做了如下优化：

1. **Inception-RPN**：将RPN的$$3\times3$$卷积划窗换成了基于[Inception](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/going-deeper-with-convolutions.html)\[3\]的划窗。这点也是这篇文章的亮点；
2. **ATC**： 将类别扩展为‘文本区域’，‘模糊区域’与‘背景区域’;
3. **MLRP**：使用了多尺度的特征，ROI提供的按Grid的池化的方式正好融合不同尺寸的Feature Map。
4. **IBBV**：使用多个Iteration的bounding boxes的集合使用NMS

###### 图1：DeepText网络结构图

![](/assets/DeepText_1.png)

在阅读本文前，一定要先搞清楚Faster R-CNN，本文只会对DeepText对Faster R-CNN的改进进行说明，相同部分不再重复。

## 1. DeepText详解

DeepText的结构如Faster R-CNN如出一辙：首先特征层使用的是VGG-16，其次算法均由用于提取候选区域的RPN和用于物体检测的Fast R-CNN。

下面我们对DeepText优化的四点进行讲解。

### 1.1 Inception-RPN

首先DeepText使用了GoogLeNet提出的Inception结构代替Faster R-CNN中使用的$$3\times3$$卷积在Conv5\_3上进行滑窗。Inception的作用参照GoogLeNet中的讲解。

DeepText的Inception由3路不同的卷积构成：

* padding=1的$$3\times3$$ 的Max Pooling后接128个用于降维的$$1\times1$$卷积；
* 384个padding=1的$$3\times3$$卷积；
* 128个padding=2的$$5\times5$$卷积。

由于上面的Inception的3路卷积并不会改变Feature Map的尺寸，经过Concatnate操作后，Feature Map的个数变成了$$128+384+128 = 640$$。

针对场景文字检测中Ground Truth的特点，DeepText使用了和Faster R-CNN不同的锚点：$$(32, 48, 64, 80)$$四个尺寸及$$(0.2, 0.5, 0.8, 1.0, 1.2, 1.5)$$六种比例共$$4\times6=24$$个锚点。

DeepText的采样阈值也和Faster R-CNN不同：当$$\text{IoU} > 0.5$$时，锚点为正；$$\text{IoU} < 0.3$$，锚点为负。

Inception-RPN使用了阈值为0.7的NMS过滤锚点，最终得到的候选区域是top-2000的样本。

### 1.2 Ambiguous Text Classification（ATC）

DeepText将样本分成3类：

* Text: $$\text{IoU} > 0.5$$;
* Ambiguous: $$0.2 < \text{IoU} < 0.5$$; 
* Background: $$\text{IoU} < 0.2$$.

这样做的目的是让模型在训练过程中见过所有IoU的样本，该方法对于提高模型的召回率作用非常明显。

### 1.3  Multi Layer ROI Pooling（MLRP）

DeepText使用了VGG-16的Conv4\_3和Conv5\_3的多尺度特征，使用基于Grid的ROI Pooling将两个不同尺寸的Feature Map变成$$7\times7\times512$$的大小，通过$$1\times1$$卷积将Concatnate后的1024维的Feature Map降维到512维，如图1所示。

### 1.4 Iterative Bounding Box Voting \(IBBV\)

在训练过程中，每个Iteration会预测一组检测框：$$D_c^t = \{B_{i,c}^t, S_{i,c}^t\}_{i=1}^{N_{c,t}}$$，其中$$t=1,2,...,T$$表示训练阶段，$$N_{c,t}$$表示类别$$c$$的检测框，$$B$$和$$S$$分别表示检测框和置信度。NMS合并的是每个训练阶段的并集：$$D_c=\cup_{t=1}^{T} U_c^t$$。NMS使用的合并阈值是$$0.3$$。

在IBBV之后，DeepText接了一个过滤器用于过滤多余的检测框，过滤器的具体内容不详，后续待补。

## 总结

结合当时的研究现状，DeepText结合了当时state-of-the-art的Faster R-CNN，Inception设计了该算法。算法本身的技术性和创新性并不是很强，但是其设计的ATC和MLRP均在后面的物体检测算法中多次使用，而IBBV也在实际场景中非常值得测试。

\[1\] Zhong Z, Jin L, Zhang S, et al. Deeptext: A unified framework for text proposal generation and text detection in natural images\[J\]. arXiv preprint arXiv:1605.07314, 2016.

\[2\] S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal networks. In NIPS, 2015.

\[3\] Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions\[C\]. Cvpr, 2015.

