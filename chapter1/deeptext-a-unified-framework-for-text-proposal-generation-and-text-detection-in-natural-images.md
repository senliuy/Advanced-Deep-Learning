# DeepText: A Unified Framework for Text Proposal Generation and Text Detection in Natural Images {#deeptext-a-unified-framework-for-text-proposal-generation-and-text-detection-in-natural-images}

## 前言 {#前言}

16年那段时间的文字检测的文章，多少都和当年火极一时的Faster R-CNN有关，DeepText（图1）也不例外，整体上依然是Faster R-CNN的框架，并在其基础上做了如下优化：

1. **Inception-RPN**：将RPN的$$3\times3$$卷积划窗换成了基于Inception的划窗。这点也是这篇文章的亮点；
2. **ATC**： 将类别扩展为‘文本区域’，‘模糊区域’与‘背景区域’;
3. **MLRP**：使用了多尺度的特征，ROI提供的按Grid的池化的方式正好融合不同尺寸的Feature Map。
4. **IBBV**：使用多个Iteration的bounding boxes的集合使用NMS

###### 图1：DeepText网络结构图

![](\assets\DeepText_1.png)

在阅读本文前，一定要先搞清楚Faster R-CNN，本文只会对DeepText对Faster R-CNN的改进进行说明，相同部分不再重复。

1. DeepText详解


  


