# Reading Text in the Wild with Convolutional Neural Networks

## 前言

这篇论文出自著名的牛津大学计算机视觉组（Visual Geometry Group），没错，就是发明VGG网络的那个实验室。这篇论文是比较早的研究端到端文字检测和识别的经典算法之一。参考文献显示文章发表于2016年，但是该论文于2014年就开始投稿，正好也是那一年物体检测的开山算法R-CNN发表。

论文的算法在现在看起来比较传统和笨拙，算法主要分成两个阶段：

1. 基于计算机视觉和机器学习的场景文字检测；
2. 基于深度学习的文本识别。

虽然论文说自己是端到端的系统，但是算法的阶段性特征是非常明显的，并不是纯粹意义上的端到端。这里说这些并不是要否定这篇文章的贡献，结合当时深度学习的发展条件，算法涉及成这样也是可以理解的。

虽然方法比较笨拙，但是作为OCR领域的教科书式的文章，这篇文章还是值得一读的。

## 算法详解

### 1. 候选区域生成

算法中候选区域生成是采用的两种方案的并集，它们分别是Edge Boxes \[2\]和Aggregate Channel Feature Detector \[2\]。

#### 1.1 Edge Boxes



## Reference

\[1\] Jaderberg M, Simonyan K, Vedaldi A, et al. Reading text in the wild with convolutional neural networks\[J\]. International Journal of Computer Vision, 2016, 116\(1\): 1-20.

\[2\] Zitnick, C. L., & Dollár, P. \(2014\). Edge boxes: Locating object propos- als from edges. In D. J. Fleet, T. Pajdla, B. Schiele, & T. Tuytelaars \(Eds.\),Computer vision ECCV 2014 13th European conference, Zurich, Switzerland, September 6–12, 2014, proceedings, part IV\(pp. 391–405\). New York City: Springer.

\[3\] Dollár, P., & Zitnick, C. L. \(2014\). Fast edge detection using structured forests.arXiv:1406.5549.

