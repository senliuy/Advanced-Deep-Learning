# Detecting Text in Natural Image with Connectionist Text Proposal Network

CTPN和Faster R-CNN \[1\] 出自同系，根据文本区域的特点做了专门的调整，一个重要的地方是RNN的引人，笔者在实现CTPN的时候也是直接在Faster-RCNN基础上改的。理解了Faster R-CNN之后，CTPN理解的难度也不大，下面开始分析这篇论文。

作者提供的CTPN demo源码：[https://github.com/tianzhi0549/CTPN](https://github.com/tianzhi0549/CTPN)

基于TensorFlow的CTPN开源代码：[https://github.com/eragonruan/text-detection-ctpn](https://github.com/eragonruan/text-detection-ctpn)

## 简介

传统的文字检测是一个自底向上的过程，总体上的处理方法分成连通域和滑动窗口两个方向。基于连通域的方法是先通过快速滤波器得到文字的像素点，再根据图像的低维特征（颜色，纹理等）贪心的构成线或者候选字符。基于滑动窗口的方法是通过在图像上通过多尺度的滑窗，根据图像的人工设计的特征（HOG，SIFT等），使用预先训练好的分类器判断是不是文本区域。但是这两个途径在健壮性和可信性上做的并不好，而且滑窗是一个非常耗时的过程。

卷积网络在2012年的图像分类上取得了巨大的成功，在2015年Faster R-CNN在物体检测上提供了非常好的算法框架。所以用深度学习的思想解决场景文字检测自然而然的成为研究热点。对比发现，场景文字检测和物体检测存在两个显著的不同之处

1. 场景文字检测有明显的边界，例如Wolf 准则 \[2\]，而物体检测的边界要求较松，一般IoU为0.7便可以判断为检测正确；
2. 场景文字检测有明显的序列特征，而物体检测没有这些特征；
3. 和物体检测相比，场景文字检测含有更多的小尺寸的物体。

针对以上特点，CTPN做了如下优化：

1. 在RPN中使用更符合场景文字检测特点的锚点；
2. 针对锚点的特征使用新的损失函数；
3. RNN（双向LSTM）的引入用于处理场景文字检测中存在的序列特征；
4. Sife-refinement的引入进一步优化文字区域。

## 算法详解

算法流程

CTPN的流程和Faster R-CNN的RPN网络类似，首先使用VGG-16提取特征，在conv5进行3\*3，步长为1的滑窗。设conv5的尺寸是W\*H，这样在conv5的同一行，我们可以得到W个256维的特征向量，

## 参考文献

\[1\] Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks \(2015\), in Neural Information Processing Systems \(NIPS\)

\[2\] Wolf, C., Jolion, J.: Object count / area graphs for the evaluation of object detection and segmentation algorithms. International Journal of Document Analysis 8, 280–296 \(2006\)



