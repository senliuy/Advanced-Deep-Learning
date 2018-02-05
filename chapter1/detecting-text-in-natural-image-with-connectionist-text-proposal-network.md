# Detecting Text in Natural Image with Connectionist Text Proposal Network

CTPN和Faster-RCNN \[1\] 出自同系，根据文本区域的特点做了专门的调整，一个重要的地方是RNN的引人，笔者在实现CTPN的时候也是直接在Faster-RCNN基础上改的。理解了Faster R-CNN之后，CTPN理解的难度也不大，下面开始分析这篇论文。

## 简介

场景文字检测和物体检测存在两个显著的不同之处

1. 场景文字检测有明显的边界，例如Wolf 准则 \[2\]，而物体检测的边界要求较松，一般IoU为0.7便可以判断为检测正确；
2. 场景文字检测有明显的序列特征，而物体检测没有这些特征；
3. 和物体检测相比，场景文字检测含有更多的小尺寸的物体。

参考文献

\[1\] Ren, S., He, K., Girshick, R., Sun, J.: Faster R-CNN: Towards real-time object detection with region proposal networks \(2015\), in Neural Information Processing Systems \(NIPS\)

\[2\] Wolf, C., Jolion, J.: Object count / area graphs for the evaluation of object detection and segmentation algorithms. International Journal of Document Analysis 8, 280–296 \(2006\)

