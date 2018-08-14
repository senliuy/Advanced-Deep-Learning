# DeepText: A Unified Framework for Text Proposal Generation and Text Detection in Natural Images

## 前言

16年那段时间的文字检测的文章，多少都和当年火极一时的Faster R-CNN有关，这篇也不例外，整体上依然是Faster R-CNN的框架，并在其基础上做了如下优化：

1. **Inception-RPN**：将RPN的$$3\times3$$卷积划窗换成了基于Inception的划窗。这点也是这篇文章的亮点；
2. **ATC**： 将类别扩展为‘文本区域’，‘模糊区域’与‘背景区域’;
3. **MLRP**：使用了多尺度的特征，ROI提供的按Grid的池化的方式正好融合不同尺寸的Feature Map。
 

## Reference

\[1\] Zhong Z, Jin L, Zhang S, et al. Deeptext: A unified framework for text proposal generation and text detection in natural images\[J\]. arXiv preprint arXiv:1605.07314, 2016.

