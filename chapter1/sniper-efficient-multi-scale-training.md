# SNIPER: Efficient Multi-Scale Training

## 前言

图像金字塔是传统的提升物体检测精度的策略之一，其能提升精度的一个原因是尺寸多样性的引入。但是图像金字塔也有一个非常严重的缺点：即增加了模型的计算量，一个经过3个尺度放大（1x,2x,3x）的图像金子塔要多处理14倍的像素点。

SNIPER（Scale Normalization for Image Pyramid with Efficient Resampling）的提出动机便是解决图像金字塔计算量大的问题，图像金字塔存在一个固有的问题：对于一张放大之后的高分辨率的图片，其物体也随之放大，有时甚至待检测物体的尺寸超过了特征向量的感受野，这时候对大尺寸物体的检测便很难做到精确，因为这时候已经没有合适的特征向量了；对于一个缩小了若干倍的图像中的一个小尺寸物体，由于网络结构中降采样的存在，此时也很难找到合适的特征向量用于该物体的检测。因此作者在之前的SNIP\[2\]论文中便提出了高分辨率图像中的大尺寸物体和低分辨率图像中的小尺寸物体时应该忽略的。

为了阐述SNIPER的提出动机，作者用很大的篇幅来对比R-CNN和Fast R-CNN的异同，一个重要的观点就是R-CNN具有尺度不变性，而Fast R-CNN不具有该特征。R-CNN先通过Selective Search选取候选区域，然后无论候选区域的尺寸是多少都会将其归一化到$$224\times224$$的尺寸，该策略虽然备受诟病，但是它却保证了R-CNN具有尺度不变性的特征。Fast R-CNN将resize的部分移到了使用了卷积之后，也就是使用RoI池化产生长度固定的特征向量，也就是说Fast R-CNN是将原始图像作为输入，无论里面的尺寸大小，均使用相同的卷积核计算特征向量。但是这样做真的合理吗？不同尺寸的待检测物体使用相同的卷积核来训练真的好吗？答案当然是否定的。

SNIPER策略最重要的贡献是提出了尺寸固定（$$512 \times 512$$）的chips的东西，chips是图像金子塔中某个尺度图像的一个子图。chips有正负之分，其中正chip中包含了很多尺寸合适的标注样本，而负chip则不包含标注样本或者如第二段中所阐述的包含的样本不适合这个尺度的图像。

需要注意的是SNIPER只是一个采样策略，是对图像金字塔的替代。在论文中作者将SNIPER应用到了Faster R-CNN和Mask R-CNN中，同理，SNIPER也几乎可以应用到其它任何检测甚至识别算法中。

## 1. SNIPER 详解

作为一个采样策略，SNIPER的输入数据是原始的数据集，输出的是在图像上采样得到的子图（chips），如图1的虚线部分所示，而这些chips会直接作为s。当然，chips上的物体的Ground Truth也需要针对性的修改。那么这些chips是怎么计算的呢，下面我们详细分析之。

<figure>
<img src="./assets/SNIPER_1.jpeg" alt="图1：SNIPER的chips示意图" />
<figcaption>图1：SNIPER的chips示意图</figcaption>
</figure>

## Reference

\[1\] Singh B, Najibi M, Davis L S. SNIPER: Efficient Multi-Scale Training[J]. arXiv preprint arXiv:1805.09300, 2018.

\[2\] 