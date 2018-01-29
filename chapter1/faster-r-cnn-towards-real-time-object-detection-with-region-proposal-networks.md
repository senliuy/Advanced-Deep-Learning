# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

Fast-RCNN \[1\]虽然实现了端到端的训练，而且也通过共享卷积的形式大幅提升了R-CNN的计算速度，但是其仍难以做到实时。其中一个最大的性能瓶颈便是候选区域的计算。在之前的物体检测系统中，Selective Search是最常用的候选区域提取方法，它贪心的根据图像的低层特征合并超像素（SuperPixel）。另外一个更快速的版本是EdgeBoxes

