# Rich feature hierarchies for accurate object detection and semantic segmentation

# 简介

论文发表于2014年，自2012年之后，物体检测的发展开始变得缓慢，一个重要的原因是基于计算机视觉的方法（SIFT，HOG）进入了一个瓶颈期。生物学上发现人类的视觉是一个多层次的流程，而SIFT或者HOG只相当于人类视觉的第一层，这是导致瓶颈期的一个重要原因。

2012年，基于随机梯度下降的卷机网络在物体识别领域的突破性进展充分展现了CNN在提取图片特征上的巨大优越性 \[1\].CNN的一个重要的特点是其多层次的结构更符合人类的生物特征。

但大规模深度学习网络的应用对数据量提出了更高的需求。在数据量稀缺的数据集上进行训练，迭代次数太少会导致模型拟合能力不足，迭代次数太多会导致过拟合。为了解决该问题，作者使用了在海量数据上的无监督学习的预训练加上稀缺专用数据集的fine-tune. 

在算法设计上，作者采用了 “Recognition Using Regions” \[2\]的思想，R-CNN使用Selective Search提取了2k-3k个候选区域，对每个候选区域单独进行特征提取和分类器训练，这也是R-CNN命名的由来。

同时，为了提高检测精度，作者使用了岭回归对检测位置进行了精校。以上方法的使用，使得算法在PASCAL数据集上的检测到达了新的高度。

# 算法详解



# 参考文献

\[1\] \] A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012. 1, 3, 4, 7

\[2\] C. Gu, J. J. Lim, P. Arbelaez, and J. Malik. Recognition ´ using regions. In CVPR, 2009. 2



