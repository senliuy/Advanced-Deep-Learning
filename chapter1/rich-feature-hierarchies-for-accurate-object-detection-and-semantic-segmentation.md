# Rich feature hierarchies for accurate object detection and semantic segmentation

# 简介

论文发表于2014年，自2012年之后，物体检测的发展开始变得缓慢，一个重要的原因是基于计算机视觉的方法（SIFT，HOG）进入了一个瓶颈期。生物学上发现人类的视觉是一个多层次的流程，而SIFT或者HOG只相当于人类视觉的第一层，这是导致瓶颈期的一个重要原因。

2012年，基于随机梯度下降的卷机网络在物体识别领域的突破性进展充分展现了CNN在提取图片特征上的巨大优越性 \[1\].CNN的一个重要的特点是其多层次的结构更符合人类的生物特征。

但大规模深度学习网络的应用对数据量提出了更高的需求。在数据量稀缺的数据集上进行训练，迭代次数太少会导致模型拟合能力不足，迭代次数太多会导致过拟合。为了解决该问题，作者使用了在海量数据上的无监督学习的预训练加上稀缺专用数据集的fine-tune.

在算法设计上，作者采用了 “Recognition Using Regions” \[2\]的思想，R-CNN使用Selective Search提取了2k-3k个候选区域，对每个候选区域单独进行特征提取和分类器训练，这也是R-CNN命名的由来。

同时，为了提高检测精度，作者使用了岭回归对检测位置进行了精校。以上方法的使用，使得算法在PASCAL数据集上的检测到达了新的高度。

# 算法详解

## 1. R-CNN流程

R-CNN测试过程可分成五个步骤

1. 使用Selective Search在输入图像上提取候选区域;
2. 使用CNN对每个wrap到固定大小（227 \* 227）的候选区域上提取特征;
3. 将CNN得到的特征Pool5层的特征输入N个（类别数量）SVM分类器对物体类别进行打分
4. 根据Pool5的特征输入岭回归器进行位置精校。
5. 使用贪心的非极大值抑制（NMS）合并候选区域，得到输出结果

所以，R-CNN的训练过程也涉及

* CNN特征提取器
* SVM分类器
* 岭回归位置精校器

三个模块的学习。

论文中给出的图（图1）没有画出回归器部分。

\[图1\]

# 2. 候选区域提取

R-CNN输入网络的并不是原始图片，而是经过Selective Search选择的候选区域。

1. Selective Search 使⽤ \[4\]的⽅法，将图像分成若⼲个⼩区域
2. 计算相似度，合并相似度较⾼的区域，直到⼩区域全部合并完毕
3. 输出所有存在过的区域，即候选区域 如下面伪代码：

Algorithm 1: Hierarchial Grouping Algorithm

```
Input: (color) image
Output: Set of object location hypotheses L

Obtain initial regions R = {r1, ..., r13}
Initial similarity set S = []

foreach Neighbouring region pair(ri, rj) do
    Calculate similarity s(ri, rj)
    S.insert(s(ri, rj))

while S != [] do
    Get highest similarity s(ri, rj) = max(S)
    Merge corresponding regions ri = Union(ri, rj)
    Remove similarities regarding ri: S = S.delete(ri, r*)
    Remove similarities regarding sj: S = S.delete(r*, rj)
    Calculate similarity set St between rt and its neighbours
    S = Union(S, St)
    R = Union(R, rt)

Extact object location boxes L from all regions in R
```

Selective Search 伪代码 区域的合并规则是：

1. 优先合并颜⾊相近的
2. 优先合并纹理相近的
3. 优先合并合并后总⾯积⼩的
4. 合并后，总⾯积在其BBOX中所占⽐例⼤的优先合并

图2是通过Selective Search得到的一候选区域

\[图2\]

## 3. 训练数据准备

### 3.1 CNN的数据准备

1. 预训练：使用ILSVRC 2012的数据，训练一个N类任务的分类器。在该数据集上，top-1的error是2.2%，达到了比较理想的初始化效果。
2. 微调：每个候选区域是一个N+1类的分类任务（在PASCAL上，N=20；ILSVRC，N=200）。表示该候选区域是某一类或者是背景。当候选区域和某一类物体的Ground Truth box的重合度（IoU）大于0.5时，该样本被判定为正样本，否则为负样本。

### 3.2 SVM分类器的数据准备





# 参考文献

\[1\] \] A. Krizhevsky, I. Sutskever, and G. Hinton. ImageNet classification with deep convolutional neural networks. In NIPS, 2012. 1, 3, 4, 7

\[2\] C. Gu, J. J. Lim, P. Arbelaez, and J. Malik. Recognition ´ using regions. In CVPR, 2009. 2

\[3\] J. Uijlings, K. van de Sande, T. Gevers, and A. Smeulders. Selective search for object recognition. IJCV, 2013. 1, 2, 3, 4, 5, 9

\[4\]. P. F. Felzenszwalb and D. P. Huttenlocher. Efficient GraphBased Image Segmentation. IJCV, 59:167–181, 2004. 1, 3, 4, 5, 7

