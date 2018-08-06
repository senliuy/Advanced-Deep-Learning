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

算法中候选区域生成是采用的两种方案的并集，$$B=\{B_e \cup B_d\}$$它们分别是Edge Boxes \[2\]和Aggregate Channel Feature Detector \[2\]。由于本书更偏重于深度学习领域的知识，对于上面两个算法只会粗略介绍方法，详细内容请自行查看参考文献中给出的论文。

#### 1.1 Edge Boxes

Edge boxes的提出动机是bounding box内完全包含的的轮廓越多（Contours），那么该bounding box为候选区域的概率越高，这种特征到文字检测方向尤为明显。

Edge boxes的计算步骤如下：

1. 边缘检测；
2. 计算Edge Group：通过合并近似在一条直线上的边缘点得到的；
3. 计算两个Edge Group之间的亲密度（affinity）：


   $$
   a(s_i, s_j) = |cos(\theta_i, \theta_{i,j})cos(\theta_j, \theta_{ij}))|^\gamma
   $$

其中$$\gamma$$ 为超参数，一般设置为$$2$$。$$(\theta_i, \theta_j)$$ 是两组Edge Group的平均旋转角度，$$\theta_{ij}$$ 是两组edge boxes的平均位置$$x_i$$, $$x_j$$的夹角。

1. 计算edge group的权值：


   $$
   w_b(s_i) = 1-\max\limits_{T} \prod ^{|T|-1}_j a(t_j, t_{j+1})
   $$

2. 计算最终评分：


   $$
   h_b = \frac{\sum_i w_b(s_i)m_i}{2(b_w+b_h)^\kappa}
   $$

其中 bounding box通过多尺寸，多比例的滑窗方式得到。

计算完评分之后，通过NMS得到最终的候选区域，$$B_e$$。

#### 1.2 Aggregate Channel Feature Detector

该方法的核心思想是通过Adaboost集成多尺度的ACF特征。ACF特征不具有尺度不变性，而使用多尺度输入图像计算特征的方法又过于耗时，[4]中只在每个octave重采样图像计算特征，每个octave之间的特征使用其它尺度进行估计：
$$
C_s = R(C_{s'}, s/s')(s/s')^{-\lambda\Omega}
$$
最后在通过AdaBoost集成由决策树构成的若分类器，通过计算阈值的方式得到最终的候选区域，$$B_d$$。

### 2. 候选区域的精校

通过第一节的方法得到候选区域后，作者对这些候选区域进行了进一步的精校，包括对候选区域是否包含文本的二分类和bounding box位置的调整。

#### 2.1 word/no word分类

作者通过从训练集上采样得到了一批候选区域，其中和Ground Truth的overlap大于0.5的设为正样本，小于0.5的设为负样本。提取了候选区域的HoG特征，并使用随机森林分类器训练了一个判断候选区域是否包含文本的二分类的分类器。随机森林包括10棵决策树，每棵树的最大深度为64。

#### 2.2 Bounding box回归

Bounding box回归器是通过四层CNN加上两层全连接得到的，四层卷积层卷积核的尺寸和个数分别是$$\{5,64\}, \{5,128\}, \{3,256\}, \{3,512\}$$，全连接隐层节点的个数是$$4k$$，网络有四个输出，分别用于预测bounding box的左上角和右下角坐标$$b=\{x_1, y_1, x_2, y_2\}$$。损失函数则是使用的$$L_2$$损失函数：
$$
\min\limits_{\Phi}\sum_{b\in B_{brain}} ||g(I_b; \Phi) - q(b_{gt}))||^2_2
$$
其中$$\Phi$$表示由训练集得到的参数。

作者也尝试了基于CNN的分类和回归的多任务模型，但是效果并不如基于HoG特征的随机森林分类器。

### 3. 文本识别

OCR中，有基于字符的时序序列分类器和基于单词的分类器。前者的优点是分类器类别数目少(26类)，缺点是序列模型识别困难，一个字符错误导致整个单词的识别失败，基于序列的建模要比简单的分类任务复杂得多。后者的优点是无时序特征，缺点是类别数目太多，穷举所有类别不太可能，如果采用基于抽样的方法的话，抽样类别太少导致在测试时遇见未定义的单词的概率过高，抽样类别多的话导致训练集样本不够，不足以覆盖所有类别。

在这里，作者使用的是第二种方案，也就是基于单词的分类方法。为了解决未定义单词的问题，作者采样了90k个最常见的单词，样本不足的问题则是通过合成数据解决的。

#### 3.1 合成数据

合成模型使用了90k的单词作为合成文本，在图像变化上，考虑了以下几点：



## Reference

\[1\] Jaderberg M, Simonyan K, Vedaldi A, et al. Reading text in the wild with convolutional neural networks\[J\]. International Journal of Computer Vision, 2016, 116\(1\): 1-20.

\[2\] Zitnick, C. L., & Dollár, P. \(2014\). Edge boxes: Locating object propos- als from edges. In D. J. Fleet, T. Pajdla, B. Schiele, & T. Tuytelaars \(Eds.\),Computer vision ECCV 2014 13th European conference, Zurich, Switzerland, September 6–12, 2014, proceedings, part IV\(pp. 391–405\). New York City: Springer.

\[3\] Dollár, P., Appel, R., Belongie, S., & Perona, P. \(2014\). Fast feature pyramids for object detection.IEEE Transactions on Pattern Analysis and Machine Intelligence,36, 1532–1545.

