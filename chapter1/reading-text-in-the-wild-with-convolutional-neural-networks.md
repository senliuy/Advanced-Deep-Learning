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

算法中候选区域生成是采用的两种方案的并集，它们分别是Edge Boxes \[2\]和Aggregate Channel Feature Detector \[2\]。由于本书更偏重于深度学习领域的知识，对于上面两个算法只会粗略介绍方法，详细内容请自行查看参考文献中给出的论文。

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
4. 计算edge group的权值：
$$
w_b(s_i) = 1-\max\limits_{T} \prod ^{|T|-1}_j a(t_j, t_{j+1})
$$
5. 计算最终评分：
$$
h_b = \frac{\sum_i w_b(s_i)m_i}{2(b_w+b_h)^\kappa}
$$

其中 bounding box通过多尺寸，多比例的滑窗方式得到。

#### 1.2 Aggregate Channel Feature Detector


## Reference

\[1\] Jaderberg M, Simonyan K, Vedaldi A, et al. Reading text in the wild with convolutional neural networks\[J\]. International Journal of Computer Vision, 2016, 116\(1\): 1-20.

\[2\] Zitnick, C. L., & Dollár, P. \(2014\). Edge boxes: Locating object propos- als from edges. In D. J. Fleet, T. Pajdla, B. Schiele, & T. Tuytelaars \(Eds.\),Computer vision ECCV 2014 13th European conference, Zurich, Switzerland, September 6–12, 2014, proceedings, part IV\(pp. 391–405\). New York City: Springer.

\[3\] Dollár, P., & Zitnick, C. L. \(2014\). Fast edge detection using structured forests.arXiv:1406.5549.

