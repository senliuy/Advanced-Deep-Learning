# Scene Text Detection via Holistic, Multi-Channel Prediction

## 前言

本文是在边缘检测经典算法[HED](https://senliuy.gitbooks.io/advanced-deep-learning/content/qi-ta-ying-yong/holistically-nested-edge-detection.html)\[2\]之上的扩展，在这篇论文中我们讲过HED算法可以无缝转移到语义分割场景中。而这篇论文正是将场景文字检测任务转换成语义分割任务来实现HED用于文字检测的。图1是HED在身份证上进行边缘检测得到的掩码图，从图1中我们可以看出HED在文字检测场景中也是有一定效果的。

###### 图1：HED在身份证上得到的掩码图

![](/assets/HMCP_1.jpeg)

HED之所以能用于场景文字检测一个重要的原因是文字区域具有很强的边缘特征。

论文的题目为Holistic，Multi-Channel Prediction（HMCP），其中Holistic表示算法基于HED，Multi-Channel表示该算法使用多个Channel的标签训练模型。也就是为了提升HED用于文字检测的精度，这篇文章做的改进是将模型任务由单任务模型变成是由文本行分割，字符分割和字符间连接角度构成的多任务系统。由于HMCP采用的是语义分割的形式，所以其检测框可以扩展到多边形或者是带旋转角度的四边形，这也更符合具有严重仿射变换的真实场景。

## 1. HMCP详解

HMCP的流程如图2：\(a\)是输入图像，\(b\)是预测的三个mask，分别是文本行掩码，字符掩码和字符间连接角度掩码，\(c\)便是根据\(b\)的三个掩码得到的检测框。

###### 图2：HMCP流程

![](/assets/HMCP_2.png)

那么问题来了：  
1. 如何构建三个channel掩码的标签值；  
2. 如何根据预测的掩码构建文本行。

### 1.1 HMCP的标签值

在文本检测数据集中，常见的标签类型有QUAD和RBOX两种形式。其中QUAD是指标签值包含四个点$$\mathbf{G}=\{(x_i,y_i)|i\in\{1,2,3,4\}\}$$，由这四个点构成的不规则四边形（Quadrangle）便是文本区域。QBOX则是由一个规则矩形$$\mathbf{R}$$和其旋转角度$$\theta$$构成，即$$\mathbf{G} = \{\mathbf{R}, \theta \}$$。QUAD和RBOX是可以相互转换的。

HMCP的数据的Ground Truth分别包含基于文本行和基于字符的标签构成\(QUAD或RBOX\)，如图3.\(b\)和图3.\(c\)。数据集中只有基于文本行的Ground Truth，其基于单词的Ground Truth是通过SWT\[3\]得到的。\(d\)是基于文本行Ground Truth得到的二进制掩码图，文本区域的值为1，非文本区域的值为0。\(e\)是基于单词的二进制掩码图，掩码也是0或者1。由于字符间的距离往往比较小，为了能区分不同字符之间的掩码，正样本掩码的尺寸被压缩到了Ground Truth的一半。\(f\)是字符间连接角度掩码，其值为$$[-\pi/2, \pi/2]$$，然后再通过归一化映射到$$[0,1]$$之间。角度的值是由RBOX形式的Ground Truth得到的值。

###### 图3：HMCP的Ground Truth以及三种Mask

![](/assets/HMCP_3.png)

### 1.2 HMCP的骨干网络

HMCP的骨干网络继承自HED，如图4所示。HMCP的主干网络使用的是VGG-16，在每个block降采样之前通过反卷积得到和输入图像大小相同的Feature Map，最后通过fuse层将5个side branch的Feature Map拼接起来并得到预测值。HMCP和HED的不同之处是HMCP的输出节点有三个任务。

###### 图4：HMCP的骨干网络

![](/assets/HMCP_4.png)

### 1.3 HMCP的损失函数

### 1.3.1 训练

设HMCP的训练集为$$S=\{(X_n,y_n),n=1,...,N\}$$，其中$$N$$是样本的数量。标签$$Y_n$$由三个掩码图构成，即$$Y_n=\{R_n,C_n,\Theta_n\}$$，其中$$R_n=\{r_j^{(n)}\in\{0,1\},j=1,...,|R_n|\}$$表示文本区域的二进制掩码图，$$C_n=\{c_j^{(n)}\in\{0,1\},j=1,...,|C_n|\}$$是字符的二进制掩码图，$$\Theta_n=\{\theta_j^{(n)}\in\{0,1\},j=1,...,|\Theta_n|\}$$是相邻字符的连接角度。注意只有当$$r_j^{(n)}=1$$时$$\theta_j^{(n)}$$才有效。

与HED不同的是HMCP的损失函数没有使用side branch，即损失函数仅由fuse层构成：

$$
\mathcal{L} = \mathcal{L}_{\text{fuse}}(\mathbf{W}, \mathbf{w}, Y, \hat{Y})
$$

其中$$\mathbf{W}$$为VGG-16部分的参数，$$\mathbf{w}$$为fuse层部分的参数。$$\hat{Y}=\{\hat{R}, \hat{C}, \hat{\Theta}\}$$是预测值：

$$
\hat{Y} = \text{CNN}(X,\mathbf{W},\mathbf{w})
$$

$$\mathcal{L}_{\text{fuse}}(\mathbf{W}, \mathbf{w}, Y, \hat{Y})$$由三个子任务构成:

$$
\mathcal{L}_{\text{fuse}}(\mathbf{W}, \mathbf{w}, Y,\hat{Y}) = \lambda_1\Delta_r(\mathbf{W}, \mathbf{w},R,\hat{R}) + 
\lambda_2\Delta_c(\mathbf{W}, \mathbf{w},C,\hat{C}) +
\lambda_3\Delta_o(\mathbf{W}, \mathbf{w},\Theta,\hat{\Theta},R) 
$$

其中$$\Delta_r(\mathbf{W}, \mathbf{w})$$表示基于文本掩码的损失值，$$\Delta_c(\mathbf{W}, \mathbf{w})$$是基于字符掩码的损失值，两个均是使用HED采用过的类别平衡交叉熵损失函数：

$$
\Delta_r(\mathbf{W}, \mathbf{w})
$$

## Reference

\[1\] Yao C, Bai X, Sang N, et al. Scene text detection via holistic, multi-channel prediction\[J\]. arXiv preprint arXiv:1606.09002, 2016.

\[2\] Xie S, Tu Z. Holistically-nested edge detection \[C\]//Proceedings of the IEEE international conference on computer vision. 2015: 1395-1403.

\[3\] B. Epshtein, E. Ofek, and Y. Wexler. Detecting text in natural scenes with stroke width transform. In Proc. of CVPR, 2010.

