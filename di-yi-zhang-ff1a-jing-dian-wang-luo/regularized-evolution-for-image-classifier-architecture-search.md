# Regularized Evolution for Image Classifier Architecture Search

tags: AutoML, Aging Evolution, CNN

## 前言

在讲解AmoebaNet之前，先给大家讲一个故事：在一个物质资源非常匮乏的外星球上居住着一种只能进行无性繁殖的绝顶聪明的外星人，这个物质匮乏的星球上的资源匮乏到只够养活$$P$$个外星人。然而外星人为了种族进化还是要产生新的后代的，那么谁有资格产生后代呢，最厉害的那个外星人A提出基因最好的外星人才有资格产生后代。其它外星人不高兴了，因为他们担心整个星球都是A家族的人进而破坏了基因多样性。于是他们提出了一个折中方案，每次随机抽$$S$$个候选者参与竞争，里面最厉害的才有资格产生后代。如果A被抽中了，那A是里面最厉害的，就让A产生后代，如果A没有被抽中，也给其它不是很优秀的外星人一个机会。这样即保证了优秀基因容易产生更多后代，也保证了星球上的基因多样性。接着，由于产生了一个新的外星人，但是星球上的资源有限，所以必须杀死一个外星人给新的外星人留位置。A又提议了，我们杀死最笨的那个吧，其它外星人又不高兴了，生孩子的时候你A的概率最高，杀人的时候轮不到你了，久而久之这个星球上不全是你的后代了吗。经过商议，它们提出了一个最简单的方法：杀死岁数最大的那个。

故事讲完，我们开始正文。在我们之前介绍的NAS系列算法中，模型结构的搜索均是通过强化学习实现的。这篇要介绍的AmoebaNet是通过遗传算法的进化策略（Evolution）实现的模型结构的学习过程。该算法的主要特点是在进化过程中引入了年龄的概念，使进化时更倾向于选择更为年轻的性能好的结构，这样确保了进化过程中的多样性和优胜劣汰的特点，这个过程叫做年龄进化（Aging Evolution，AE）。作者为他的网络取名AmoebaNet，Amoeba中文名为变形体，是对形态不固定的生物体的统称，作者也是借这个词来表达AE拥有探索更广的搜索空间的能力。AmoebaNet取得了当时在ImageNet数据集上top-1和top-5的最高精度。

## 1. AmoebaNet算法详解

### 1.1 搜索空间

AmoebaNet使用的是和[NASNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/learning-transferable-architectures-for-scalable-image-recognition.html){{"zoph2018learning"|cite}}相同的搜索空间。仿照NASNet的思想，AmoebaNet也是学习两个Cell：\(1\) Normal Cell，\(2\) Reduction Cell，在这里两个Cell是完全独立的。然后通过重复堆叠Normal Cell和Reduction Cell的形式我们可以得到一个完整的网络，如图1左所示。其中Normal Cell中步长始终为1，因此不会改变Feature Map的尺寸，Reduction Cell的步长为2，因此会将Feature Map的尺寸降低为原来的1/2。因此我们可以连续堆叠更多的Normal Cell以获得更大的模型容量（不能堆叠Reduction Cell），如图1左侧图中Normal Cell右侧的$$\times$$N的符号所示。在堆叠Normal Cell时，AmoebaNet使用了shortcut的机制，即一个Normal Cell的输入来自上一层，另外一个输入来自上一层的上一层，如图1中间部分。

<figure>
<img src="/assets/AmoebaNet_1.png" alt="图1：AmoebaNet的搜索空间：(左)由Normal Cell和Reduction Cell构成的完整网络；(中)Normal Cell内部的跳跃连接；(右)一个Normal/Reduction Cell 的内部结构"/>
<figcaption>图1：AmoebaNet的搜索空间：(左)由Normal Cell和Reduction Cell构成的完整网络；(中)Normal Cell内部的跳跃连接；(右)一个Normal/Reduction Cell 的内部结</figcaption>
</figure>


在每个卷积操作中，我们需要学习两个参数：

1. 卷积操作的类型：类型空间参考NASNet。
2. 卷积核的输入：从该Cell中所有可以选择的Feature Map选择两个，每个Feature Map选择一个操作，通过合并这两个Feature Map得到新的Feature Map。最后将所有没有扇出的Feature Map合并作为最终的输出。上面所说的合并是单位加操作，因此Feature Map的个数不会改变。举例说明一下这个过程，根据图1中的跳跃连接，每个Cell有两个输入，对应图1右的0，1。那么第一个操作（红圈部分）选择0，1作为输入以及average池化和max池化作为操作构成新的Feature Map 2。接着第二个操作可以从（0，1，2）中选择两个作为输入，形成Feature Map 3，依次类推可以得到Feature Map 4，5，6，7。

最终AmoebaNet仅仅有两个变量需要决定，一个是每个Feature Map的卷积核数量$$F$$，另一个是堆叠的Normal Cell的个数$$N$$，这两个参数作为了人工设定的超参数，作者也实验了$$N$$和$$F$$的各种组合。

### 1.2 Aging Evolution

AmoebaNet的进化算法Aging Evolution（AE）如图2所示。

![](/assets/AmoebaNet_2.png)

在介绍代码之前，我们先看三条血淋淋的社会现实：

1. 优秀的父代更容易留下后代；
2. 年轻人比岁数大的更受欢迎；
3. 无论多么优秀的人都会有死去的一天。

这三个现实正是我从图2中的代码总结出来的，也是AE拿来进化网络的动机，现在我们来看看AE是如何反应这三点的。

第1行是使用队列（queue）初始化一个`population`变量。在AE中每个变量都有一个固定的生存周期，这个生存周期便是通过队列来实现的，因为队列的“先进先出”的特征正好符合AE的生命周期的特征。`population`的作用是保存当前的存活模型，而只有存活的模型才有产生后代的能力。

第2行的`history`是用来保存所有训练好的模型。

第3行的作用是使用随机初始化的形式产生第一代存活的模型，个数正是循环的终止条件$$P$$。$$P$$的值在实验中给出的个数有20，64，100三个，其中$$P=100$$的时候得到了最优解。

`while` 循环中（4-7行）便是随机初始化一个网络，然后训练并在验证集上测试这个网络的精度，最后将网络的架构和精度保存到`population`和`history`变量中。这里所有的模型评估都是在CIFAR-10上完成的。首先注意保存的是**架构**而不是模型，所以保存的变量的内容不会很多，因此并不会占用特别多的内存。其次由于`population`是一个队列，所以需要从右侧插入。而`history`插入变量时则没有这个要求。

第9行的第二个`while`循环表示的是进化的时长，即不停的向`history`中添加产生的优秀模型，直到`history`中模型的数量达到$$C$$个。$$C$$的值越大就越有可能进化出一个性能更为优秀的模型，我们也可以选择在模型开始收敛的结束进化。在作者的实验中$$C=20,000$$。

第10行的`sample`变量用于从存活的样本中随机选取$$S$$个模型进行竞争，第三个while循环中的代码（11-15行）便是用于随机选择候选父带。

第16行代码是从这$$S$$个模型只有精度最高的产生后代。这个有权利产生后代的变量命名为`parent`。论文实验中$$S$$的值设定的值有2，16，20，25，50，其中效果最好的值是25。

第17行是使用变异（mutation）操作产生父代的子代，变量名是`child`。变异的操作包括随机替换卷积操作（op mutation）和随机替换输入Feature Map（hidden state mutation），如图3所示。在每次变异中，只会进行一次变异操作，亦或是操作变异，亦或是输入变异。

<figure>
<img src="/assets/AmoebaNet_3.png" alt="图3：AmoebaNet的变异操作：(上)Hidden State Mutation改变模型的输入Feature Map；(下)Op Mutation改变一个卷积操作"/>
<figcaption>图3：AmoebaNet的变异操作：(上)Hidden State Mutation改变模型的输入Feature Map；(下)Op Mutation改变一个卷积操作</figcaption>
</figure>

第18-20行依次是训练这个子代网络架构并将它依次插入`population`和`history`中。

第21-22行是从`population`顶端移除最老的架构，这一行也是AE最核心的部分。另外一种很多人想要使用的策略是移除效果最差的那个，这个方法在论文中叫做Non Aging Evolution（NAE）。作者这么做的动机是如果一个模型效果足够好，那么他有很大概率在他被淘汰之前已经在`population`中留下了自己的后代。如果按照NAE的思路淘汰差样本的话，`population`中留下的样本很有可能是来自一个共同祖先，所以AE的方法得到的架构具有更强大的多样性。而NAE得到的架构由于多样性非常差，使得架构非常容易陷入局部最优值。这种情况在遗传学中也有一个名字：近亲繁殖。

最后一行代码是从所有训练过的模型中选择最好的那个作为最终的输出。

再回去看看开始的那个故事，讲的就是AE算法。

### 1.3 AmoebaNet网络结构

通过上面的进化策略，产生的网络结构如图4所示，作者将其命名为AmoebaNet-A：

<figure>
<img src="/assets/AmoebaNet_4.png" alt="图4：AmoebaNet-A结构：(左)由Normal Cell和Reduction Cell构成的AmoebaNet-A；(中)Normal Cell；(下)Reduction Cell"/>
<figcaption>图4：AmoebaNet-A结构：(左)由Normal Cell和Reduction Cell构成的AmoebaNet-A；(中)Normal Cell；(右)Reduction Cell</figcaption>
</figure>

在图4中还有两个要手动设置的参数，一个参数是连续堆叠的Normal Cell的个数$$N$$，另外一个是卷积核的数量。在第一个Reduction之前卷积核的数量是$$F$$，后面每经过一次Reduction，卷积核的数量$$\times$$2。这两个参数是需要人工设置的超参数。

实验结果表明，当AmoebaNet的参数数量（$$N=6$$，$$F=190$$）达到了NASNet以及[PNASNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/progressive-neural-architecture-search.html)\[3\]的量级（80MB+）时，AmoebaNet和其它两个网络在ImageNet上的精度是非常接近的。虽然AmoebaNet得到的网络和NASNet以及PNASNet非常接近，但是其基于AE的收敛速度是要明显快于基于强化学习的收敛速度的。

而最好的AmoebaNet的参数数量达到了469M时，AmoebaNet-A取得了目前在ImageNet上最优的测试结果。但是不知道是得益于AmoebaNet的网络结构还是其巨大的参数数量带来的模型容量的巨大提升。

最后作者通过一些传统的进化算法得到了AmoebaNet-B，AmoebaNet-C，AmoebaNet-D三个模型。由于它们的效果并不如AmoebaNet-A，所以这里不再过多介绍，感兴趣的同学去读论文的附录D部分。

从模型的精度上来看Aging Evolution（AE）和RL（Reinfrocement Learning）得到的同等量级参数的架构在ImageNet上的表现是几乎相同的，因此我们无法冒然的下结论说AE得到的模型要优于AL。但是AE的收敛速度快于RL是非常容易从实验结果中看到的。另外作者也添加了一个Random Search（RS）做对照实验，三个方法的收敛曲线图如图5所示：

<figure>
<img src="/assets/AmoebaNet_5.png" alt="图5：AE，RL及RS在收敛速度上的对比曲线"/>
<figcaption>图5：AE，RL及RS在收敛速度上的对比曲线</figcaption>
</figure>


## 2. 总结

这篇文章在NASNet的搜索空间的基础上尝试着使用进化策略来搜索网络的架构，并提出了一个叫做Aging Evolution（AE）的进化策略。AE可以看做一个带有正则项的进化策略，它使得训练过程可以更加关注于网络架构而非模型参数。无论是拿强化学习和AE对比还是拿NAE和AE对比，AE在收敛速度上均有明显的优势。同时AE的算法非常简单，正如我们在图2中的伪代码所示的它只有$$P,C,S$$三个参数，对比之下RL需要构建一个由LSTM构成的控制器，AE的超参数明显少了很多。

最后，作者得到了一个目前为止在ImageNet上分类效果最好的AmoebaNet-A，虽然它的参数到达了4.69亿个。


## Reference

\[1\] Real E, Aggarwal A, Huang Y, et al. Regularized evolution for image classifier architecture search\[J\]. arXiv preprint arXiv:1802.01548, 2018.

\[2\] Zoph B, Vasudevan V, Shlens J, et al. Learning transferable architectures for scalable image recognition\[J\]. arXiv preprint arXiv:1707.07012, 2017, 2\(6\).

\[3\] Liu C, Zoph B, Shlens J, et al. Progressive neural architecture search\[J\]. arXiv preprint arXiv:1712.00559, 2017.

