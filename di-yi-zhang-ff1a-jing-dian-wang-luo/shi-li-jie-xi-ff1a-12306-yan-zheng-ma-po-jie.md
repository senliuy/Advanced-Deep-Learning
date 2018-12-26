# 实例解析：12306验证码破解

** 注意：本文章只适用于技术交流，切勿用于商业用途！**

## 前言

又到一年一度的抢票季，12306是一个让人头疼的网站，一大因素是因为其变态的验证码机制。在这篇文章中我们将使用深度学习来破解12306验证码，并使用深度学习常见的策略，例如Dropout，迁移学习，数据增强等trick一步步提升模型识别率。

这里我们使用简单易用的Keras作为开源工具。上面提到的Trick也都是Keras自带的功能，所以在这里我们也会介绍一些Keras的基本用法，下面全部实验见链接：[https://github.com/senliuy/12306\_crack](https://github.com/senliuy/12306_crack) 。

## 1. 数据分析

12306的验证码是从8个图片中找到要求的物体，如图1所示。

![](/assets/12306_1.png)

我统计了1000个样本，发现12306的类别数其实只有80类，它们的类别以及对应的统计个数如表1

| 安全帽: 18 | 本子: 18 | 鞭炮: 10 | 冰箱: 18 | 菠萝: 12 | 苍蝇拍: 12 | 茶几: 12 | 茶盅: 13 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 创可贴: 9 | 刺绣: 16 | 打字机: 10 | 档案袋: 11 | 电饭煲: 16 | 电线: 12 | 电子秤: 11 | 调色板: 15 |
| 订书机: 14 | 耳塞: 16 | 风铃: 20 | 高压锅: 9 | 公交卡: 14 | 挂钟: 15 | 锅铲: 10 | 海报: 8 |
| 海鸥: 16 | 海苔: 13 | 航母: 14 | 黑板: 14 | 红豆: 14 | 红酒: 11 | 红枣: 13 | 护腕: 7 |
| 话梅: 10 | 剪枝: 10 | 金字塔: 10 | 锦旗: 13 | 卷尺: 13 | 开瓶器: 15 | 口哨: 12 | 蜡烛: 11 |
| 辣椒酱: 7 | 篮球: 13 | 老虎: 9 | 铃铛: 15 | 龙舟: 12 | 漏斗: 15 | 路灯: 17 | 绿豆: 11 |
| 锣: 11 | 蚂蚁: 10 | 毛线: 15 | 蜜蜂: 7 | 棉棒: 15 | 排风机: 13 | 牌坊: 12 | 盘子: 14 |
| 跑步机: 20 | 啤酒: 14 | 热水袋: 11 | 日历: 14 | 沙包: 13 | 沙拉: 13 | 珊瑚: 8 | 狮子: 8 |
| 手掌印: 11 | 薯条: 12 | 双面胶: 17 | 拖把: 2 | 网球拍: 10 | 文具盒: 9 | 蜥蜴: 12 | 药片: 13 |
| 仪表盘: 18 | 印章: 15 | 樱桃: 12 | 雨靴: 13 | 蒸笼: 11 | 中国结: 6 | 钟表: 12 | 烛台: 15 |

从上面的统计中我们可以看出，12306的验证码的破解工作可以转换成一个80类的分类问题，而这正是我们所擅长的，因为在物体分类领域我们尝试了太多的实验，例如MNIST，CIFAR-10，CIFAR-100等。

很幸运的是我们不需要人工标注数据，Kaggle上提供了一份开源的12306已标注图片数据集，注册之后即可下载，链接见：[https://www.kaggle.com/libowei/12306-captcha-image](https://www.kaggle.com/libowei/12306-captcha-image) 。

在搭建模型之前我们需要将数据集分成训练集，验证集和测试集三个部分，我采用的策略是分别随机的从每类物体中各随机选取20个作为验证集和测试集。为了保证实验结果的可复现，我已将分好的数据集上传到百度云，下载链接见：[https://pan.baidu.com/s/1LksQZes3C1bM8ubIKUF6ag](https://pan.baidu.com/s/1LksQZes3C1bM8ubIKUF6ag) 。

## 2. 破解过程

物体分类的代码可以简单分成三个部分：

1. 网络搭建；
2. 数据读取；
3. 模型训练。

但是在上面的三步中每一步都存在一些超参数，怎么设置这些超参数是一个有经验的算法工程师必须掌握的技能。我们会在下面的章节中介绍每一步的细节，并给出我自己的经验和优化策略。

### 2.1 网络搭建

我们搭建一个分类网络时，可以使用上面几篇文章中介绍的经典的网络结构，也可以自行搭建。当自行搭建分类网络时，可以使用下面几步：

1. 堆积卷积操作（Conv2D）和最大池化操作（MaxPooling2D），第一层需要指定输入图像的尺寸和通道数；
2. Flatten\(\)用于将Feature Map展开成特征向量；
3. 之后接全连接层和激活层，注意多分类应该使用softmax激活函数。

自行搭建网络时，我有几个经验：

1. 通道数的数量取 $$2^n$$；
2. 每次MaxPooling之后通道数乘2；
3. 最后一层Feature Map的尺寸不宜太大也不宜太小\(7-20直接是个不错的选择\)；
4. 输出层和Flatten\(\)层往往需要加最少一个隐层用于过渡特征；
5. 根据计算Flatten\(\)层的节点数量设计隐层节点的个数。

下面代码是我搭建的一个分类网络，结构非常简单。

```py
model_simple = models.Sequential()
model_simple.add(layers.Conv2D(32, (3,3), padding='same', activation='relu', input_shape = (66,66,3)))
model_simple.add(layers.MaxPooling2D((2,2)))
model_simple.add(layers.Conv2D(64, (3,3), padding='same', activation='relu'))
model_simple.add(layers.MaxPooling2D((2,2)))
model_simple.add(layers.Conv2D(128, (3,3), padding='same', activation='relu'))
model_simple.add(layers.MaxPooling2D((2,2)))
model_simple.add(layers.Flatten())
model_simple.add(layers.Dense(1024, activation='relu'))
model_simple.add(layers.Dense(80, activation='softmax'))
```

或者我们也可以使用之前提到的经典卷积网络，这里以VGG-16为例。Keras提供了VGG-16在ImageNet-2012（1000类）上的分类网络，由于输出节点数不一样，这里我们只取VGG-16的表示层，代码如下。

```py
model_rand_VGG16 = models.Sequential()
rand_VGG16 = VGG16(weights=None, include_top=False, input_shape=(224,224,3))
model_rand_VGG16.add(rand_VGG16)
model_rand_VGG16.add(layers.Flatten())
model_rand_VGG16.add(layers.Dense(1024, activation='relu'))
model_rand_VGG16.add(layers.Dropout(0.25))
model_rand_VGG16.add(layers.BatchNormalization()) # 梯度爆炸
model_rand_VGG16.add(layers.Dense(80, activation='softmax'))
model_rand_VGG16.summary()
```

在上面代码中`VGG16()`函数用于调用Keras自带的VGG-16网络，`weights`参数指定网络是否使用迁移学习模型，值为`None`时表示随机初始化，值为`ImageNet`时表示使用ImageNet数据集训练得到的模型。`include_top`参数表示是否使用后面的输出层，我们确定了只使用表示层，所以取值为`False`。`input_shape`表示输入图片的尺寸，由于VGG-16会进行5次降采样，所以我们使用它的默认输入尺寸$$224\times224\times3$$，所以输入之前会将输入图片放大。

### 2.2 数据读取

Keras提供了多种读取数据的方法，我们推荐使用**生成器**的方式。在生成器中，Keras在训练模型的同时把下一批要训练的数据预先读取到内存中，这样会节约内存，有利于大规模数据的训练。Keras的生成器的初始化是`ImageDataGenerator`类，它有一些自带的数据增强的方法，我们会在2.5节进行介绍。

在这个实验中我们将不同的分类置于不同的目录之下，因此读取数据时使用的是`flow_from_directory()`函数，训练数据读取代码如下（验证和测试相同）：

```py
train_data_gen = ImageDataGenerator(rescale=1./255)
train_generator = train_data_gen.flow_from_directory(train_folder, 
                                                     target_size=(66, 66), 
                                                     batch_size=128, 
                                                     class_mode='categorical')
```

我们已近确定了是分类任务，所以`class_mode`的值取`categorical`。

### 2.3 模型训练

当我们训练模型时首先我们要确定的优化策略和损失函数，这里我们选择了`Adagrad`作为优化策略，损失函数选择多分类交叉熵`categorical_crossentropy`。由于我们使用了生成器读取数据，所以要使用fit\_generator来向模型喂数据，代码如下。

```py
model_simple.compile(loss='categorical_crossentropy', optimizer=optimizers.Adagrad(lr=0.01), metrics=['acc'])
history_simple = model_simple.fit_generator(train_generator, 
                                            steps_per_epoch=128, 
                                            epochs=20, 
                                            validation_data=val_generator)
```

经过20个Epoch之后，模型会趋于收敛，损失值曲线和精度曲线见图2，此时的测试集的准确率是0.8275。从收敛情况我们可以分析到模型此时已经过拟合，我们需要一些策略来解决这个问题。

![](/assets/12306_2.png)

### 2.4 Dropout

Dropout[1]一直是解决过拟合非常有效的策略。在使用dropout时丢失率的设置是一个技术活，丢失率太小的话Dropout不能发挥其作用，丢失率太大的话模型会不容易收敛，甚至会一直震荡。在这里我在后面的全连接层和最后一层卷积层各加一个丢失率为0.25的Dropout。收敛曲线和精度曲线见图3，我们可以看出过拟合问题依旧存在，但是略有减轻，此时得到的测试集准确率是0.83375。

![](/assets/12306_3.png)

### 2.5 数据增强

Keras提供在调用ImageDataGenerator类的时候根据它的参数添加数据增强策略，在进行数据扩充时，我有几点建议：

1. 扩充策略的设置要建立在对数据集充分的观测和理解上；
2. 正确的扩充策略能增加样本数量，大幅减轻过拟合的问题；
3. 错误的扩充策略很有可能导致模型不好收敛，更严重的问题是使训练集和测试集的分布更加不一致，加剧过拟合的问题；
4. 往往开发者需要根据业务场景自行实现扩充策略。

下面代码是我使用的数据增强的几个策略。

```py
train_data_gen_aug = ImageDataGenerator(rescale=1./255,
                                        horizontal_flip = True, 
                                        zoom_range = 0.1,
                                       width_shift_range= 0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       rotation_range=5)
train_generator_aug = train_data_gen_aug.flow_from_directory(train_folder, 
                                                     target_size=(66, 66), 
                                                     batch_size=128, 
                                                     class_mode='categorical')
```

其中```rescale=1./255```参数的作用是对图像做归一化，归一化是一个在几乎所有图像问题上均有用的策略；```horizontal_flip = True```，增加了水平翻转，这个是适用于当前数据集的，但是在OCR等方向水平翻转是不能用的；其它的包括缩放，平移，旋转等都是常见的数据增强的策略，此处不再赘述。

结合Dropout，数据扩充可以进一步减轻过拟合的问题，它的收敛曲线和精度曲线见图4，此时得到的测试集准确率是0.84875。

![](/assets/12306_4.png)

### 2.6 迁移学习

在2.1节中我们介绍了搭建模型中有自行搭建和使用经典模型两种策略，通过调用Keras的applications模块我们可以找到Keras中在ImageNet上训练过的几个模型，他们依次是：

* [Xception](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/xception-deep-learning-with-depthwise-separable-convolutions.html)
* [VGG16](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/very-deep-convolutional-networks-for-large-scale-image-recognition.html)
* [VGG19](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/very-deep-convolutional-networks-for-large-scale-image-recognition.html)
* [ResNet50](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/deep-residual-learning-for-image-recognition.html)
* [InceptionV3](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/going-deeper-with-convolutions.html)
* [InceptionResNetV2](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/going-deeper-with-convolutions.html)
* [MobileNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/mobilenetxiang-jie.html)
* [DenseNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/densely-connected-convolutional-networks.html)
* [NASNet](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-yi-zhang-ff1a-jing-dian-wang-luo/neural-architecture-search-with-reinforecement-learning.html)

使用经典模型往往和迁移学习配合使用效果更好，所谓迁移学习是将训练好的任务A（最常用的是ImageNet）的模型用于当前任务的网络的初始化，然后在自己的数据上进行微调。该方法在数据集比较小的任务上往往效果很好。Keras提供用户自定义迁移学习时哪些层可以微调，哪些层不需要微调，通过layer.trainable设置。Keras使用迁移学习提供的模型往往比较深，容易产生梯度消失或者梯度爆炸的问题，建议添加BN层。最好的策略是选择好适合自己任务的网络后自己使用ImageNet数据集进行训练。

以VGG-16为例，其使用迁移学习的代码如下。第一次运行这段代码时需要下载供迁移学习的模型，因此速度会比较慢，请耐心等待。

```py
model_trans_VGG16 = models.Sequential()
trans_VGG16 = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
model_trans_VGG16.add(trans_VGG16)
model_trans_VGG16.add(layers.Flatten())
model_trans_VGG16.add(layers.Dense(1024, activation='relu'))
model_trans_VGG16.add(layers.BatchNormalization())
model_trans_VGG16.add(layers.Dropout(0.25))
model_trans_VGG16.add(layers.Dense(80, activation='softmax'))
model_trans_VGG16.summary()
```

它的收敛曲线和精度曲线见图5，此时得到的测试集准确率是0.774375，此时迁移学习的效果反而不如我们前面随便搭建的网络。在这个问题上导致迁移学习模型表现效果不好的原因有两个：

1. VGG-16的网络过深，在12306验证码这种简单的验证码上容易过拟合；
2. 由于```include_top```的值为```False```，所以网络的全连接层是随机初始化的，导致开始训练时损失值过大，带偏已经训练好的表示层。

![](/assets/12306_5.png)

为了防止表示层被带偏，我们可以将Keras中的层的```trainable```值设为```False```来达到此目的。结合之前介绍的数据增强和Dropout，最终我们得到的收敛曲线和精度曲线见图6，此时得到的测试集准确率是0.91625。

```py
for layer in trans_VGG16.layers:
    layer.trainable = False
```

![](/assets/12306_6.png)

## 总结

在这篇文章中，我们将12306网站验证码的破解工作转换成了一个经典的多分类问题，并通过深度学习和一些trick将识别率提高到了91.625%。也许这个精度不能让您满意，此时你需要自己做一些工作来提升精度，以下是可能有用的几点：

1. 更合理的网络结构：网络层数，节点数量，卷积、池化、全连接的搭配；
2. 更好的缓解过拟合的策略：Dropout数量和位置，正则项；
3. 更合理的数据扩充策略；
4. 更合适的迁移学习模型以及冻结策略；
5. 者其它自己了解的其它优化方向的策略（例如自适应学习率，L1正则，Attention等）；
6. 采集并标注更多的数据。

91%的精度远远不是我们利用这批数据能达到的最高精度，写作这篇文章的目的是为了探讨深度学习在物体分类中的使用方法和针对训练日志优化模型的过程，如果你有更好的策略欢迎在评论区给出。

## Reference

[1] Hinton G E, Srivastava N, Krizhevsky A, et al. Improving neural networks by preventing co-adaptation of feature detectors[J]. arXiv preprint arXiv:1207.0580, 2012.


