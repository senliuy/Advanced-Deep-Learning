# multi-task learning for text recognition with joint CTC-attention

## 前言

## 1. MTL详解

![](/assets/MTL-OCR_1.png)

## 1.1 网络概览

MTL的网络结构的后半部分如图1所示，在它之前是一个由CNN组成的特征提取网络，最终得到的Feature Map会以列为时间片为单位输入到RNN中，也就是输入到图像的$$x_1, x_2, ..., x_T$$。RNN之后有两个Head，一个是CTC，另外一个是Attention Decoder，他们两个共同组成网络的损失函数。如果只考虑左侧CTC的话，那么它就是一个标准的CRNN模型。

## 1.2 代码梳理

### 1.2.1 执行脚本

要梳理MTL的代码流程，我们先要知道网络的一些超参，在源码的README中，多任务模型的调用方式如下（源文件README有误）：

```bash
CUDA_VISIBLE_DEVICES=0 python mtl_train.py \
    --train_data data/synch/lmdb_train \
    --valid_data data/synch/lmdb_val \
    --select_data / --batch_ratio 1 \
    --sensitive \
    --num_iter 400000 \
    --output_channel 512 \
    --hidden_size 256 \
    --Transformation None \
    --FeatureExtraction ResNet \
    --SequenceModeling BiLSTM \
    --Prediction CTC \
    --mtl \
    --without_prediction \
    --experiment_name none_resnet_bilstm_ctc \
    --continue_model saved_models/pretrained_model.pth
```

前面四项是用来控制读取数据的超参。5-7个比较直观，第8个`--Transformation`是用来控制是否使用STN，第9个`--FeatureExtraction`是提取图像特征的网络结构，第10个`--SequenceModeling`是图1中‘Shared Encoder’的结构。`--Prediction`是预测的时候选择图1中的CTC的分支或者是Attention Decoder分支。`--mtl`是选择模型的训练方式，是选择一个任务进行训练还是训练多任务模型。`--without_prediction`是指模型加载的方式，是否需要预测模块。

除了上面列出的，在`train.py`或者`mtl_train.py`文件中还有很多可以调整的超参，例如优化方式中涉及的学习策略，学习率；数据处理方式的图像尺寸等。

作者在这里面有个错误，如果要执行多任务模型，需要执行`python mtl_train.py`而不是上面给出的`python train.py`。因为在作者的代码中，`train.py`调用的是`model.py`，而`mtl_train.py`则调用的是`mtl_model.py`。

### 1.2.2 网络模型

在上面提到了执行多任务训练要调用`mtl_model.py`文件，下面我们来梳理一下这个文件。这个文件的第一个重点在源码的第19-24行，在这里我们确认模型是否要使用STN。STN的实现定义在`./modules/transformation.py`文件中。

```py
""" Transformation """
if opt.Transformation == 'TPS':
    self.Transformation = TPS_SpatialTransformerNetwork(
    F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)
else:
    print('No Transformation module specified')
```

第26-36用于选择模型的特征提取网络，卷积网络的具体细节定义在`./modules/feature_extraction.py`文件中：

```py
""" FeatureExtraction """
if opt.FeatureExtraction == 'VGG':
    self.FeatureExtraction = VGG_FeatureExtractor(opt.input_channel, opt.output_channel)
elif opt.FeatureExtraction == 'RCNN':
    self.FeatureExtraction = RCNN_FeatureExtractor(opt.input_channel, opt.output_channel)
elif opt.FeatureExtraction == 'ResNet':
    self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
else:
    raise Exception('No FeatureExtraction module specified')
self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
```

第38-46行是网络的RNN部分，作者只给出了双向LSTM的支持

```py
""" Sequence modeling"""
if opt.SequenceModeling == 'BiLSTM':
    self.SequenceModeling = nn.Sequential(
        BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
        BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
    self.SequenceModeling_output = opt.hidden_size
else:
    print('No SequenceModeling module specified')
    self.SequenceModeling_output = self.FeatureExtraction_output
```

第48-53行则是给出多任务模型的CTC输出和Attention输出

```py
""" Prediction """
if opt.mtl:
    self.CTC_Prediction = nn.Linear(self.SequenceModeling_output, opt.ctc_num_class)
    self.Attn_Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)
else:
    raise Exception('Prediction is not Joint CTC-Attention')
```

Attention预测的计算方式在`./modules/prediction.py`文件中，它采用了Image Caption式的生成模型来进行预测，核心代码在第43-66行

```py
if is_train:
    for i in range(num_steps):
        # one-hot vectors for a i-th char. in a batch
        char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
        # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
        hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
        output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
        probs = self.generator(output_hiddens)
```

我们先看训练时的分支，在预测第$$i+1$$个时间片的输出时，我们首先对Ground Truth的前$$i$$个时间片的内容进行one-hot编码，之后通过attention_cell得到当前时间片的输出。遍历的时间片的总个数超参由`--batch_max_length`定义的，源码中的值是25，也就是说Attention Decoder分支最大支持25个字符的识别。

```py
else:
    if torch.cuda.is_available():
        targets = torch.cuda.LongTensor(batch_size).fill_(0)  # [GO] token
        probs = torch.cuda.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0)
    else:
        targets = torch.LongTensor(batch_size).fill_(0)  # [GO] token
        probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0)

        for i in range(num_steps):
            char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
            hidden, alpha = self.attention_cell(hidden, batch_H, char_onehots)
            probs_step = self.generator(hidden[0])
            probs[:, i, :] = probs_step
            _, next_input = probs_step.max(1)
            targets = next_input
```

解码过程和训练过程略有不同，它使用的预测的结果（最后三行）作为上个时间片的输入编码，而训练的时候使用的是GroundTruth。

AttentioCell的核心代码在`prediction.py`文件的81-91行，它是一个基于单向LSTM的生成器，它的输入有三个，其中`prev_hidden`是上一个时间片的隐层状态，`batch_H`是双向LSTM的输出，也就是图1中的$$h_1, h_2, ..., h_L$$，`char_onehots`是GroundTruth的前$$i-1$$个时间片（训练时）或者之前的预测结果（测试时）的one-hot编码。

```py
 def forward(self, prev_hidden, batch_H, char_onehots):
    # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
    batch_H_proj = self.i2h(batch_H)
    prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
    e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

    alpha = F.softmax(e, dim=1)
    context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel
    concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
    cur_hidden = self.rnn(concat_context, prev_hidden)
    return cur_hidden, alpha
```
