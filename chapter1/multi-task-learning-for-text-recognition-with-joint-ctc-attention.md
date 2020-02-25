# multi-task learning for text recognition with joint CTC-attention

## 前言

## 1. MTL详解

![](/assets/MTL-OCR_1.png)

## 1.1 网络概览

MTL的网络结构的后半部分如图1所示，在它之前是一个由CNN组成的特征提取网络，最终得到的Feature Map会以列为时间片为单位输入到RNN中，也就是输入到图像的$$x_1, x_2, ..., x_T$$。RNN之后有两个Head，一个是CTC，另外一个是Attention Decoder，他们两个共同组成网络的损失函数。如果只考虑左侧CTC的话，那么它就是一个标准的CRNN模型。

## 1.2 代码梳理

要梳理MTL的代码流程，我们先要知道网络的一些超参，在源码的README中，作者给出源码的调用方式如下：

```
CUDA_VISIBLE_DEVICES=0 python train.py \
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
	--experiment_name none_resnet_bilstm_ctc \
	--continue_model saved_models/pretrained_model.pth
```

前面四项是用来控制读取数据的超参。5-7个比较直观，第8个`Transformation`是用来控制是否使用STN
