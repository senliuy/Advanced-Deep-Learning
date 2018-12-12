# Neural Architecture Search with Reinforecement Learning

tags: Reinforcement Learning, CNN, RNN

## 前言

CNN和RNN是目前主流的CNN框架，这些网络均是由人为手动设计，然而这些设计是非常困难以及依靠经验的。作者在这篇文章中提出了使用强化学习（Reinforcement Learning）学习一个CNN（后面简称NAS-CNN）或者一个RNN cell（后面简称NAS-RNN），并通过最大化网络在验证集上的精度期望来优化网络，在CIFAR-10数据集上，NAS-CNN的错误率已经逼近当时最好的DenseNet，在TreeBank数据集上，NAS-RNN要优于LSTM。

## 1. 背景介绍

文章提出了Neural Architecture Search（NAS），算法的主要目的是使用强化学习寻找最优网络，包括一个图像分类网络的卷积部分（表示层）和RNN的一个类似于LSTM的cell。由于现在的神经网络一般采用堆叠block的方式搭建而成，这种堆叠的超参数可以通过一个序列来表示。而这种序列的表示方式正是RNN所擅长的工作。

所以，NAS会使用一个RNN构成的控制器（controller）以概率$$p$$随机采样一个网络结构$$A$$，接着在CIFAR-10上训练这个网络并得到其在验证集上的精度$$R$$，然后在使用$$R$$更新控制器的参数，如此循环执行直到模型收敛，如图1所示。

![](/assets/NAS_1.png)

## 2. NAS详细介绍

### 2.1 NAS-CNN

## Reference

\[1\] Zoph B, Le Q V. Neural architecture search with reinforcement learning\[J\]. arXiv preprint arXiv:1611.01578, 2016.

