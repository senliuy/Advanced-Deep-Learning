# Group Normalization

## 前言

Group Normalization（GN）是何恺明提出的一种归一化策略，它是介于Layer Normalization（LN）\[2\]和 Instance Normalization（IN）\[3\]之间的一种折中方案，图1最右。它通过将**通道**数据分成几组计算归一化统计量，因此GN也是和批量大小无关的算法，因此可以用在batchsize比较小的环境中。作者在论文中指出GN要比LN和IN的效果要好。

![](/assets/GN_1.png)

## Reference

\[1\] Wu Y, He K. Group normalization\[J\]. arXiv preprint arXiv:1803.08494, 2018.

\[2\] Ba J L, Kiros J R, Hinton G E. Layer normalization\[J\]. arXiv preprint arXiv:1607.06450, 2016.

\[3\] Vedaldi V L D U A. Instance Normalization: The Missing Ingredient for Fast Stylization\[J\]. arXiv preprint arXiv:1607.08022, 2016.

