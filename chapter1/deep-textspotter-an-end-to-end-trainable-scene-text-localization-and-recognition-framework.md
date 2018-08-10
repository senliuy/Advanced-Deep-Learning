# Deep TextSpotter: An End-to-End Trainable Scene Text Localization and Recognition Framework

## 前言

Deep TextSpotter的创新点并不多，基本上遵循了传统OCR或者物体检测的两步走的流程，即先进行场景文字检测，再进行文字识别。在这个算法中，检测模块基于[YOLOv2](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/yolo9000-better-faster-stronger.html)\[2\]，识别模块基于[STN](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/spatial-transform-networks.html)\[3\]，损失函数则使用了精度的[CTC](https://senliuy.gitbooks.io/advanced-deep-learning/content/di-er-zhang-ff1a-xu-lie-mo-xing/connectionist-temporal-classification-labelling-unsegmented-sequence-data-with-recurrent-neural-networks.html)\[4\]。这些知识点已分别在本书的第四章，第五章和第二章进行了解析，算法细节可参考具体内容或者阅读论文。这里不在对上面三个算法的细节再做重复，只会对Deep TextSpotter的流程做一下梳理和解释。

Deep TextSpotter的一个创新点是将NMS放到了识别之后，使用识别置信度替代了传统的检测置信度。

## Deep TextSpotter解析


## Reference

\[1\] Bušta M, Neumann L, Matas J. Deep textspotter: An end-to-end trainable scene text localization and recognition framework\[C\]//Computer Vision \(ICCV\), 2017 IEEE International Conference on. IEEE, 2017: 2223-2231.

\[2\] Redmon J, Farhadi A. YOLO9000: better, faster, stronger\[J\]. arXiv preprint, 2017.

\[3\] Jaderberg M, Simonyan K, Zisserman A. Spatial transformer networks\[C\]//Advances in neural information processing systems. 2015: 2017-2025.

\[4\] Connectionist Temporal Classification : Labelling Unsegmented Sequence Data with Recurrent Neural Networks. Graves, A., Fernandez, S., Gomez, F. and Schmidhuber, J., 2006. Proceedings of the 23rd international conference on Machine Learning, pp. 369--376. DOI: 10.1145/1143844.1143891

