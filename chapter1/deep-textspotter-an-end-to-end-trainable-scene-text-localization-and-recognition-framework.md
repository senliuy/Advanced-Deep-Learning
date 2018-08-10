# Deep TextSpotter: An End-to-End Trainable Scene Text Localization and Recognition Framework

## 前言

Deep TextSpotter的创新点并不多，基本上遵循了传统OCR或者物体检测的两步走的流程，即先进行场景文字检测，再进行文字识别。在这个算法中，检测模块基于[YOLOv2](https://senliuy.gitbooks.io/advanced-deep-learning/content/chapter1/yolo9000-better-faster-stronger.html)\[2\]，网络结果使用了STN

## Reference

\[1\] Bušta M, Neumann L, Matas J. Deep textspotter: An end-to-end trainable scene text localization and recognition framework\[C\]//Computer Vision \(ICCV\), 2017 IEEE International Conference on. IEEE, 2017: 2223-2231.

\[2\] Redmon J, Farhadi A. YOLO9000: better, faster, stronger\[J\]. arXiv preprint, 2017.

\[4\] Connectionist Temporal Classification : Labelling Unsegmented Sequence Data with Recurrent Neural Networks. Graves, A., Fernandez, S., Gomez, F. and Schmidhuber, J., 2006. Proceedings of the 23rd international conference on Machine Learning, pp. 369--376. DOI: 10.1145/1143844.1143891

