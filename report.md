# 2023.03.06-2023.03.14
### paper

- [Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks [CVPR 2023]](https://github.com/JierunChen/FasterNet)

  快速卷积：和[GhostNet[CVPR 2020]](https://zhuanlan.zhihu.com/p/109325275)差不多，改了后面倒置残差块

  PConv：特征先分为两半，一半用卷积，一半直接连接，最后拼接。Backbone：PConv + （倒置残差块：Dwconv或Conv1x1）

- [MedViT: A Robust Vision Transformer for Generalized Medical Image Classification（医学分类）](chrome-extension://cdonnmffkdaoajfknoeeecmchibpmkmg/assets/pdf/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F2302.09462.pdf)

​		Backbone：Grouped conv 3x3，最后一层用传统vit（q，k，v：k和v做一次平均池化，把FFN换为ResNet结构（Dwconv3x3）

- [RepUX-Net：Scaling Up 3D Kernels with Bayesian Frequency Re-parameterization for Medical Image Segmentation](chrome-extension://cdonnmffkdaoajfknoeeecmchibpmkmg/assets/pdf/web/viewer.html?file=https%3A%2F%2Farxiv.org%2Fpdf%2F2303.05785.pdf)

  3DUX Net升级版，把卷积核升到21x21x21，在训练期间对卷积核参数加入贝叶斯权重
