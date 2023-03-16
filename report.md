# 2023.03.15-2023.03.20

## ***视频语义分割和2.5D网络差不多，视频语义是不同帧进行处理，而2.5D网络也是不同的切片进行处理***

- #### 视频语义分割：


。

<img src="https://pic4.zhimg.com/80/v2-7054fab074486e5d343c4542b09412ff_720w.webp" alt="img" style="zoom:80%;" />

- #### 2.5D网络

[**2.5D方法所用方法**](https://www.sciencedirect.com/science/article/pii/S0895611122000611#bib21) 

1. **融合三个视角的2D结果**

   从矢状面、冠状面、横断面分别切slice，然后分别训练三个2D的模型，对这三个模型结果进行融合.

   <img src="https://ars.els-cdn.com/content/image/1-s2.0-S0895611122000611-gr2_lrg.jpg" style="zoom:20%;" />

2. **输入相邻层引入层间信息**

   把slice的相邻层也引入网络，这样就能够利用到图像的空间信息了。对于第i层的分割，我们可以把相邻的几层（i+n到i-n）也一块作为多通道输入（类似自然图像的RGB）。 

   <img src="https://ars.els-cdn.com/content/image/1-s2.0-S0895611122000611-gr4_lrg.jpg" style="zoom:20%;" />

3.  **2D3D特征融合**

   对2D与3D网络/模块及提取特征进行了融合，从而兼顾模型的分割效率与分割精度。

   <img src="https://ars.els-cdn.com/content/image/1-s2.0-S0895611122000611-gr5_lrg.jpg" style="zoom:20%;" />

   **[1]** **[Liver tumor segmentation using 2.5D UV-Net with multi-scale convolution](https://www.sciencedirect.com/science/article/pii/S0010482521002183)**

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0010482521002183-gr4_lrg.jpg" style="zoom:20%;" />

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0010482521002183-gr5_lrg.jpg" style="zoom:20%;" />



DR模块使用一个3D卷积，z轴没有填零操作，使用卷积核大小为3x3x3.我们将x和y空间分辨率填零，因此空间分辨率不变，而z轴分辨率将从3降低到1，归因于z轴非零填充卷积。实现3D-2D转换的关键步骤是非零填充3x3x3卷积。我们在深度维度设置卷积大小为3，不填零，这样卷积核会自动学习一个参数来融合3层连续切片，从而实现降维。

**[2]** **[2.5D lightweight RIU-Net for automatic liver and tumor segmentation from CT](https://www.sciencedirect.com/science/article/pii/S1746809422000891)**

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S1746809422000891-gr3_lrg.jpg" style="zoom:20%;" />

相邻切片作为输入通道，经过第一个卷积把通道做处理

**[3]** **[Automatic Segmentation of Vestibular Schwannoma from T2-Weighted MRI by Deep Spatial Attention with Hardness-Weighted Loss (MICCAI 2019)](https://arxiv.org/abs/1906.03906)**

<img src="https://media.springernature.com/full/springer-static/image/chp%3A10.1007%2F978-3-030-32245-8_30/MediaObjects/490275_1_En_30_Fig2_HTML.png?as=webp" style="zoom:50%;" />

最开始的两个层次 (L1-L2)使用的是2D卷积，后面的三个层次 (L3-L5)使用的是3D卷积。这样做的原因在于，该任务的层面内分辨率大概是层面间分辨率的4倍，经过两个层次的2D卷积之后，层面内和层面间的分辨率就一致了，因此在之后就可以使用3D CNN进行处理。

上述的过程其实可以举个例子理解：假设现在的输入是一个MRI图像，zyx三个轴上的spacing为(4,1,1)，spacing可以理解为上述的分辨率。在2.5D CNN中，遍历z轴，可以得到一层一层的2D图像，这些2D图像经过两个阶段的最大池化之后，yx两轴上的分辨率就和z上的分辨率一致了，因此后续就可以采用3D CNN进行处理。

**[4]** **[A 2.5D Cancer Segmentation for MRI Images Based on U-Net](https://ieeexplore.ieee.org/document/8612509)**

<img src="https://ieeexplore.ieee.org/mediastore_new/IEEE/content/media/8609878/8612498/8612509/8612509-fig-2-source-large.gif" style="zoom:80%;" />

输入图像为来自三个方向(表示为1、2、3)的二维patch，在训练阶段分别输入到U-Net框架中。U网由收缩路径(左侧)和扩展路径(右侧)组成。每个灰色方框对应于一个多通道特征地图。白色方框表示复制的要素地图。在箭头上方表示不同的操作，然后对三个训练模型的分割结果进行平均，得到最终的分割结果。

**[5]** **[Automatic Segmentation of Gross Target Volume of Nasopharynx Cancer using Ensemble of Multiscale Deep Neural Networks with Spatial Attention](https://www.sciencedirect.com/science/article/pii/S0925231221001077)**

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0925231221001077-gr2_lrg.jpg" style="zoom:20%;" />

来自不同尺度图像训练的几个独立网络的概率映射被融合以生成初始分割结果。然后，我们选取最大的连通区域以获得最终的分割结果。

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0925231221001077-gr3_lrg.jpg" style="zoom:20%;" />

骨架遵循U-Net 编码-解码器设计，一共有9个卷积块，每个卷积块包含两个卷积层，然后是BN和Leaky ReLU。除第一个卷积块外，每个卷积块前面都有一个PE块，因为第一个卷积块的输入是带一个通道的输入图像，PE块的主要目的是获取通道信息，同时考虑空间信息。注意力模块位于解码器中PE块和底部块的前面，以捕获小GTV区域的空间信息。最后一层由卷积层和提供分割概率的softmax函数组成。

图像输入切片大小和输出切片大小都是16×64×64，即它们具有相同的slice大小

**[6]**  [**Kidney tumor segmentation from computed tomography images using DeepLabv3+ 2.5D model**](https://www.sciencedirect.com/science/article/pii/S0957417421015797)

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0957417421015797-gr9_lrg.jpg" style="zoom:25%;" />

最初，堆叠在 3 个通道上的切片作为红-绿-蓝 (RGB) 图像输入。然后，它们通过编码器-解码器结构。在编码阶段，从 DPN-131 编码器中提取的特征用作空洞卷积和空洞空间金字塔池 (ASPP) 操作的输入。空洞卷积在多个尺度上捕获特征（Rate 6、Rate 12、和Rate 18)，而 ASPP 是一种池化操作，有助于计算不同尺度的对象。

**[7]** **[Ψ-Net: Focusing on the border areas of intracerebral hemorrhage on CT images](https://www.sciencedirect.com/science/article/pii/S0169260719322333)**

<img src="https://ars.els-cdn.com/content/image/1-s2.0-S0169260719322333-gr1_lrg.jpg" style="zoom:25%;" />



### Idea

[MorphMLP: An Efficient MLP-Like Backbone for Spatial-Temporal Representation Learning](https://arxiv.org/abs/2111.12527)

**把这个作为Backbone试试看，把每个病例截取三个或者更高的切片作为网络的输入，使用Unet结构做端到端的分割**

**作者分析了在时空表示学习中使用MLP的主要挑战。**

1、从空间角度来看，当前的MLP类模型缺乏对语义细节的进一步理解。

2、从时间角度来看，学习帧之间的长期依赖关系。

为了应对这些挑战，作者提出了一种有效的类似MLP的体系结构，即MorphMLP，用于视频表示学习。它由两个关键层组成，即MorphFCs和MorphFCt，分别利用空间和时间建模上的简明FC操作。MorphFCs可以有效地捕捉空间中的核心语义。此外，MorphFCt可以自适应地捕获帧上的长距离依赖关系。

**方法**

**用于空间建模的MorphFCs**

MorphFCs可以分层扩展FC的感受野，使其从小区域扩展到大区域，而且MorphFCs在水平和垂直路径上独立处理每一帧视频。以水平的(图中的蓝色块)为例。

![img](https://pic1.zhimg.com/80/v2-a37eea7f64e2c1d08330a8746ce42c34_720w.webp)

图4:空间维度上的MorphFCs。

具体来说就是，给定一帧输入视频X，该视频已被投影到一系列tokens中，首先沿水平方向拆分X，这样就得到了分割块。然后将每个chunk扁平化为1维向量，并应用FC权值矩阵W对每个chunk进行变换，得到

![img](https://pic4.zhimg.com/80/v2-fcbd1bffa57dff8e9af1acec1acac1eb_720w.webp)

经过特征变换后，将所有块Yki重构回原始维度Y。垂直的方式(图4中的绿色块)也是如此，只是沿着垂直方向将tokens序列分割开来。为了使组之间沿通道维度进行通信，还应用FC层来单独处理每个tokens。最后，将水平、垂直和通道特征相加得到输出。

随着网络的深化，区块长度L会逐级增加，从而使FC过滤器从小到大的空间区域逐步发现更多的核心语义。

**用于时间建模的MorphFCt**

除了水平和垂直方向，本文还提出了另一个MorphFCt。MorphFCt利用简单的FC层以较低的计算成本捕获长期时间信息。如图6所示，给定一个输入视频剪辑tokensX，首先将X沿着通道维数分成若干组(每组中有D个通道)，以减少计算成本，得到Xk。

![img](https://pic3.zhimg.com/80/v2-2fd1f41cdf0dfb4df1f953ffb59511d2_720w.webp)

图6:时间维度上的MorphFCt。

然后，对于每个空间位置s，将所有帧的特征连接到一个块Xks。接下来应用F C矩阵W，对时域特征进行变换，得到

![img](https://pic3.zhimg.com/80/v2-ffaca17abec5da9bfe1a4ebb665933c2_720w.webp)

最后，将所有块Yks重构回原始tokens维度，并输出Y。通过这种方式，FC过滤器可以简单地在块中沿着时间维度聚合tokens关系，以对时间依赖性建模。

**时空MorphMLP块**

在MorphFCs和MorphFCt的基础上，在视频领域提出了一种分解时空的MorphMLP块，用于高效的视频表示学习。如图7所示，MorphMLP块按顺序包含了MorphFCt、MorphFCs和MLP模块。

![img](https://pic3.zhimg.com/80/v2-7d88bd125ef99ea40d4462d39292c5ea_720w.webp)

图:时空MorphMLP块。

联合时空优化是困难的，而对空间和时间进行分解建模可以显著降低计算成本。因此，作者将时间和空间MorphFCs层置于顺序样式中。各模块前使用LN层，MorphFCt和MLP模块后使用标准residual连接。

另外，在MorphFCs层的原始输入和输出特征之间增加了一个跳跃式residual连接(红线)，而不是在MorphFCs层后应用标准的residual连接。发现这样的联系可以使训练更加稳定。

**网络体系结**

在视频识别方面，如图8所示，将时空MorphMLP块进行分层叠加，构建网络。给定一个视频序列X，MorphMLP首先对视频片段进行补丁嵌入，得到一个tokens序列。然后，存在四个连续的阶段，每个阶段包含两个MorphMLP块。特征通过同一阶段内的各层时，特征大小保持不变。在每个阶段结束时，除最后一个阶段外，扩大通道维数，并按2的比例下采样特征的空间分辨率。

![img](https://pic3.zhimg.com/80/v2-b00ee162cf8a6520ae9b8bc3c3a6cc66_720w.webp)

图8:MorphMLP结构















































# 2023.03.06-2023.03.14

### paper

- [Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks [CVPR 2023]](https://github.com/JierunChen/FasterNet)

  快速卷积：和[GhostNet[CVPR 2020]](https://zhuanlan.zhihu.com/p/109325275)差不多，改了后面倒置残差块

  PConv：特征先分为两半，一半用卷积，一半直接连接，最后拼接。Backbone：PConv + （倒置残差块：Dwconv或Conv1x1）

- [MedViT: A Robust Vision Transformer for Generalized Medical Image Classification（医学分类）](https://arxiv.org/abs/2302.09462)

​		Backbone：Grouped conv 3x3，最后一层用传统vit（q，k，v：k和v做一次平均池化，把FFN换为ResNet结构（Dwconv3x3）

- [RepUX-Net：Scaling Up 3D Kernels with Bayesian Frequency Re-parameterization for Medical Image Segmentation](chttps://arxiv.org/abs/2303.05785)

  3DUX Net升级版，把卷积核升到21x21x21，在训练期间对卷积核参数加入贝叶斯权重
