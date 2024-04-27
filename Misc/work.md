# 4.23

## cyj:关于vit

vit+频率：据情感脑机组，他们的输入是(通道，时间，频段）输入，满足类似图像的三维输入。因为他们情感特征就是用五个频段

AffectBCI用于存放情感脑机参考代码,其中AffectBCI/VIT/AITST.py做的vit

## cyj:关于数据集

ssvepformer:两个数据集

Dataset1就是这个仓库的数据，12分类，[Nakanishi et al., 2015](https://www.sciencedirect.com/science/article/pii/S0893608023002319#b28)

Dataset2 [Wang et al., 2016](https://www.sciencedirect.com/science/article/pii/S0893608023002319#b42) 40 目标脑机接口 (BCI) 拼写器*A* *Benchmark* *Dataset* *for*  *SSVEP*-*Based*  *Brain* -*Computer* *Interfaces* 也就是清华数据集http://bci.med.tsinghua.edu.cn/download.html

DDGCNN:两个数据集

Dataset1 [MH et all,2019](https://academic.oup.com/gigascience/article/8/5/giz002/5304369)

Dataset2 [Wang et al., 2016](https://www.sciencedirect.com/science/article/pii/S0893608023002319#b42) 40 目标脑机接口 (BCI) 拼写器*A* *Benchmark* *Dataset* *for*  *SSVEP*-*Based*  *Brain* -*Computer* *Interfaces 也就是清华数据集http://bci.med.tsinghua.edu.cn/download.html

### 清华数据集描述

网址：[清华大学脑机接口研究组 (tsinghua.edu.cn)](http://bci.med.tsinghua.edu.cn/download.html)

* [64, 1500, 40, 6]**。四个维度分别表示“电极索引”、“时间点”、“目标索引”和“区块索引”
* 实验由6个区块组成。每个区块包含40次试验代表40个频率，每个看5s
* 为什么采样时间是1500(250hz采样率x6s）每个试验6s，闪烁5s加上前后的休息的0.5s就总共是6s

该数据集收集了35名健康受试者（17名女性，年龄17-34岁，平均年龄22岁）的SSVEP-BCI记录，重点关注40个字符以不同频率闪烁（8-15.8赫兹，间隔0.2赫兹）。对于每个受试者，**实验由6个区块组成。每个区块包含40次试验**，对应于随机顺序指示的所有40个字符。每次试验开始时有一个视觉提示（一个红色方块），指示目标刺激。提示在屏幕上显示0.5秒。受试者被要求在提示持续时间内尽快将目光转移到目标上。在提示消失后，**所有刺激同时开始闪烁，持续5秒**。刺激消失后，屏幕空白0.5秒，然后开始下一次试验，这样受试者可以在连续试验之间有短暂的休息。每个试验总共持续6秒。为了便于视觉注视，刺激期间目标下方会出现一个红色三角形。在每个区块中，受试者被要求在刺激期间避免眨眼。为了避免视觉疲劳，两个连续区块之间休息几分钟。
使用Synamps2系统（Neuroscan公司）以1000赫兹的采样率获取EEG数据。放大器频率通带范围为0.15赫兹至200赫兹。64个通道覆盖了受试者整个头皮，并按照国际10-20系统对齐。接地电极位于Fz和FPz之间。参考电极位于头顶。电极阻抗保持在10 KΩ以下。为了去除常见的电力线噪音，**在数据记录中应用了一个50赫兹的陷波滤波器**。计算机生成的事件触发器发送到放大器，并在与EEG数据同步的事件通道上记录。
连续的EEG数据被分割成6秒的时段（500毫秒前刺激，5.5秒后刺激开始）。随后，时段被重新采样到250赫兹。因此，每次试验包含1500个时间点。最后，这些数据以双精度浮点值的形式存储在MATLAB中，并命名为受试者索引（即S01.mat，…，S35.mat）。对于每个文件，在MATLAB中加载的数据生成一个名为“data”的4-D矩阵，其维度为**[64, 1500, 40, 6]**。四个维度分别表示“电极索引”、“时间点”、“目标索引”和“区块索引”。电极位置保存在一个“64-channels.loc”文件中。每种SSVEP频率都有6次试验可用。40个目标指数的频率和相位值保存在一个“Freq_Phase.mat”文件中。
所有受试者的信息列在一个“Sub_info.txt”文件中。对于每个受试者，有五个因素，包括“受试者索引”、“性别”、“年龄”、“用手习惯”和“组别”。根据他们在SSVEP-based BCIs中的经验，受试者被分为“有经验”组（8名受试者，S01-S08）和“无经验”组（27名受试者，S09-S35）。

# 4.22

1.论文标题or链接

2.论文摘要(大致内容)

3.网络结构

输入特征：（描述输入形状和输入的分类）

网络结构：描述思想和形状（最好找有代码的）

4.创新点

1.EEG-Based Emotion Recognition via Convolutional Transformer withClass Confusion-Aware Attention

提出了一种新的情绪脑电模型CSET—CCA，该模型能够全面提取脑电信号的长、短时特征，并选择关键的空间信息。

![image-20240423201830517](.\image-20240423201830517.png)

时序特征作为输入，1DCNN用于提取短期信息，Channel SE and Spatial Future Aggregation用注意力机制提取时空特征，Transformer用于提取长期信息

2.A Deep Channel Attention Transformer for Multimodal EEG-EOG-Based Vigilance Estimation

![image-20240423203559905](.\image-20240423203559905.png)

DCAT被细分为四个部分：

**Conv层**、**深度通道注意（DCA）层**、**Transformer层**和**回归层**。具体来说，在Conv层中，2D-CNN用于提取初步的**浅层特征**。浅层特征被放入下一层DCA中，以有效地提取深层局部特征。然后利用Transformer中的多头自注意算法提取被忽略的全局特征。全局特征的提取可以提高模型的鲁棒性。在最后的回归层中，为了完成回归任务，进行了一些处理并使用了线性层。
