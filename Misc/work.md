# 4.23

vit+频率：据情感脑机组，他们的输入是(通道，时间，频段）输入，满足类似图像的三维输入。因为他们情感特征就是用五个频段

AffectBCI用于存放情感脑机参考代码,其中AffectBCI/VIT/AITST.py做的vit

# 4.22

1.论文标题or链接

2.论文摘要(大致内容)

3.网络结构

输入特征：（描述输入形状和输入的分类）

网络结构：描述思想和形状（最好找有代码的）

4.创新点

1.EEG-Based Emotion Recognition via Convolutional Transformer withClass Confusion-Aware Attention

提出了一种新的情绪脑电模型CSET—CCA，该模型能够全面提取脑电信号的长、短时特征，并选择关键的空间信息。

![image-20240423201830517](F:\DL_Classifier\Misc\image-20240423201830517.png)

时序特征作为输入，1DCNN用于提取短期信息，Channel SE and Spatial Future Aggregation用注意力机制提取时空特征，Transformer用于提取长期信息

2.A Deep Channel Attention Transformer for Multimodal EEG-EOG-Based Vigilance Estimation

![image-20240423203559905](F:\DL_Classifier\Misc\image-20240423203559905.png)

DCAT被细分为四个部分：

**Conv层**、**深度通道注意（DCA）层**、**Transformer层**和**回归层**。具体来说，在Conv层中，2D-CNN用于提取初步的**浅层特征**。浅层特征被放入下一层DCA中，以有效地提取深层局部特征。然后利用Transformer中的多头自注意算法提取被忽略的全局特征。全局特征的提取可以提高模型的鲁棒性。在最后的回归层中，为了完成回归任务，进行了一些处理并使用了线性层。