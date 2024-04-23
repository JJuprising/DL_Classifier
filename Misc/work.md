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

时序特征作为输入，1DCNN用于提取短期信息，Channel SE and Spatial Future Aggregation用注意力机制提取时空特征，Transformer用于提取长期信息s

