import numpy as np
from scipy.signal import welch
from torch import nn
from Utils import Constraint
import torch.nn.functional as F
from fightingcv_attention.attention.CBAM import CBAMBlock
import torch
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_classes)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 将输入张量展平为 (batch_size, 32 * 1 * 124)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

#平均叠加
# def calculate_sliding_psd(data, sampling_rate, window_length=None, overlap=None):
#     """
#     计算滑动窗口叠加 PSD
#
#     参数:
#     data (ndarray): 输入数据，形状为 (batch_size, 1, channels, signal_length)
#     sampling_rate (int): 采样率，单位为 Hz
#     window_length (int): 窗口长度，单位为样本数，默认为信号长度的1/4
#     overlap (float): 重叠率，默认为窗口长度的1/2
#
#     返回:
#     average_psd (ndarray): 平均 PSD，形状为 (batch_size, channels, frequency_bins)
#     """
#     batch_size, _, channels, signal_length = data.shape
#
#     if window_length is None:
#         window_length = signal_length // 4  # 默认窗口长度为信号长度的1/4
#
#     if overlap is None:
#         overlap = window_length / 4  # 默认重叠率为窗口长度的1/2
#
#     # 计算重叠率对应的样本数
#     overlap_samples = int(overlap * sampling_rate / 1000)
#
#     # 计算窗口数
#     num_windows = (signal_length - window_length) // (window_length - overlap_samples) + 1
#
#     # 初始化 PSD 结果
#     psd_result = np.zeros((batch_size, channels, num_windows, window_length // 2 + 1))
#
#     # 滑动窗口叠加 PSD
#     for i in range(num_windows):
#         start_idx = i * (window_length - overlap_samples)
#         end_idx = start_idx + window_length
#
#         # 获取当前窗口的信号
#         windowed_signal = data[:, 0, :, start_idx:end_idx]
#
#         # 计算当前窗口的 PSD
#         for b in range(batch_size):
#             for c in range(channels):
#                 _, psd = welch(windowed_signal[b, c], fs=sampling_rate, nperseg=window_length)
#                 psd_result[b, c, i] = psd
#
#     # 叠加 PSD
#     average_psd = np.mean(psd_result, axis=2)
#
#     return average_psd
def calculate_sliding_psd(data, sampling_rate, window_length=None, overlap=None):
    """
    计算滑动窗口叠加 PSD

    参数:
    data (ndarray): 输入数据，形状为 (batch_size, 1, channels, signal_length)
    sampling_rate (int): 采样率，单位为 Hz
    window_length (int): 窗口长度，单位为样本数，默认为信号长度的1/4
    overlap (float): 重叠率，默认为窗口长度的1/4

    返回:
    psd_result (ndarray): PSD 结果，形状为 (batch_size, channels, frequency_bins)
    """
    batch_size, _, channels, signal_length = data.shape

    if window_length is None:
        window_length = signal_length // 4  # 默认窗口长度为信号长度的1/4

    if overlap is None:
        overlap = window_length // 4  # 默认重叠率为窗口长度的1/4

    # 计算重叠率对应的样本数
    overlap_samples = int(overlap * sampling_rate / 1000)

    # 计算窗口数
    num_windows = (signal_length - window_length) // (window_length - overlap_samples) + 1

    # 初始化 PSD 结果
    psd_result = np.zeros((batch_size, channels, window_length // 2 + 1))

    # 滑动窗口叠加 PSD
    for i in range(num_windows):
        start_idx = i * (window_length - overlap_samples)
        end_idx = start_idx + window_length

        # 获取当前窗口的信号
        windowed_signal = data[:, 0, :, start_idx:end_idx]

        # 计算当前窗口的 PSD
        for b in range(batch_size):
            for c in range(channels):
                _, psd = welch(windowed_signal[b, c], fs=sampling_rate, nperseg=window_length)
                # 累加到 PSD 结果中
                psd_result[b, c] += psd

    return psd_result
class ESNet(nn.Module):
    def calculateOutSize(self, model, nChan, nTime):
        '''
            Calculate the output based on input size
            model is from nn.Module and inputSize is an array
        '''
        data = torch.randn(1, 1, nChan, nTime)
        out = model(data).shape
        return out[1:]

    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block, assign different weights to different channels and fuse them
        '''
        block = []
        block.append(Constraint.Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
                                                     max_norm=1.0))
        block.append(nn.BatchNorm2d(num_features=nChan * 2))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))

        layer = nn.Sequential(*block)
        return layer

    def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Enhanced structure block, build a CNN block to absorb data and output its stable feature
        '''
        block = []
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride)))
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))

        layer = nn.Sequential(*block)
        return layer

    def __init__(self, num_channels, T, num_classes):
        super(ESNet, self).__init__()
        self.dropout_level = 0.5
        self.F = [num_channels * 2] + [num_channels * 4]
        self.G = [256,124]
        self.K = 10
        self.S = 2
        self.spatial_conv = self.spatial_block(num_channels, self.dropout_level)
        self.enhanced_conv =   self.enhanced_block(self.F[0], self.F[1], self.dropout_level, self.K, self.S)
        #self.enhanced_conv = self.enhanced_block(256, self.F[1], self.dropout_level, self.K, self.S)
        self.cbam_block1 = CBAMBlock(channel=self.F[0], reduction=16, kernel_size=7)
        self.cbam_block2 = CBAMBlock(channel=self.F[1], reduction=16, kernel_size=7)

        # self.mlp = nn.Sequential(
        #     nn.Linear(32 * 1 * 124, 512),
        #     nn.GELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 12)
        # )
        self.mlp = MLP(input_size=32 * 1 * 60, hidden_size=512, output_classes=12)
        #self.mlp = MLP(input_size=32 * 1 * 21, hidden_size=512, output_classes=12)0.2
        #self.mlp = MLP(input_size=32 * 1 * 60, hidden_size=512, output_classes=12)0.5
        #self.mlp = MLP(input_size=32 * 1 * 508, hidden_size=512, output_classes=12)

    def forward(self, x):
        out1 = self.spatial_conv(x)
        out1 = self.cbam_block1(out1)
        out2 = self.enhanced_conv(out1)
        out2 = self.cbam_block2(out2)
       # 移除维度为 1 的维度，使得输入为 (30, 32, 256)
        #out2 = self.pool(out2)  # 进行平均池化操作
        #out2 = out2.permute(0, 2, 1, 3)
        #out3 = self.cbam_block(out2)
        #out3 = F.dropout(out2, p=0.5, training=self.training)
        #r_out = self.resnet(out3)
        out =self.mlp(out2)

        return out

