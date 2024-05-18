import numpy as np
import torch
from fightingcv_attention.attention.CBAM import CBAMBlock

from torch.fft import fft
from torch import nn, einsum, Tensor
from torch.nn import Module, ModuleList
from einops import rearrange, reduce, repeat, pack, unpack
from einops.layers.torch import Rearrange
from fightingcv_attention.attention.ECAAttention import ECAAttention

from torch.nn import init

from etc.global_config import config

ws = config["data_param_12"]["ws"]
Fs = config["data_param_12"]["Fs"]
# 自注意力
from Utils import Constraint


#
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3,
                                   dim=-1)  # x:(30, 16, 220) qkv:(tuple:3)[0:(30, 16, 64),1:(30, 16, 64),2:(30, 16, 64)]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads),
                      qkv)  # q:(30, 8, 16, 8) (batch_size, channels, heads * dim_head)

        # 点乘操作
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # dots:(30, 8, 16, 16)， transpose是转置

        attn = self.attend(dots)  # attn:(30, 8, 16, 16)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)  # out:(30, 8, 16, 8)
        out = rearrange(out, 'b h n d -> b n (h d)')  # out:(30, 16, 64)
        return self.to_out(out)  # out:(30, 16, 220)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def complex_spectrum_features(segmented_data, FFT_PARAMS):
    sample_freq = FFT_PARAMS[0]
    time_len = FFT_PARAMS[1]
    resolution, start_freq, end_freq = 0.2, 8, 64
    NFFT = round(sample_freq / resolution)
    fft_index_start = int(round(start_freq / resolution))
    fft_index_end = int(round(end_freq / resolution)) + 1
    sample_point = int(sample_freq * time_len)
    # 将 segmented_data 移动到 CPU
    segmented_data_cpu = segmented_data.cpu()

    # 转换为 NumPy 数组
    segmented_data_np = segmented_data_cpu.numpy()

    # 进行 FFT
    fft_result = np.fft.fft(segmented_data_np, axis=-1, n=NFFT) / (sample_point / 2)
    real_part = np.real(fft_result[:, :, :, fft_index_start:fft_index_end - 1])
    imag_part = np.imag(fft_result[:, :, :, fft_index_start:fft_index_end - 1])
    features_data = np.concatenate([real_part, imag_part], axis=-1)
    return features_data


class convAttention(nn.Module):
    def __init__(self, token_num, token_length, kernal_length=31, dropout=0.5):
        super().__init__()

        self.att2conv = nn.Sequential(
            nn.Conv1d(token_num, token_num, kernel_size=kernal_length, padding=kernal_length // 2, groups=1),
            nn.LayerNorm(token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.att2conv(x)
        return out


class freqFeedForward(nn.Module):
    def __init__(self, token_length, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(token_length, token_length),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PreNorm(nn.Module):
    def __init__(self, token_dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(token_dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class convTransformer(nn.Module):
    def __init__(self, depth, token_num, token_length, kernal_length, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(token_length, convAttention(token_num, token_length, kernal_length, dropout=dropout)),
                PreNorm(token_length, freqFeedForward(token_length, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Transformer(Module):
    # 空间滤波器模块，为每个通道分配不同的权重并融合它们
    def spatial_block(self, nChan, dropout_level):
        '''
           Spatial filter block,assign different weight to different channels and fuse them
        '''
        block = []
        block.append(Constraint.Conv2dWithConstraint(in_channels=1, out_channels=nChan * 2, kernel_size=(nChan, 1),
                                                     max_norm=1.0))  # (30, 16, 1, 256)
        block.append(nn.BatchNorm2d(num_features=nChan * 2))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    # 增强模块，使用CNN块吸收数据并输出其稳定特征
    def enhanced_block(self, in_channels, out_channels, dropout_level, kernel_size, stride):
        '''
           Enhanced structure block,build a CNN block to absorb data and output its stable feature
        '''
        block = []
        block.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, kernel_size),
                               stride=(1, stride)))  # input(30, 16, 8, 256) output(30, 32, 8, 124)
        block.append(nn.BatchNorm2d(num_features=out_channels))
        block.append(nn.PReLU())
        block.append(nn.Dropout(dropout_level))
        layer = nn.Sequential(*block)
        return layer

    '''
    T: 时间序列长度
    '''
    def __init__(self, T, depth, heads, chs_num, class_num, tt_dropout, ff_dropout, dim_thead=8, dim_fhead=8, dim=560):
        super().__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # 时间序列的网络层
        self.attentionEncoder = ModuleList([])
        self.fc = nn.Linear(T, class_num)
        self.Linear = nn.Linear(560, dim)
        self.F = [chs_num * 2] + [chs_num * 4]
        self.K = 10
        self.S = 2
        output_dim = int((T - 1 * (self.K - 1) - 1) / self.S + 1)
        net = []
        net.append(self.spatial_block(chs_num, ff_dropout))  # （30， 16， 1， 256）
        net.append(self.enhanced_block(self.F[0], self.F[1], ff_dropout,
                                       self.K, self.S))  # (30, 32, 1, 124)

        self.conv_layers = nn.Sequential(*net).to(device)
        for _ in range(depth):
            self.attentionEncoder.append(ModuleList([
                # ECAAttention(kernel_size=3),
                # SimplifiedScaledDotProductAttention(d_model=dim, h=8),
                Attention(chs_num, dim_head=dim_thead, heads=heads, dropout=tt_dropout),
                # Attention(token_num=self.F[0], token_length=self.F[1]),
                nn.LayerNorm(chs_num),
                FeedForward(chs_num, hidden_dim=dim_fhead, dropout=ff_dropout),
                nn.LayerNorm(chs_num)
            ]).to(device))

        self.mlp_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(ff_dropout),
            nn.Linear(T * chs_num, class_num * 6),
            nn.LayerNorm(class_num * 6),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(class_num * 6, class_num)
        ).to(device)


        # 结果融合层
        self.fusion_layer = nn.Conv1d(2, 1, kernel_size=1).to(device)

    # 有两个子网络，一个子网络处理时间序列，一个子网络处理频谱序列
    def forward(self, x):

        # 处理时间序列
        # 通过线性层对序列进行裁剪
        # x_t = complex_spectrum_features(x, FFT_PARAMS=[Fs, ws])
        # x_t = torch.from_numpy(x_t)
        # x_t = x_t.to(x.device)
        # x_t = torch.Tensor(x_t.squeeze(1))
        # x_t = x_t.float()

        x_t = x
        # x_t = self.conv_layers(x_t)
        # 消除为1的维度
        x_t = x_t.squeeze(1)
        x_t = rearrange(x_t, 'b c t -> b t c')
        for attn, attn_post_norm, ff, ff_post_norm in self.attentionEncoder:
            x_t = attn(x_t) + x_t
            x_t = attn_post_norm(x_t)
            x_t = ff(x_t) + x_t
            x_t = ff_post_norm(x_t)

        # x_t = x_t.mean(dim=1)  # (30, 220)
        # x_t = self.fc(x_t)  # (30, 12)
        x_t = self.mlp_head(x_t)

        return x_t