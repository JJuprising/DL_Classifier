import torch
from einops import rearrange


from torch import nn
from Utils import Constraint
import torch.nn.functional as F
from fightingcv_attention.attention.CBAM import CBAMBlock


class GRU(nn.Module):
    '''
        Employ the Bi-GRU to learn the reliable dependency between spatio-temporal features
    '''
    def __init__(self, input_size, hidden_size):
        super(GRU, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, bidirectional=True, num_layers=1)

    def forward(self, x):
        b, c, T = x.size()
        x = x.view(x.size(-1), -1, c)  # (b, c, T) -> (T, b, c)
        r_out, _ = self.rnn(x)  # r_out shape [time_step * 2, batch_size, output_size]
        out = r_out.view(b, 2 * T * c, -1)
        return out
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_classes)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 将输入张量展平为 (batch_size, 7936)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

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
        self.K = 10
        self.S = 2
        self.spatial_conv = self.spatial_block(num_channels, self.dropout_level)
        self.enhanced_conv =   self.enhanced_block(self.F[0], self.F[1], self.dropout_level, self.K, self.S)
        #self.enhanced_conv = self.enhanced_block(256, self.F[1], self.dropout_level, self.K, self.S)
        self.cbam_block = CBAMBlock(channel=self.F[1], reduction=16, kernel_size=7)
        self.rnn  =GRU(input_size=self.F[1], hidden_size=self.F[1])
        # self.mlp = nn.Sequential(
        #     nn.Linear(32 * 1 * 124, 512),
        #     nn.GELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, 256),
        #     nn.GELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(256, 12)
        # )
        self.mlp = MLP(input_size=7936, hidden_size=512, output_classes=12)
    def forward(self, x):
        out1 = self.spatial_conv(x)
        out2 = self.enhanced_conv(out1)
       # 移除维度为 1 的维度，使得输入为 (30, 32, 256)
        #out2 = self.pool(out2)  # 进行平均池化操作
        #out2 = out2.permute(0, 2, 1, 3)

        out3 = self.cbam_block(out2)
        #out3 = F.dropout(out3, p=0.5, training=self.training)
        out3 = out3.squeeze(2)
        #r_out = self.resnet(out3)


        out =self.mlp(out3)


        return out
