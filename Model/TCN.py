import torch
import torch.nn as nn


class Chomp1d(nn.Module):
    """
    用于从输入张量中删除多余的 padding 元素。
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    定义一个 TCN 的基本模块，包含两个膨胀卷积层、ReLU 激活函数、dropout 层以及残差连接。
    """

    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """
    构建完整的 TCN 模型，通过堆叠多个 `TemporalBlock` 来实现。
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SSVEP_TCN(nn.Module):
    """
    基于 TCN 的 SSVEP 分类模型，可以灵活处理不同的通道数、时间点数和分类数。
    """

    def __init__(self, num_channels, num_classes, tcn_channels=[64, 128, 256], kernel_size=3, dropout=0.5):
        super(SSVEP_TCN, self).__init__()

        # 将输入数据的维度转换为 TCN 需要的形状
        self.reshape_layer = nn.Sequential(
            nn.Linear(num_channels, tcn_channels[0]),
            nn.ReLU()
        )

        # TCN 模块
        self.tcn = TemporalConvNet(tcn_channels[0], tcn_channels, kernel_size=kernel_size, dropout=dropout)

        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # 分类器
        self.classifier = nn.Linear(tcn_channels[-1], num_classes)

    def forward(self, x):
        x = x.squeeze(1)
        # 输入 x 的形状: (batch_size, channels, time_points)

        # Reshape for TCN
        x = self.reshape_layer(x.transpose(1, 2)).transpose(1, 2)

        # Apply TCN
        x = self.tcn(x)

        # Global average pooling
        x = self.global_avg_pool(x).squeeze(-1)

        # Classification
        x = self.classifier(x)

        return x

if __name__ == '__main__':
    # 示例用法
    num_channels = 8  # 通道数
    num_time_points = 256  # 时间点数
    num_classes = 12  # 分类数

    # 初始化模型
    model = SSVEP_TCN(num_channels, num_classes)

    # 创建示例输入数据
    x = torch.randn(30, 1, num_channels, num_time_points)  # batch_size = 30

    # 推理
    output = model(x)

    print(output.shape)  # torch.Size([30, 12])