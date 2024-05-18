import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from Utils.Mamba.mamba_ssm import Mamba
from sklearn.metrics import accuracy_score

# 定义 SSVEP 分类模型
class SSVEPMamba(nn.Module):
    def __init__(self, num_classes=12, input_size=250):
        super(SSVEPMamba, self).__init__()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_classes = num_classes
        self.input_size = input_size

        # 使用 Mamba 模型作为特征提取器
        self.mamba = Mamba(
            d_model=8,  # 调整模型维度以适应您的 SSVEP 数据
            d_state=16,  # 调整 SSM 状态扩展因子
            d_conv=5,  # 调整局部卷积宽度
            expand=2,  # 调整块扩展因子
        ).to(device)

        # 使用一个全连接层进行分类
        self.fc = nn.Linear(8 * input_size, num_classes).to(device)

    def forward(self, x):
        """
        Args:
            x: SSVEP 数据，维度为(batch_size, channels, time_points)

        Returns:
            分类结果，维度为(batch_size, num_classes)
        """
        x = x.squeeze(1)
        # Reshape data to match Mamba's input shape
        x = x.permute(0, 2, 1)  # (batch_size, channels, time_points) -> (batch_size, time_points, channels)

        # 使用 Mamba 模型提取特征
        x = self.mamba(x)

        # 将特征向量展平
        x = torch.flatten(x, start_dim=1)

        # 使用全连接层进行分类
        x = self.fc(x)

        return x


