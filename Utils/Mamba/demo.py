import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from mamba_ssm import Mamba
from sklearn.metrics import accuracy_score


# 模拟 SSVEP 数据集
class SSVEPDataset(Dataset):
    def __init__(self, num_samples=1000, num_channels=8, time_points=250, num_classes=12):
        super().__init__()
        self.num_samples = num_samples
        self.num_channels = num_channels
        self.time_points = time_points
        self.num_classes = num_classes

        # 生成随机数据
        self.data = torch.randn(num_samples, num_channels, time_points)
        # 生成随机标签
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 定义 SSVEP 分类模型
class SSVEP_Classifier(nn.Module):
    def __init__(self, num_classes=12, input_size=250):
        super(SSVEP_Classifier, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size

        # 使用 Mamba 模型作为特征提取器
        self.mamba = Mamba(
            d_model=8,  # 调整模型维度以适应您的 SSVEP 数据
            d_state=16,  # 调整 SSM 状态扩展因子
            d_conv=4,  # 调整局部卷积宽度
            expand=2,  # 调整块扩展因子
        )

        # 使用一个全连接层进行分类
        self.fc = nn.Linear(8 * input_size, num_classes)

    def forward(self, x):
        """
        Args:
            x: SSVEP 数据，维度为(batch_size, channels, time_points)

        Returns:
            分类结果，维度为(batch_size, num_classes)
        """
        # Reshape data to match Mamba's input shape
        x = x.permute(0, 2, 1)  # (batch_size, channels, time_points) -> (batch_size, time_points, channels)

        # 使用 Mamba 模型提取特征
        x = self.mamba(x)

        # 将特征向量展平
        x = torch.flatten(x, start_dim=1)

        # 使用全连接层进行分类
        x = self.fc(x)

        return x


# 训练模型
def train_model(model, train_loader, optimizer, criterion, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print(
                f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}")


# 评估模型
def evaluate_model(model, val_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}")
    return accuracy


# 初始化模型、优化器、损失函数和设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SSVEP_Classifier().to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# 创建训练集和验证集
train_dataset = SSVEPDataset(num_samples=800)
val_dataset = SSVEPDataset(num_samples=200)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 训练模型
num_epochs = 10
best_accuracy = 0
for epoch in range(1, num_epochs + 1):
    train_model(model, train_loader, optimizer, criterion, device)
    accuracy = evaluate_model(model, val_loader, device)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        torch.save(model.state_dict(), "best_model.pth")

# 加载最佳模型
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# 评估模型
test_dataset = SSVEPDataset(num_samples=100)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
y_true = []
y_pred = []
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        y_true.extend(target.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

# 计算测试精度
test_accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {test_accuracy:.4f}")