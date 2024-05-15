# -*- coding: utf-8 -*-
# @Time: 2024/5/15 13:50
# @Author : Young
# @File : ALSTM_FCN.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.optim as optim
class ALSTM_FCN(nn.Module):
    def __init__(self, num_classes, input_channels=8, time_steps=256, lstm_hidden_units=64, dropout_rate=0.8):
        super(ALSTM_FCN, self).__init__()

        self.lstm = nn.LSTM(input_size=input_channels, hidden_size=lstm_hidden_units, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.lstm_bn = nn.BatchNorm1d(lstm_hidden_units * 2)  # Batch normalization after LSTM

        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=128, kernel_size=8, stride=1, padding=4)
        self.bn1 = nn.BatchNorm1d(128)

        self.conv2 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(256)

        self.conv3 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.global_pooling = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(lstm_hidden_units * 2 + 128, num_classes)

        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_units * 2, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = x.squeeze(1)
        batch_size, channels, time_steps = x.size()

        x_lstm, _ = self.lstm(x.permute(0, 2, 1))  # permute to (batch_size, time_steps, channels)
        x_lstm = self.dropout(x_lstm)
        attention_weights = torch.softmax(self.attention(x_lstm).squeeze(-1), dim=1)
        x_lstm = (x_lstm * attention_weights.unsqueeze(-1)).sum(dim=1)
        x_lstm = self.lstm_bn(x_lstm)

        x_conv = torch.relu(self.bn1(self.conv1(x)))
        x_conv = torch.relu(self.bn2(self.conv2(x_conv)))
        x_conv = torch.relu(self.bn3(self.conv3(x_conv)))
        x_conv = self.global_pooling(x_conv).squeeze(-1)

        x = torch.cat((x_lstm, x_conv), dim=1)
        x = self.fc(x)

        # Apply softmax to the output of the fully connected layer
        x = torch.softmax(x, dim=1)

        return x




