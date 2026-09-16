import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class CrossAttentionModule(nn.Module):
    def __init__(self, in_channels1, in_channels2, reduction_ratio=8):
        super().__init__()
        self.reduced_dim = in_channels1 // reduction_ratio
        self.query_conv = nn.Conv2d(in_channels1, self.reduced_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels2, self.reduced_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels2, in_channels1, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input1, input2):
        batch_size, C, H, W = input1.size()

        # 尺寸对齐
        if input2.size()[-2:] != input1.size()[-2:]:
            input2 = F.interpolate(input2, size=(H, W), mode='bilinear', align_corners=False)

        # 计算query, key, value
        query = self.query_conv(input1).view(batch_size, -1, H * W).permute(0, 2, 1)  # (B, N, C')
        key = self.key_conv(input2).view(batch_size, -1, H * W)  # (B, C', N)
        value = self.value_conv(input2).view(batch_size, -1, H * W)  # (B, C, N)

        # 计算注意力
        energy = torch.bmm(query, key)  # (B, N, N)
        attention = self.softmax(energy)

        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)

        return self.gamma * out + input1  # 残差连接


class AttentionResNet34(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        # 主干网络
        self.resnet = models.resnet34(weights=None)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_class)

        # 注意力模块（与权重文件完全匹配的结构）
        self.attention3 = CrossAttentionModule(256, 256)  # 对应layer3
        self.attention4 = CrossAttentionModule(512, 512)  # 对应layer4

        # 参考图像下采样路径（与权重文件完全匹配）
        self.er_three = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=8, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.er_four = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, x_ref):
        # 检查输入尺寸
        assert x.size()[2:] == x_ref.size()[2:], \
            f"Input shapes mismatch: img {x.shape}, ref {x_ref.shape}"

        # 标准ResNet前处理
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # layer1和layer2无注意力
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)

        # layer3 + 注意力
        x = self.resnet.layer3(x)
        x_ref = self.er_three(x_ref)
        x = self.attention3(x, x_ref)

        # layer4 + 注意力
        x = self.resnet.layer4(x)
        x_ref = self.er_four(x_ref)
        x = self.attention4(x, x_ref)

        # 分类头
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return self.resnet.fc(x)