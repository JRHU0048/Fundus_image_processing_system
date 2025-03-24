# """ Full assembly of the parts to form the complete network """

# from .unet_parts import *


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         self.inc = (DoubleConv(n_channels, 64))
#         self.down1 = (Down(64, 128))
#         self.down2 = (Down(128, 256))
#         self.down3 = (Down(256, 512))
#         factor = 2 if bilinear else 1
#         self.down4 = (Down(512, 1024 // factor))
#         self.up1 = (Up(1024, 512 // factor, bilinear))
#         self.up2 = (Up(512, 256 // factor, bilinear))
#         self.up3 = (Up(256, 128 // factor, bilinear))
#         self.up4 = (Up(128, 64, bilinear))
#         self.outc = (OutConv(64, n_classes))

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         logits = self.outc(x)
#         return logits

#     def use_checkpointing(self):
#         self.inc = torch.utils.checkpoint(self.inc)
#         self.down1 = torch.utils.checkpoint(self.down1)
#         self.down2 = torch.utils.checkpoint(self.down2)
#         self.down3 = torch.utils.checkpoint(self.down3)
#         self.down4 = torch.utils.checkpoint(self.down4)
#         self.up1 = torch.utils.checkpoint(self.up1)
#         self.up2 = torch.utils.checkpoint(self.up2)
#         self.up3 = torch.utils.checkpoint(self.up3)
#         self.up4 = torch.utils.checkpoint(self.up4)
#         self.outc = torch.utils.checkpoint(self.outc)

# 上面是原来的模型
#====================================================================================
# 下面是修改后的模型:增加了两层下采样，一共6层
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(mid_channels),  # 使用 InstanceNorm2d 替代 BatchNorm2d
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),  # 使用 InstanceNorm2d 替代 BatchNorm2d
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Simple convolution."""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# class UNet(nn.Module):
#     def __init__(self, n_channels, n_classes, bilinear=False):
#         super(UNet, self).__init__()
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.bilinear = bilinear

#         # 增加额外的下采样和上采样层
#         self.inc = DoubleConv(n_channels, 64)
#         self.down1 = Down(64, 128)
#         self.down2 = Down(128, 256)
#         self.down3 = Down(256, 512)
#         self.down4 = Down(512, 1024)
#         self.down5 = Down(1024, 2048)  # 新增的下采样层
#         factor = 2 if bilinear else 1
#         self.down6 = Down(2048, 4096 // factor)  # 新增的下采样层
#         self.up1 = Up(4096, 2048 // factor, bilinear)
#         self.up2 = Up(2048, 1024 // factor, bilinear)
#         self.up3 = Up(1024, 512 // factor, bilinear)
#         self.up4 = Up(512, 256 // factor, bilinear)
#         self.up5 = Up(256, 128 // factor, bilinear)  # 新增的上采样层
#         self.up6 = Up(128, 64, bilinear)  # 新增的上采样层
#         self.outc = OutConv(64, n_classes)

#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#         x6 = self.down5(x5)  # 新增的下采样层
#         x7 = self.down6(x6)  # 新增的下采样层
#         x = self.up1(x7, x6)
#         x = self.up2(x, x5)
#         x = self.up3(x, x4)
#         x = self.up4(x, x3)
#         x = self.up5(x, x2)  # 新增的上采样层
#         x = self.up6(x, x1)  # 新增的上采样层
#         logits = self.outc(x)
#         return logits

#     def use_checkpointing(self):
#         self.inc = torch.utils.checkpoint(self.inc)
#         self.down1 = torch.utils.checkpoint(self.down1)
#         self.down2 = torch.utils.checkpoint(self.down2)
#         self.down3 = torch.utils.checkpoint(self.down3)
#         self.down4 = torch.utils.checkpoint(self.down4)
#         self.down5 = torch.utils.checkpoint(self.down5)  # 新增的下采样层
#         self.down6 = torch.utils.checkpoint(self.down6)  # 新增的下采样层
#         self.up1 = torch.utils.checkpoint(self.up1)
#         self.up2 = torch.utils.checkpoint(self.up2)
#         self.up3 = torch.utils.checkpoint(self.up3)
#         self.up4 = torch.utils.checkpoint(self.up4)
#         self.up5 = torch.utils.checkpoint(self.up5)  # 新增的上采样层
#         self.up6 = torch.utils.checkpoint(self.up6)  # 新增的上采样层
#         self.outc = torch.utils.checkpoint(self.outc)


# 定义DANN（Domain Adversarial Neural Network）结构
class DomainClassifier(nn.Module):
    def __init__(self, feature_dim=512):
        super(DomainClassifier, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 1024)
        self.fc2 = nn.Linear(1024, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 梯度反转层类
import torch.autograd as autograd
class GradientReversalFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def forward(self, x, alpha):
        return GradientReversalFunction.apply(x, alpha)

# 修改UNet类以支持DANN
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False, feature_dim=512):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # 定义UNet的主干网络
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.down5 = Down(1024, 2048)  # 新增的下采样层
        self.down6 = Down(2048, 4096 // factor)  # 新增的下采样层
        
        # # 定义用于领域适应的特征提取层
        # self.feature_extractor = nn.Sequential(
        #     nn.Conv2d(1024, feature_dim, kernel_size=1),
        #     nn.ReLU(inplace=True),
        #     nn.AdaptiveAvgPool2d((1,1))
        # )
        
        # # 定义DANN的领域分类器
        # self.domain_classifier = DomainClassifier(feature_dim)
        
        # 定义上采样层和输出层
        # self.up1 = Up(1024, 512 // factor, bilinear)
        # self.up2 = Up(512, 256 // factor, bilinear)
        # self.up3 = Up(256, 128 // factor, bilinear)
        # self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

        self.up1 = Up(4096, 2048 // factor, bilinear)
        self.up2 = Up(2048, 1024 // factor, bilinear)
        self.up3 = Up(1024, 512 // factor, bilinear)
        self.up4 = Up(512, 256 // factor, bilinear)
        self.up5 = Up(256, 128 // factor, bilinear)  # 新增的上采样层
        self.up6 = Up(128, 64, bilinear)  # 新增的上采样层
        self.outc = OutConv(64, n_classes)

        input_height = 256  # 假设输入图像高度为 256
        input_width = 256   # 假设输入图像宽度为 256
        downsample_levels = 6  # 总共有 6 层下采样

        final_height = input_height // (2 ** downsample_levels)
        final_width = input_width // (2 ** downsample_levels)
        channels = 4096 // factor  # 最深层的通道数

        self.bottleneck_dim =  16384 # channels * final_height * final_width
        self.domain_adaptation_branch = nn.Sequential(
            nn.Linear(self.bottleneck_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1)
        )

        # # 定义领域适配分支
        # bottleneck_dim = (4096 // factor) * 8 * 8  # 瓶颈层输出形状为 [batch_size, 512, 8, 8]
        # self.domain_adaptation_branch = nn.Sequential(
        #     nn.Linear(bottleneck_dim, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 1)
        # )

    def forward(self, x, domain_label=None, alpha=1.0):
        # 前向传播UNet主干网络
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        
        # 解码路径
        x = self.up1(x7, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        logits = self.outc(x)
        return logits

    def get_features_and_output(self, x):
        """提取中间特征图并返回分割输出"""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        features = self.down6(x6)  # 中间特征图

        x = self.up1(features, x6)
        x = self.up2(x, x5)
        x = self.up3(x, x4)
        x = self.up4(x, x3)
        x = self.up5(x, x2)
        x = self.up6(x, x1)
        segmentation_output = self.outc(x)
        return features, segmentation_output

    def domain_adaptation_loss(self, features, domain_label, alpha):
        """计算领域适配损失"""
        reversed_features = GradReverse.apply(features, alpha)  # 反转梯度
        domain_output = self.domain_adaptation_branch(reversed_features.view(features.size(0), -1))
        domain_loss = F.binary_cross_entropy_with_logits(domain_output, domain_label)
        return domain_loss


import torch.nn as nn
import torch.autograd as autograd

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None