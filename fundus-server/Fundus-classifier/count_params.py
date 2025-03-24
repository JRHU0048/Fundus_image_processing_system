import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class MobileNet_model(nn.Module):
    def __init__(self, num_classes=2, feature_dim=128, img_size=512):
        """
        基于MobileNet的分类网络
        :param num_classes: 分类类别数（默认2类）
        :param feature_dim: 特征维度（用于特征提取）
        :param img_size: 输入图像尺寸（默认512）
        """
        super(MobileNet_model, self).__init__()

        # 加载预训练MobileNetV2作为基础网络
        self.base_model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)

        # 修改分类器结构 -----------------------------------------------------------------
        # 解冻所有基础网络参数
        # for param in self.base_model.parameters():
        #     param.requires_grad = True

        # 自定义分类头
        self.custom_head = nn.Sequential(
            nn.Linear(1280, 512),  # MobileNetV2最后输出维度是1280
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

        # 特征投影层
        # self.feature_proj = nn.Linear(1280, feature_dim)

    def forward(self, x):
        """
        前向传播过程
        :return: (logits, features) 
        """
        # 从mobilenet中进行特征提取
        features = self.base_model.features(x)
        features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        # 特征投影
        # proj_features = self.feature_proj(features)
        # 分类预测
        logits = self.custom_head(features)

        # return logits, proj_features
        return logits, features


# 初始化模型
model = MobileNet_model()

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())

print(f"模型的总参数数量: {total_params}")