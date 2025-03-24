# models 包含了：
# 1、自编码器结构定义
# 2、分类器结构定义
# 3、FedavgClassifier结构定义
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import mobilenet_v2
from torchvision.models import MobileNet_V2_Weights
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import torch.nn.utils.spectral_norm as spectral_norm

class MobileNet_model_v3(nn.Module):
    def __init__(self, num_classes=8, feature_dim=128, img_size=224, selected_layers=[3, 6, 9, 13]):
        """
        基于MobileNet的分类网络
        :param num_classes: 分类类别数（默认2类）
        :param feature_dim: 特征维度（用于特征提取）
        :param img_size: 输入图像尺寸（默认224）
        :param selected_layers: 要提取特征的层索引列表
        """
        super(MobileNet_model_v3, self).__init__()

        # 加载预训练MobileNetV3作为基础网络
        self.base_model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)

        # 自定义分类头，MobileNetV3 Large最后输出维度是960,输入维度是1184
        self.custom_head = nn.Sequential(
            nn.Linear(960, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

        self.selected_layers = selected_layers

    def forward(self, x):
        """
        前向传播过程
        :return: (logits, features) 
        """
        all_features = []
        for i, layer in enumerate(self.base_model.features):
            x = layer(x)
            if i in self.selected_layers:
                # 对当前层的特征图进行全局平均池化
                layer_features = nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
                all_features.append(layer_features)

        # 从mobilenet最后一层提取特征
        final_features = nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)
        all_features.append(final_features)

        # 拼接所有提取的特征
        combined_features = torch.cat(all_features, dim=1)

        # 分类预测
        logits = self.custom_head(final_features)

        return logits, combined_features

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



class CNNFeatureExtractor_256_good(nn.Module):
    def __init__(self, feature_dim, num_classes=2):
        super().__init__()
        # 卷积层
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # 128x128
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 64x64
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 32x32
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        )
        # 分类器
        self.classifier = nn.Linear(256, num_classes)
        # 特征投影
        self.feature_proj = nn.Linear(256, feature_dim)

    def forward(self, x):
        x = self.features(x)        # [batch, 256, 1, 1]
        x = torch.flatten(x, 1)     # [batch, 256]
        logits = self.classifier(x) # [batch, num_classes]
        features = self.feature_proj(x)  # [batch, feature_dim]
        return logits, features


class FedAvgClassifier_with_custom_dataset(nn.Module):
    def __init__(self, feature_dim=128, num_classes=2):
        super().__init__()
        # 512输入
        self.features = nn.Sequential(
            # 512x512 -> 256x256
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 256x256 -> 128x128
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 128x128 -> 64x64
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            # 64x64 -> 32x32（新增一层）
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        # 调整全连接层输入维度
        self.classifier = nn.Linear(512, num_classes)  # 输入维度改为512
        self.feature_proj = nn.Linear(512, feature_dim)

    def forward(self, x):
        # 特征提取流程
        x = self.features(x)        # [batch, 256, 1, 1]
        x = torch.flatten(x, 1)     # [batch, 256]
        
        # 同时返回分类logits和特征向量
        logits = self.classifier(x)        # [batch, num_classes]
        features = self.feature_proj(x)    # [batch, feature_dim]
        return logits, features


class CNNFeatureExtractor(nn.Module):
    """CNN特征提取器（倒数第二层输出特征）"""
    def __init__(self, feature_dim=128, num_classes=10):
        super().__init__()
        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 ->14x14
            nn.ReLU(),
            nn.MaxPool2d(2)   # 7x7
        )
        
        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),  # 倒数第二层特征输出
            nn.ReLU()
        )
        
        # 分类头（最后一层）
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # 输入形状调整：原始输入是展平的784维向量
        x = x.view(-1, 1, 28, 28)  # [batch, 1, 28, 28]
        
        # 特征提取
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(conv_out.size(0), -1)  # 展平
        features = self.fc_layers(conv_out)
        
        # 分类输出
        logits = self.classifier(features)
        return logits, features

    def get_features(self, x):
        """单独获取特征的方法"""
        with torch.no_grad():
            _, features = self.forward(x)
        return features

# class Classifier(nn.Module):
#     """分类器"""
#     def __init__(self, input_dim, num_classes=2):
#         super().__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(input_dim, num_classes)
#         )

#     def forward(self, x):
#         return self.layers(x)

class Classifier(nn.Module):
    """分类器"""
    def __init__(self, input_dim, num_classes=8):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class FedAvgClassifier(nn.Module):
    """用于FedAvg的全局模型结构（与CNNFeatureExtractor结构一致）"""
    def __init__(self, feature_dim=128, num_classes=10):
        super().__init__()
        # 卷积层
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14 ->14x14
            nn.ReLU(),
            nn.MaxPool2d(2)  # 7x7
        )

        # 全连接层
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim),  # 倒数第二层特征输出
            nn.ReLU()
        )

        # 分类头（最后一层）
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):
        # 输入形状调整：原始输入是展平的784维向量
        x = x.view(-1, 1, 28, 28)  # [batch, 1, 28, 28]

        # 特征提取
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(conv_out.size(0), -1)  # 展平
        features = self.fc_layers(conv_out)

        # 分类输出
        logits = self.classifier(features)
        return logits
    
class DomainDiscriminator_v1(nn.Module):
    """简单映射"""
    def __init__(self, input_dim, num_domains):
        """"
        input_dim：输入特征的维度
        num_domains：参与训练的本地站点的个数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, num_domains),
        )
    def forward(self, x):
        return self.net(x)

class DomainDiscriminator_v2(nn.Module):
    """多层感知机"""
    def __init__(self, input_dim, num_domains):
        """
        input_dim：输入特征的维度
        num_domains：参与训练的本地站点的个数
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_domains)
        )
    def forward(self, x):
        return self.net(x)

class DomainDiscriminator_v3(nn.Module):
    """谱归一化（Spectral Normalization）"""
    def __init__(self, input_dim, num_domains):
        """
        input_dim：输入特征的维度
        num_domains：参与训练的本地站点的个数
        """
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, 128)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(128, 64)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Linear(64, num_domains))
        )
    def forward(self, x):
        return self.net(x)

class DomainDiscriminator_v4(nn.Module):
    """注意力机制"""
    def __init__(self, input_dim, num_domains):
        """
        input_dim：输入特征的维度
        num_domains：参与训练的本地站点的个数
        """
        super().__init__()
        self.attention = AttentionBlock(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_domains)
        )

    def forward(self, x):
        x = self.attention(x)
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim // 16, input_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class Autoencoder(nn.Module):
    """自编码器"""
    def __init__(self, input_dim, feature_dim):
        super(Autoencoder, self).__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x):
        x = x.view(-1, 784)
        encoded = self.encoder(x)  # 压缩特征
        decoded = self.decoder(encoded)  # 重建特征
        return decoded

    def encode(self, x):
        return self.encoder(x)  # 提取隐空间特征