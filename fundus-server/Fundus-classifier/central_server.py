# central_server 的功能：
# 1、聚合所有站点的特征数据
# 2、在聚合数据上训练分类器
# 3、返回训练好的模型
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from models import Classifier, DomainDiscriminator_v2
import torch.nn.functional as F 

class CentralServer_use_custom_dataset:
    def __init__(self, input_dim, output_dim, n_sites):
        """初始化中心服务器
        参数:
            input_dim: 特征维度 (与编码器输出维度一致)
            output_dim: 分类类别数
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Classifier(input_dim, output_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        # 新增域判别相关组件
        self.domain_discriminator = DomainDiscriminator_v2(input_dim, n_sites).to(self.device)
        self.domain_optimizer = optim.Adam(self.domain_discriminator.parameters())
        self.domain_criterion = nn.CrossEntropyLoss()
        self.n_sites = n_sites
        self.site_features = {i: [] for i in range(n_sites)}
        self.site_labels = {i: [] for i in range(n_sites)}
    
    def initialize_features_list(self): 
        # 存储各站点特征
        self.site_features = {i: [] for i in range(self.n_sites)}
        self.site_labels = {i: [] for i in range(self.n_sites)}
        
    def receive_features(self, site_id, features):
        """接收并处理站点特征"""
        # 存储特征用于后续域判别器的训练
        # self.site_features[site_id].append(features.detach())
        features = features.to(self.device).detach()
        self.site_features[site_id].append(features)
    
    def train_central_domain_discriminator(self):
        """训练域判别器"""
        self.domain_discriminator.train()
        
        # 准备数据
        all_features = []
        domain_labels = []
        for site_id, features in self.site_features.items():
            if len(features) > 0:

                batch_size = features[-1].shape[0]  # 获取最后一个batch的样本数
                labels = torch.full(
                    size=(batch_size,), 
                    fill_value=site_id,
                    dtype=torch.long,
                    device=self.device  # 确保设备一致
                )
                all_features.append(torch.cat(features).to(self.device))
                domain_labels.append(labels)

        # 添加空数据检查
        if len(all_features) == 0:
            return 0.0  # 无数据时返回零损失

        # 合并数据
        dataset = TensorDataset(
            torch.cat(all_features).to(self.device),
            torch.cat(domain_labels).to(self.device)
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        # 训练循环
        for _ in range(5):  # 5次
            for feat, label in loader:
                self.domain_optimizer.zero_grad()
                pred = self.domain_discriminator(feat.to(self.device))
                loss = self.domain_criterion(pred, label.to(self.device))
                loss.backward()
                self.domain_optimizer.step()
        
        return loss.item()

    def calculate_site_domain_loss(self, site_id):
        """计算指定站点的领域损失"""
        if len(self.site_features[site_id]) == 0:
            return torch.tensor(0.0, device=self.device)

        features = torch.cat([
            f.to(self.device) for f in self.site_features[site_id]
        ])

        # 生成真实域标签（全为当前站点ID）
        batch_size = features.size(0)
        true_labels = torch.full(
            size=(batch_size,), 
            fill_value=site_id,
            dtype=torch.long,
            device=self.device
        )

        # 计算域判别器预测
        pred = self.domain_discriminator(features)

        # 使用交叉熵损失（注意：这里要最大化判别器损失，所以取负号）
        loss = self.domain_criterion(pred, true_labels)  # 对抗训练需要最大化判别器损失

        return loss

    def aggregate_features(self, all_features, all_labels, batch_size=64):
        """聚合所有站点的特征"""
        features = torch.cat([f.to(self.device) for f in all_features])
        labels = torch.cat([l.to(self.device) for l in all_labels])
        # 创建数据集
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_central_classifier_model(self, train_loader, epochs=50):
        """训练中心分类器"""
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for features, labels in train_loader:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # 前向传播
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 统计信息
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # 打印训练进度
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {epoch_loss/len(train_loader):.4f} | "
                f"Accuracy: {100*correct/total:.2f}%")

        return self.model


class CentralServer:
    def __init__(self, input_dim, output_dim, n_sites):
        """初始化中心服务器
        参数:
            input_dim: 特征维度 (与编码器输出维度一致)
            output_dim: 分类类别数 (MNIST是10分类)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Classifier(input_dim, output_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters())
        # 新增域判别相关组件
        self.domain_discriminator = DomainDiscriminator(input_dim, n_sites).to(self.device)
        self.domain_optimizer = optim.Adam(self.domain_discriminator.parameters())
        self.domain_criterion = nn.CrossEntropyLoss()
        self.n_sites = n_sites
        self.site_features = {i: [] for i in range(n_sites)}
        self.site_labels = {i: [] for i in range(n_sites)}
    
    def initialize_features_list(self): 
        # 存储各站点特征
        self.site_features = {i: [] for i in range(self.n_sites)}
        self.site_labels = {i: [] for i in range(self.n_sites)}
        
    def receive_features(self, site_id, features):
        """接收并处理站点特征"""
        # 存储特征用于后续域判别器的训练
        # self.site_features[site_id].append(features.detach())
        features = features.to(self.device).detach()
        self.site_features[site_id].append(features)
    
    def train_central_domain_discriminator(self):
        """训练域判别器"""
        self.domain_discriminator.train()
        
        # 准备数据
        all_features = []
        domain_labels = []
        for site_id, features in self.site_features.items():
            if len(features) > 0:

                batch_size = features[-1].shape[0]  # 获取最后一个batch的样本数
                labels = torch.full(
                    size=(batch_size,), 
                    fill_value=site_id,
                    dtype=torch.long,
                    device=self.device  # 确保设备一致
                )
                all_features.append(torch.cat(features).to(self.device))
                domain_labels.append(labels)

        # 添加空数据检查
        if len(all_features) == 0:
            return 0.0  # 无数据时返回零损失

        # 合并数据
        dataset = TensorDataset(
            torch.cat(all_features).to(self.device),
            torch.cat(domain_labels).to(self.device)
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        # 训练循环
        for _ in range(5):  # 5次
            for feat, label in loader:
                self.domain_optimizer.zero_grad()
                pred = self.domain_discriminator(feat.to(self.device))
                loss = self.domain_criterion(pred, label.to(self.device))
                loss.backward()
                self.domain_optimizer.step()
        
        return loss.item()

    def calculate_site_domain_loss(self, site_id):
        """计算指定站点的领域损失"""
        if len(self.site_features[site_id]) == 0:
            return torch.tensor(0.0, device=self.device)

        features = torch.cat([
            f.to(self.device) for f in self.site_features[site_id]
        ])

        # 生成真实域标签（全为当前站点ID）
        batch_size = features.size(0)
        true_labels = torch.full(
            size=(batch_size,), 
            fill_value=site_id,
            dtype=torch.long,
            device=self.device
        )

        # 计算域判别器预测
        pred = self.domain_discriminator(features)

        # 使用交叉熵损失（注意：这里要最大化判别器损失，所以取负号）
        loss = self.domain_criterion(pred, true_labels)  # 对抗训练需要最大化判别器损失

        return loss

    def aggregate_features(self, all_features, all_labels, batch_size=64):
        """聚合所有站点的特征"""
        features = torch.cat([f.to(self.device) for f in all_features])
        labels = torch.cat([l.to(self.device) for l in all_labels])
        # 创建数据集
        dataset = TensorDataset(features, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def train_central_classifier_model(self, train_loader, epochs=50):
        """训练中心分类器"""
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for features, labels in train_loader:
                features = features.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # 前向传播
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 统计信息
                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            # 打印训练进度
            print(f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {epoch_loss/len(train_loader):.4f} | "
                f"Accuracy: {100*correct/total:.2f}%")

        return self.model
