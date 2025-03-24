# local_site 功能：
# 1、训练自编码器
# 2、用自己训练好的编码器提取特征
# 3、用central_server 训练的分类器进行分类
import torch
import torch.optim as optim
from models import Autoencoder, Classifier, CNNFeatureExtractor, CNNFeatureExtractor_256_good, MobileNet_model, MobileNet_model_v3
import torch.nn as nn
import numpy as np

class LocalSite_use_custom_dataset:
    def __init__(self, site_id, train_loader, test_loader, input_dim, feature_dim, learning_rate=0.001, local_epochs=30, lambda_domain=0.1):
        """初始化本地站点"""
        self.site_id = site_id  # 站点 ID
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader  # 本地站点的训练数据加载器
        self.test_loader = test_loader

        self.model = MobileNet_model_v3().to(self.device)
        # self.model = MobileNet_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
        # self.criterion = nn.BCEWithLogitsLoss()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

        self.local_epochs = local_epochs  # 训练轮数
        self.train_loss_history = []
        # 新增
        self.lambda_domain = lambda_domain  # 领域损失权重
        self.domain_loss = 0.0       # 当前领域损失

    def train_local_model(self, verbose=True):
        """本地特征提取器训练流程"""
        self.model.train()
        for epoch in range(self.local_epochs):
            for data, labels in self.train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # 前向传播
                logits, _ = self.model(data)
                # print(f"Logits shape: {logits.shape}, Labels shape: {labels.shape}")  # 检查
                train_loss = self.criterion(logits, labels)

                # 计算领域对抗损失
                domain_loss = self.domain_loss * self.lambda_domain

                # 总损失
                total_loss = train_loss + domain_loss
                
                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            if (epoch+1) % 5 == 0:
                print(f"[Site {self.site_id}] AE Epoch {epoch+1}/{self.local_epochs} | "
                      f"train_loss: {train_loss:.4f}, domain_loss:{domain_loss:.4f}")

    def update_domain_loss(self, new_loss):
        """接收服务器返回的领域损失"""
        self.domain_loss = new_loss.detach()  # 阻断梯度回传

    def get_id(self):  # 用于在通信阶段获取site的id
        return self.site_id

    def extract_features(self, data_loader=None, include_labels=True):
        """
        提取本地数据的特征并与标签一起返回
        返回:
            features (list): 特征列表。
            labels (list): 标签列表。
        """
        if data_loader is None:
            data_loader = self.train_loader

        features_list = []
        label_list = []
        self.model.eval()
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                _, features = self.model(data)
                features_list.append(features.cpu())
                label_list.append(labels.cpu())
        
        return torch.cat(features_list), torch.cat(label_list)

    def evaluate_local_model(self, test_loader=None, verbose=True):
        """
        本地模型评估方法
        参数:
            test_loader: 测试数据加载器 (默认使用本地测试集)
        返回:
            accuracy: 分类准确率
        """
        if test_loader is None:
            test_loader = self.test_loader
        
        self.model.eval()  # 确保模型处于评估模式
        total = 0
        correct = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                # 数据转移到设备
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # 分类预测
                logits, _ = self.model(data)
                _, predicted = torch.max(logits.data, 1)
                
                # 统计结果
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy

    def evaluate_classifier(self, classifier, test_loader=None, verbose=True):
        """
        分类器评估方法
        参数:
            classifier: 中心服务器下发的分类器
            test_loader: 测试数据加载器 (默认使用本地测试集)
        返回:
            accuracy: 分类准确率
        """
        if test_loader is None:
            test_loader = self.test_loader
        
        classifier.eval().to(self.device)
        self.model.eval()  # 确保特征提取器处于评估模式
        total = 0
        correct = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                # 数据转移到设备
                data = data.to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                _, features = self.model(data)  # 直接使用模型前向传播
                
                # 分类预测
                outputs = classifier(features)
                _, predicted = torch.max(outputs.data, 1)
                
                # 统计结果
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy


class LocalSite:
    def __init__(self, site_id, train_loader, test_loader, input_dim, feature_dim, learning_rate=0.001, local_epochs=30, lambda_domain=0.1):
        """初始化本地站点"""
        self.site_id = site_id  # 站点 ID
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_loader = train_loader  # 本地站点的训练数据加载器
        self.test_loader = test_loader

        self.model = CNNFeatureExtractor(feature_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()  # 分类任务使用交叉熵损失
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.local_epochs = local_epochs  # 训练轮数
        self.train_loss_history = []
        # 新增
        self.lambda_domain = lambda_domain  # 领域损失权重
        self.domain_loss = 0.0       # 当前领域损失

    def train_local_model(self, verbose=True):
        """本地特征提取器训练流程"""
        self.model.train()
        for epoch in range(self.local_epochs):
            for data, labels in self.train_loader:
                data = data.to(self.device)
                labels = labels.to(self.device)

                # 前向传播
                logits, _ = self.model(data)
                train_loss = self.criterion(logits, labels)

                # 计算领域对抗损失
                domain_loss = self.domain_loss * self.lambda_domain

                # 总损失
                total_loss = train_loss + domain_loss

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

            if (epoch+1) % 5 == 0:
                print(f"[Site {self.site_id}] AE Epoch {epoch+1}/{self.local_epochs} | "
                      f"train_loss: {train_loss:.4f}, domain_loss:{domain_loss:.4f}")

    def update_domain_loss(self, new_loss):
        """接收服务器返回的领域损失"""
        self.domain_loss = new_loss.detach()  # 阻断梯度回传

    def get_id(self):  # 用于在通信阶段获取site的id
        return self.site_id

    def extract_features(self, data_loader=None, include_labels=True):
        """
        提取本地数据的特征并与标签一起返回
        返回:
            features (list): 特征列表。
            labels (list): 标签列表。
        """
        if data_loader is None:
            data_loader = self.train_loader

        features_list = []
        label_list = []
        self.model.eval()
        with torch.no_grad():
            for data, labels in data_loader:
                data = data.to(self.device)
                _, features = self.model(data)
                features_list.append(features.cpu())
                label_list.append(labels.cpu())
        
        return torch.cat(features_list), torch.cat(label_list)

    def evaluate_classifier(self, classifier, test_loader=None, verbose=True):
        """
        分类器评估方法
        参数:
            classifier: 中心服务器下发的分类器
            test_loader: 测试数据加载器 (默认使用本地测试集)
        返回:
            accuracy: 分类准确率
        """
        if test_loader is None:
            test_loader = self.test_loader
        
        classifier.eval().to(self.device)
        total = 0
        correct = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                # 数据转移到设备
                data = data.view(-1, 784).to(self.device)
                labels = labels.to(self.device)
                
                # 提取特征
                features = self.model.get_features(data)
                
                # 分类预测
                outputs = classifier(features)
                _, predicted = torch.max(outputs.data, 1)
                
                # 统计结果
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy