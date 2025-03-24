import torch
from torch import nn, optim
from models import FedAvgClassifier, MobileNet_model

class FedAvgLocal_256:
    def __init__(self, site_id, train_loader, test_loader,
                 lr=1e-3, device="cuda"):
        """
        FedAvg本地站点
        :param site_id: 站点标识
        :param train_loader: 训练数据加载器
        :param test_loader: 测试数据加载器
        """
        self.site_id = site_id
        self.device = torch.device(device)
        
        # 初始化本地模型（与全局模型结构一致）
        self.model = MobileNet_model().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = train_loader
        self.test_loader = test_loader

    def local_train(self, global_params=None, epochs=1):
        """
        本地训练过程
        :param global_params: 全局模型参数（用于初始化）
        :param epochs: 本地训练轮数
        :return: 更新后的模型参数
        """
        # 加载全局参数
        if global_params is not None:
            self.model.load_state_dict(global_params)
            
        self.model.train()
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        
        # 返回更新后的参数
        return self.model.state_dict()

    def evaluate(self, model_params=None):
        """
        评估当前模型性能
        :param model_params: 要评估的模型参数（使用本地模型）
        """
        model = MobileNet_model().to(self.device)
        if model_params is not None:
            model.load_state_dict(model_params)
        model.eval()
        
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output,features = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = 100. * correct / len(self.test_loader.dataset)
        print(f"Site {self.site_id} | Test Accuracy: {acc:.2f}%")
        return acc

class FedAvgLocal:
    def __init__(self, site_id, train_loader, test_loader, 
                 lr=1e-3, device="cuda"):
        """
        FedAvg本地站点
        :param site_id: 站点标识
        :param train_loader: 训练数据加载器
        :param test_loader: 测试数据加载器
        """
        self.site_id = site_id
        self.device = torch.device(device)
        
        # 初始化本地模型（与全局模型结构一致）
        self.model = FedAvgClassifier().to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_loader = train_loader
        self.test_loader = test_loader

    def local_train(self, global_params=None, epochs=1):
        """
        本地训练过程
        :param global_params: 全局模型参数（用于初始化）
        :param epochs: 本地训练轮数
        :return: 更新后的模型参数
        """
        # 加载全局参数
        if global_params is not None:
            self.model.load_state_dict(global_params)
            
        self.model.train()
        for _ in range(epochs):
            for data, target in self.train_loader:
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
        
        # 返回更新后的参数
        return self.model.state_dict()

    def evaluate(self, model_params=None):
        """
        评估当前模型性能
        :param model_params: 要评估的模型参数（默认使用本地模型）
        """
        model = FedAvgClassifier().to(self.device)
        if model_params is not None:
            model.load_state_dict(model_params)
        model.eval()
        
        correct = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
        
        acc = 100. * correct / len(self.test_loader.dataset)
        print(f"Site {self.site_id} | Test Accuracy: {acc:.2f}%")
        return acc