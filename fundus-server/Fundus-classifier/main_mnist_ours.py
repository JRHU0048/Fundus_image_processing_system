# ======================
# 使用mnist数据集构建ours框架，本地站点进行特征提取（比fedavg中的完整分类器少最后一层线性映射），中心服务器进行基于特征的线性分类
# ======================
import torch
from data_loader import load_mnist_data
from local_site import LocalSite
from central_server import CentralServer
from utils import save_model, load_model
from models import Classifier

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 参数配置
    config = {
        "n_sites": 5,               # 站点数量
        "input_dim": 784,           # MNIST图片维度 28x28=784
        "feature_dim": 128,           # 编码维度
        "batch_size": 64,            # 数据加载批次大小
        "local_epochs": 10,             # 本地站点训练轮数
        "central_epochs": 50,        # 中心分类器训练轮数
        "learning_rate": 1e-3,       # 学习率
        "model_save_path": "./central_classifier.pth",  # 模型保存路径
        "communication_epochs": 5,     # 通信轮次
        "domain_lambda": 0.1,        # 领域损失权重
        # "total_comm_rounds": 10      # 总通信轮次
    }

    # 1. 准备数据
    print("\n=== 阶段1: 数据准备 ===")
    train_loaders, test_loaders = load_mnist_data(
        n_sites=config["n_sites"],
        batch_size=config["batch_size"]
    )
    print(f"数据准备完成")

    # 2. 初始化本地站点
    print("\n=== 阶段2: 初始化本地站点 ===")
    sites = []
    for site_id in range(config["n_sites"]):
        site = LocalSite(
            site_id=site_id,
            train_loader=train_loaders[site_id],
            test_loader=test_loaders[site_id],
            input_dim=config["input_dim"],
            feature_dim=config["feature_dim"],
            learning_rate=config["learning_rate"],
            local_epochs=config["local_epochs"],
            lambda_domain=config["domain_lambda"]
        )
        sites.append(site)
        print(f"站点 {site_id} 初始化完成")

    # 3. 初始化中心服务器
    print("\n=== 阶段3: 初始化中心服务器 ===")
    central_server = CentralServer(
        input_dim=config["feature_dim"],
        output_dim=10,   # MNIST有10个类别
        n_sites=config["n_sites"]
    )
    print(f"中心服务器初始化完成")

    # 4. 训练自编码器
    print("\n=== 阶段4: 本地自编码器训练 ===")
    for _ in range(config["communication_epochs"]):  # 通信轮次
        central_server.initialize_features_list()  # 每次通信前需要初始化中心存储的特征列表
        for site in sites:  # 每轮通信涉及到全部的sites
            site.train_local_model(verbose=True)
            features, _ = site.extract_features()  # 本地站点将特征上传
            central_server.receive_features(site.get_id(), features)  #中心服务器接收特征及id
        central_server.train_central_domain_discriminator()  # 训练中心服务器的域判别器
        for site in sites:
            site_loss = central_server.calculate_site_domain_loss(site.get_id())  # 计算指定站点的域判别器损失
            site.update_domain_loss(site_loss)  # 本地站点接收域判别器损失

    # 5. 特征提取与聚合
    print("\n=== 阶段5: 特征聚合 ===")
    all_features = []
    all_labels = []
    for site in sites:
        features, labels = site.extract_features()
        all_features.append(features)
        all_labels.append(labels)

    # 6. 中心服务器训练
    print("\n=== 阶段6: 中心分类器训练 ===")
    train_loader = central_server.aggregate_features(all_features, all_labels)
    central_server.train_central_classifier_model(
        train_loader=train_loader,
        epochs=config["central_epochs"]
    )

    # 7. 保存/加载模型
    print("\n=== 阶段7: 模型保存 ===")
    save_model(central_server.model, config["model_save_path"])
    print(f"中心模型已保存至 {config['model_save_path']}")

    # 8. 本地站点评估
    print("\n=== 阶段8: 本地评估 ===")
    # 加载模型
    central_classifier = Classifier(config["feature_dim"], 10).to(device)
    load_model(central_classifier, config["model_save_path"], map_location=device)

    # 各站点评估
    for site in sites:
        accuracy = site.evaluate_classifier(
            classifier=central_classifier,
            test_loader=site.test_loader
        )
        print(f"站点 {site.site_id} 测试准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    main()