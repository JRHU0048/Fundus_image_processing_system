# ======================
# 使用mnist数据集构建fedavg框架，本地站点训练完整的分类器，中心服务器做模型参数的加权平均
# ======================
import torch
from data_loader import load_mnist_data
from fedavg.fedavg_local import FedAvgLocal
from fedavg.fedavg_server import FedAvgServer
from models import FedAvgClassifier

def main():
    # 实验配置
    config = {
        "n_sites": 5,           # 站点数量
        "global_rounds": 10,    # 全局通信轮次
        "local_epochs": 5,      # 本地训练轮数
        "batch_size": 64,
        "lr": 0.01,  # 0.01可能效果最好
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # 1. 准备数据
    print("\n=== 阶段1: 数据准备 ===")
    train_loaders, test_loaders = load_mnist_data(
        n_sites=config["n_sites"],
        batch_size=config["batch_size"]
    )

    # 2. 初始化本地站点
    print("\n=== 阶段2: 初始化本地站点 ===")
    sites = []
    for site_id in range(config["n_sites"]):
        site = FedAvgLocal(
            site_id=site_id,
            train_loader=train_loaders[site_id],
            test_loader=test_loaders[site_id],
            lr=config["lr"],
            device=config["device"]
        )
        sites.append(site)

    # 3. 初始化全局服务器
    print("\n=== 阶段3: 初始化全局服务器 ===")
    global_model = FedAvgClassifier().to(config["device"])
    server = FedAvgServer(global_model)

    # 4. 联邦训练循环
    print("\n=== 阶段4: 联邦训练循环 ===")
    for round in range(config["global_rounds"]):
        print(f"\n=== Global Round {round+1}/{config['global_rounds']} ===")
        
        # 选择参与客户端（全参与）
        selected_sites = sites
        
        # 客户端本地训练
        params_list = []
        data_nums = []
        for site in selected_sites:
            # 获取全局参数
            global_params = server.get_global_params()
            
            # 本地训练
            local_params = site.local_train(
                global_params=global_params,
                epochs=config["local_epochs"]
            )
            
            # 记录参数和数据量
            params_list.append(local_params)
            data_nums.append(len(site.train_loader.dataset))

        # 服务器聚合
        server.aggregate(params_list, data_nums)

        # 全局评估
        avg_acc = 0
        for site in sites:
            acc = site.evaluate(server.get_global_params())
            avg_acc += acc
        print(f"Global Average Accuracy: {avg_acc/len(sites):.2f}%")

if __name__ == "__main__":
    main()