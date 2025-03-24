import torch
from data_loader import load_custom_dataset, load_OIA_ODIR_dataset
from local_site import LocalSite_use_custom_dataset
from central_server import CentralServer_use_custom_dataset
from utils import save_model, load_model
from models import Classifier, MobileNet_model, MobileNet_model_v3

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 参数配置
    config = {
        "n_sites": 3,               # 站点数量
        "input_dim": 224 * 224,     # 自定义数据集图片维度 224x224
        "feature_dim": 1264,        # 编码维度(mobilenet输出的维度)
        "batch_size": 64,           # 数据加载批次大小
        "local_epochs": 15,         # 本地站点训练轮数
        "central_epochs": 20,       # 中心分类器训练轮数
        "learning_rate": 0.01,      # 学习率 0.001或者0.01  隔壁fedavg是0.01
        "model_save_path": "./central_classifier.pth",  # 模型保存路径
        "communication_epochs": 10,  # 通信轮次  20应该就够了
        "domain_lambda": 0.01,       # 领域损失权重 0.1 或者 0.01, 0.001
        "train_root": "/home/tangzhiri/yanhanhu/framework/data/OIA-ODIR/preprocessed_v2/training_set",  # 训练集根目录路径
        "test_root": "/home/tangzhiri/yanhanhu/framework/data/OIA-ODIR/preprocessed_v2/on_site_test_set"     # 测试集根目录路径
    }

    # train_image_root = "/home/tangzhiri/yanhanhu/framework/data/OIA-ODIR/preprocessed/training_set"
    # test_image_root = "/home/tangzhiri/yanhanhu/framework/data/OIA-ODIR/preprocessed/on_site_test_set"
    # train_gt_filepath = "/home/tangzhiri/yanhanhu/framework/data/Training Set/Annotation/training annotation (English).xlsx"
    # test_gt_filepath = "/home/tangzhiri/yanhanhu/framework/data/On-site Test Set/Annotation/on-site test annotation (English).xlsx"
    # n_sites = 5
    # class_names = 8

    # train_loaders, val_loaders, test_loaders = load_OIA_ODIR_dataset(
    #     train_image_root, test_image_root, train_gt_filepath, test_gt_filepath, n_sites
    # )

    # 1. 准备数据
    print("\n=== 阶段1: 数据准备 ===")
    train_loaders, val_loaders, test_loaders, class_names = load_custom_dataset(
        train_root=config["train_root"],
        test_root=config["test_root"],
        n_sites=config["n_sites"],
        image_size=224,
        batch_size=config["batch_size"],
        val_ratio=0.3
    )
    print(f"数据准备完成，类别数量: {len(class_names)}")

    # 2. 初始化本地站点
    print("\n=== 阶段2: 初始化本地站点 ===")
    sites = []
    for site_id in range(config["n_sites"]):
        site = LocalSite_use_custom_dataset(
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
    central_server = CentralServer_use_custom_dataset(
        input_dim=config["feature_dim"],
        output_dim=len(class_names),  # 使用自定义数据集的类别数量
        n_sites=config["n_sites"]
    )
    print(f"中心服务器初始化完成")

    # 4. 训练自编码器
    print("\n=== 阶段4: 本地自编码器训练 ===")
    for _ in range(config["communication_epochs"]):  # 通信轮次
        print(f"\n通信轮次 : {_}轮")
        central_server.initialize_features_list()  # 每次通信前需要初始化中心存储的特征列表
        
        # all_features = []
        # all_labels = []

        for site in sites:  # 每轮通信涉及到全部的sites
            site.train_local_model(verbose=True)
            features, labels = site.extract_features()  # 本地站点将特征上传
            central_server.receive_features(site.get_id(), features)  # 中心服务器接收特征及id
            
            # 特征聚合
            # all_features.append(features)
            # all_labels.append(labels)
        
        central_server.train_central_domain_discriminator()  # 训练中心域判别器
        # train_loader = central_server.aggregate_features(all_features, all_labels)  # 中心分类器训练
        # central_server.train_central_classifier_model(
        #     train_loader=train_loader,
        #     epochs=config["central_epochs"]
        # )
        # save_model(central_server.model, config["model_save_path"]) # 模型保存
        # # 加载模型
        # central_classifier = Classifier(config["feature_dim"], len(class_names)).to(device)
        # load_model(central_classifier, config["model_save_path"], map_location=device)

        for site in sites:
            # accuracy = site.evaluate_classifier(
            #     classifier=central_classifier,
            #     test_loader=site.test_loader
            # )
            # print(f"站点 {site.site_id} 测试准确率: {accuracy:.2f}%")

            site_loss = central_server.calculate_site_domain_loss(site.get_id())  # 计算指定站点的域判别器损失
            site.update_domain_loss(site_loss)  # 本地站点接收域判别器损失

    # 5. 特征提取与聚合
    # print("\n=== 阶段5: 特征聚合 ===")
    # all_features = []
    # all_labels = []
    # for site in sites:
    #     features, labels = site.extract_features()
    #     all_features.append(features)
    #     all_labels.append(labels)

    # 6. 中心服务器训练
    # print("\n=== 阶段6: 中心分类器训练 ===")
    # train_loader = central_server.aggregate_features(all_features, all_labels)
    # central_server.train_central_classifier_model(
    #     train_loader=train_loader,
    #     epochs=config["central_epochs"]
    # )

    # 7. 保存/加载模型
    # print("\n=== 阶段7: 模型保存 ===")
    # save_model(central_server.model, config["model_save_path"])
    # print(f"中心模型已保存至 {config['model_save_path']}")

    # 8. 本地站点评估
    # print("\n=== 阶段8: 本地评估 ===")
    # # 加载模型
    # central_classifier = Classifier(config["feature_dim"], len(class_names)).to(device)
    # load_model(central_classifier, config["model_save_path"], map_location=device)

    # 各站点评估 and 全局评估
    avg_acc = 0
    for site in sites:
        # accuracy = site.evaluate_classifier(
        accuracy = site.evaluate_local_model(
            # classifier=central_classifier,
            test_loader=site.test_loader
        )
        avg_acc += accuracy
        print(f"站点 {site.site_id} 测试准确率: {accuracy:.2f}%")

    print(f"Global Average Accuracy: {avg_acc/len(sites):.2f}%")


if __name__ == "__main__":
    main()