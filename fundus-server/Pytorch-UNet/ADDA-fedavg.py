import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_score import dice_loss
from global_center import GlobalCenter

# 定义MMD损失函数
def mmd_loss(source_features, target_features, kernel_type='rbf', gamma=1.0):
    """计算源域特征和目标域特征之间的MMD损失
    :param source_features: 源域特征
    :param target_features: 目标域特征
    :param kernel_type: 核函数类型，默认为RBF核
    :param gamma: RBF核的参数
    :return: MMD损失
    """
    n_source = source_features.size(0)
    n_target = target_features.size(0)
    total = n_source + n_target
    # 拼接源域和目标域特征
    features = torch.cat((source_features, target_features), dim=0)
    # 计算核矩阵
    if kernel_type == 'rbf':
        XX = torch.cdist(features, features) ** 2
        kernel_matrix = torch.exp(-gamma * XX)
    else:
        raise NotImplementedError(f"Kernel type {kernel_type} is not implemented.")
    # 计算MMD
    K_ss = kernel_matrix[:n_source, :n_source]
    K_st = kernel_matrix[:n_source, n_source:]
    K_tt = kernel_matrix[n_source:, n_source:]
    mmd = (torch.sum(K_ss) / (n_source * n_source) + torch.sum(K_tt) / (n_target * n_target) 
           - 2 * torch.sum(K_st) / (n_source * n_target))
    return mmd

# 定义生成器模型
class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 1024, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(1024, output_channels, kernel_size=3, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 定义鉴别器模型
class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        features = self.conv(x)
        return torch.sigmoid(self.fc(features.view(-1)))


def train_model(
        source_datasets,
        target_dataset,
        model,
        device,
        epochs,  
        batch_size,
        learning_rate,
        global_epoch,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        client_id: int = 0,
        global_center: GlobalCenter = None
):
    # 合并所有源域数据集
    all_source_data = torch.utils.data.ConcatDataset(source_datasets)
    
    n_val = int(len(all_source_data) * 0.1 )
    n_train = len(all_source_data) - n_val
    train_set, val_set = random_split(all_source_data, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # 创建当前“站点”的数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    target_loader = DataLoader(target_dataset, shuffle=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(project='U-Net', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training for 
        Global_epoch:    {global_epoch + 1}
        Client_id:       {client_id}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')

    # 设置优化器、损失函数、学习率调度器等
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)

    # 添加生成器和鉴别器的优化器
    generator = Generator(input_channels=4096, output_channels=4096)
    generator = generator.to(device)
    generator_params = list(generator.parameters())
    generator_optimizer = optim.AdamW(generator_params, lr=learning_rate * 0.1)
    
    discriminator = Discriminator(4096)
    discriminator = discriminator.to(device)
    discriminator_params = list(discriminator.parameters())
    discriminator_optimizer = optim.AdamW(discriminator_params, lr=learning_rate * 0.1)

    # 设定混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # 定义任务损失函数和领域适应损失函数
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 定义任务损失函数和领域适应损失函数
    if model.n_classes == 1:
        criterion = nn.BCEWithLogitsLoss()
        dice_loss_fn = dice_loss
    else:
        criterion = nn.CrossEntropyLoss()
        dice_loss_fn = dice_loss

    # 开始当前“站点”的训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        target_iter = iter(target_loader)

        # 引入生成器和鉴别器的训练循环
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                try:
                    target_batch = next(target_iter)
                except StopIteration:
                    target_iter = iter(target_loader)
                    target_batch = next(target_iter)
                source_images, source_true_masks = batch['image'], batch['mask']
                target_images = target_batch['image']
                
                source_images = source_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                source_true_masks = source_true_masks.to(device=device, dtype=torch.long)
                target_images = target_images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
                # 生成目标域特征
                source_features, source_segmentation_output = model.get_features_and_output(source_images)
                with torch.no_grad():
                    generated_features = generator(source_features)
                
                # 计算目标域特征
                target_features, _ = model.get_features_and_output(target_images)

                # 打印特征尺寸进行调试
                # print("Generated features shape:", generated_features.shape)
                # print("Target features shape:", target_features.shape)
                
                # 计算任务损失
                if model.n_classes == 1:
                    task_loss = criterion(source_segmentation_output.squeeze(1), source_true_masks.float())
                else:
                    task_loss = criterion(source_segmentation_output, source_true_masks)
                
                # 计算领域适应损失（MMD损失）
                domain_loss = mmd_loss(source_features, target_features)
                
                # 生成器损失
                generator_loss = F.mse_loss(generated_features, target_features)
                
                # 总损失
                loss = task_loss + 0.001 * domain_loss + 0.001 * generator_loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                scaler.step(optimizer)
                scaler.update()

                pbar.update(source_images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (4 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # val_score = evaluate(model, val_loader, device, amp)
                        val_results = evaluate(model, val_loader, device, amp)
                        
                        # 打印所有评价指标
                        print(f"Validation Dice: {val_results['dice']:.4f}")
                        print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
                        print(f"Validation Precision: {val_results['precision']:.4f}")
                        print(f"Validation Recall: {val_results['recall']:.4f}")
                        print(f"Validation F1 Score: {val_results['f1_score']:.4f}")
                        print(f"Validation Specificity: {val_results['specificity']:.4f}")
                        print(f"Validation IoU: {val_results['iou']:.4f}")


                        val_score = val_results['dice']  # 使用 Dice 得分做下降
                        scheduler.step(val_score)

                        logging.info(f'Validation Dice score for sub-site {client_id}: {val_score}')
                        try:
                            masks_pred = source_segmentation_output
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(source_images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(source_true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

    # 训练完当前“站点”指定轮次后，将本地模型参数传递给全局中心
    if global_center:
        global_center.receive_client_params(client_id, model.state_dict(), n_train)


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a.pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')  

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 初始化模型
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    model.to(device=device)

    # 定义站点数量
    num_clients = 3  # 三组联邦学习
    global_epoch = 3
    args.epochs = 3

    # 初始化全局中心
    global_center = GlobalCenter(num_clients)

    # 加载每个站点的数据集
    dir_img_list = [
        Path('/home/tangzhiri/yanhanhu/dataset/dataset/G1020-img'),
        Path('/home/tangzhiri/yanhanhu/dataset/dataset/ORIGA-img'),
        Path('/home/tangzhiri/yanhanhu/dataset/dataset/REFUGE-img'),
        Path('/home/tangzhiri/yanhanhu/dataset/dataset/PAPILA-img')
    ]
    dir_mask_list = [
        Path('/home/tangzhiri/yanhanhu/dataset/dataset/G1020_disc_mask'),
        Path('/home/tangzhiri/yanhanhu/dataset/dataset/ORIGA_disc_mask'),
        Path('/home/tangzhiri/yanhanhu/dataset/dataset/REFUGE_disc_mask'),
        Path('/home/tangzhiri/yanhanhu/dataset/dataset/PAPILA_disc_mask')
    ]
    dir_checkpoint = Path('./checkpoints/')

    sub_datasets = []
    for img_dir, mask_dir in zip(dir_img_list, dir_mask_list):
        try:
            dataset = CarvanaDataset(img_dir, mask_dir, args.scale)
        except (AssertionError, RuntimeError, IndexError):
            dataset = BasicDataset(img_dir, mask_dir, args.scale)
        sub_datasets.append(dataset)

    target_dataset = sub_datasets[3]  # 第四个数据集作为目标域

    # 联邦学习训练循环
    for p in range(global_epoch):
        for n in range(num_clients):
            source_datasets = [sub_datasets[n]]  # 当前源域数据集
            # 训练当前站点
            train_model(
                source_datasets=source_datasets,
                target_dataset=target_dataset,
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent=0.1,
                amp=args.amp,
                client_id=n,
                global_center=global_center,
                global_epoch=p
            )

        # 全局中心聚合参数
        global_center.aggregate_params()

        # 分发聚合后的参数到每个站点
        aggregated_params = global_center.distribute_params()
        model.load_state_dict(aggregated_params)

        # 初始化全局中心的参数存储
        global_center.init_params()

        # 保存模型
        if True:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = sub_datasets[0].mask_values  # 使用第一个站点的 mask_values
            torch.save(state_dict, str(dir_checkpoint / f'ADDA_PAPILA_disc.pth'))
            logging.info(f'Checkpoint saved for global epoch {p + 1}!')