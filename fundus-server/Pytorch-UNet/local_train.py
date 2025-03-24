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

#=====================================dataset===================================
# preprocess data
dir_img = Path('/home/tangzhiri/yanhanhu/model/CUT/results/test2_11/train_latest/images/fake_B')
dir_mask = Path('/home/tangzhiri/yanhanhu/dataset/test1/trainA_cup_mask')
#=====================================dataset===================================
dir_checkpoint = Path('./checkpoints/')

def train_model(
        datasets,
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
        client_id: int = 0,  # 添加本地站点编号参数
        global_center: GlobalCenter = None  # 添加全局中心实例参数
):

    n_val = int(len(datasets) * 0.1 )
    n_train = len(datasets) - n_val
    train_set, val_set = random_split(datasets, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # # 循环模拟每个“站点”进行训练、参数传递、更新等操作
    # for sub_dataset_index in range(len(sub_datasets)):
    #     # 2. 划分当前“站点”的训练/验证分区
    #     n_val = int(len(sub_datasets[sub_dataset_index]) * val_percent)
    #     n_train = len(sub_datasets[sub_dataset_index]) - n_val
    #     train_set, val_set = random_split(sub_datasets[sub_dataset_index], [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. 创建当前“站点”的数据加载器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

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

    # 4. 设置优化器、损失函数、学习率调度器等
    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    #====================================
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    #====================================
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 5. 开始当前“站点”的训练
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'global_epoch: {global_epoch + 1} | index: {client_id} | local_epoch: {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type!= 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
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
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        # val_score = evaluate(model, val_loader, device, amp)
                        val_results = evaluate(model, val_loader, device, amp)
                        
                        # 打印所有评价指标（可选）
                        print(f"Validation Dice: {val_results['dice']:.4f}")
                        print(f"Validation Accuracy: {val_results['accuracy']:.4f}")
                        print(f"Validation Precision: {val_results['precision']:.4f}")
                        print(f"Validation Recall: {val_results['recall']:.4f}")
                        print(f"Validation F1 Score: {val_results['f1_score']:.4f}")
                        print(f"Validation Specificity: {val_results['specificity']:.4f}")
                        print(f"Validation IoU: {val_results['iou']:.4f}")


                        val_score = val_results['dice']  # 使用 Dice 得分
                        scheduler.step(val_score)

                        logging.info(f'Validation Dice score for sub-site {client_id}: {val_score}')
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

    # 训练完当前“站点”指定轮次后，将本地模型参数传递给全局中心
    # if global_center:
        # global_center.receive_client_params(client_id, model.state_dict())

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
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')   # 青光眼分割是三类！！！

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    
    model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                 f'\t{model.n_channels} input channels\n'
                 f'\t{model.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)
        logging.info(f'Model loaded from {args.load}')

    model.to(device=device)

    #===============================修改========================================
    num_clients = 1    # 本地站点数
    global_epoch = 1   # 全局 epoch
    args.epochs = 10    # 本地 epoch
    #===========================================================================

    global_center = GlobalCenter(num_clients)

    # 1. 创建数据集（划分成多份模拟不同站点数据）
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, args.scale)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, args.scale)

    # 将数据集均分成 num_clients 份
    sub_datasets = random_split(dataset, [len(dataset) // num_clients] * num_clients, generator=torch.Generator().manual_seed(0))

    for p in range(global_epoch):
        for n in range(num_clients):
            # 传入本地站点编号和全局中心实例到训练函数
            train_model(
                datasets=sub_datasets[n],
                model=model,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
                device=device,
                img_scale=args.scale,
                val_percent= 0.1,
                amp=args.amp,
                client_id=n,
                global_center=global_center,
                global_epoch = p
            )
            # 将本地模型参数传递给全局中心
            global_center.receive_client_params(n, model.state_dict())
        
        # 所有“站点”都训练完一轮后，全局中心进行参数聚合
        global_center.aggregate_params()

        # 获取聚合后的参数并更新本地模型
        aggregated_params = global_center.distribute_params()
        
        # 初始化------这里总是感觉逻辑有点问题，待修正
        global_center.init_params()
        model.load_state_dict(aggregated_params)

        # 保存节点
        if True:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, str(dir_checkpoint / f'test2_disc.pth'))
            logging.info(f'Checkpoint saved!')