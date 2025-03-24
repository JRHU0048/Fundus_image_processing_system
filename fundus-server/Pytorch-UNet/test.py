import argparse
import logging
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from unet import UNet
from utils.data_loading import BasicDataset, CarvanaDataset
from evaluate import evaluate

# 定义获取测试参数的函数
def get_test_args():
    parser = argparse.ArgumentParser(description='Test the UNet on images and target masks using a pretrained model')
    parser.add_argument('--pretrained_model_path', '-p', type=str, required=False, help='Path to the pretrained model')
    parser.add_argument('--test_dir_img', type=str, required=False, help='Path to the test image directory')
    parser.add_argument('--test_dir_mask', type=str, required=False, help='Path to the test mask directory')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    return parser.parse_args()

# 定义测试模型的函数
def test_model(dataset, model, device,amp):
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)
    # test_score = evaluate(model, test_loader, device,amp)
    # logging.info(f'Test Dice score: {test_score}')

    test_results = evaluate(model, test_loader, device, amp)

    # 打印所有评价指标（可选）
    print(f"Validation Dice: {test_results['dice']:.4f}")
    print(f"Validation Accuracy: {test_results['accuracy']:.4f}")
    print(f"Validation Precision: {test_results['precision']:.4f}")
    print(f"Validation Recall: {test_results['recall']:.4f}")
    print(f"Validation F1 Score: {test_results['f1_score']:.4f}")
    print(f"Validation Specificity: {test_results['specificity']:.4f}")
    print(f"Validation IoU: {test_results['iou']:.4f}")
    
    return test_results

if __name__ == '__main__':
    args = get_test_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    ########################################################################
    args.test_dir_img = "/home/tangzhiri/yanhanhu/dataset/dataset/trainB"
    args.test_dir_mask = "/home/tangzhiri/yanhanhu/dataset/dataset/PAPILA_disc_mask"
    args.pretrained_model_path = './checkpoints/no_nce_disc.pth'
    ########################################################################

    # 加载预训练模型
    model = UNet(n_channels=3, n_classes=args.classes)
    state_dict = torch.load(args.pretrained_model_path, map_location=device)

    if'mask_values' in state_dict:
        del state_dict['mask_values']
    model.load_state_dict(state_dict)
    model.to(device=device)

    # 准备测试数据集
    try:
        test_dataset = CarvanaDataset(Path(args.test_dir_img), Path(args.test_dir_mask), args.scale)
    except (AssertionError, RuntimeError, IndexError):
        test_dataset = BasicDataset(Path(args.test_dir_img), Path(args.test_dir_mask), args.scale)

    # 执行测试
    test_results = test_model(test_dataset, model, device,args.amp)
    with open('test_result.txt', 'w') as f:
        f.write(f"Final Test Dice score: {test_results['dice']:.4f}")
        f.write(f"Final Test accuracy: {test_results['accuracy']:.4f}")
        f.write(f"Validation Precision: {test_results['precision']:.4f}")
        f.write(f"Validation Recall: {test_results['recall']:.4f}")
        f.write(f"Validation F1 Score: {test_results['f1_score']:.4f}")
        f.write(f"Validation Specificity: {test_results['specificity']:.4f}")
        f.write(f"Validation IoU: {test_results['iou']:.4f}")
        

# import argparse
# import logging
# import os
# import random
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
# from pathlib import Path
# from torch import optim
# from torch.utils.data import DataLoader, random_split
# from tqdm import tqdm

# import wandb
# from evaluate import evaluate
# from unet import UNet
# from utils.data_loading import BasicDataset, CarvanaDataset
# from utils.dice_score import dice_loss
# from global_center import GlobalCenter

# dir_checkpoint = Path('./checkpoints/')

# def get_args():
#     parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
#     parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1, help='Number of epochs')
#     parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
#     parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
#                         help='Learning rate', dest='lr')
#     parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a.pth file')
#     parser.add_argument('--pretrained_model_path', '-p', type=str, default='./checkpoints/checkpoint_epoch_1.pth', help='Path to the pretrained model for testing')  # 新增参数
#     parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
#     parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
#                         help='Percent of the data that is used as validation (0-100)')
#     parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
#     parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
#     parser.add_argument('--classes', '-c', type=int, default=3, help='Number of classes')

#     return parser.parse_args()

# def test_model(
#         dataset,
#         model,
#         device,
#         amp: bool = False
# ):
#     """
#     用于测试给定模型在指定数据集上的性能。

#     Args:
#         dataset (torch.utils.data.Dataset): 测试数据集。
#         model (nn.Module): 已训练好的模型。
#         device (torch.device): 运行设备（如 'cuda' 或 'cpu'）。
#         amp (bool, optional): 是否启用混合精度，默认为 False。
#     """
#     loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
#     test_loader = DataLoader(dataset, shuffle=False, **loader_args)

#     test_score = evaluate(model, test_loader, device, amp)
#     logging.info(f'Test Dice score: {test_score}')
#     return test_score

# if __name__ == '__main__':
#     args = get_args()

#     logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f'Using device {device}')

#     # 直接加载预训练模型进行测试
#     model = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
#     model = model.to(memory_format=torch.channels_last)
#     state_dict = torch.load(args.pretrained_model_path, map_location=device, weights_only=True)
#     del state_dict['mask_values']
#     model.load_state_dict(state_dict)
#     logging.info(f'Model loaded from {args.pretrained_model_path}')

#     model.to(device=device)

#     # 准备测试数据集
#     # test_dir_img = Path('/root/autodl-tmp/Harvard_seg/Testing/image/')
#     # test_dir_mask = Path('/root/autodl-tmp/Harvard_seg/Testing/mask3CLASS/')

#     # test_dir_img = Path('/root/autodl-tmp/3dataset/test/image/')
#     # test_dir_mask = Path('/root/autodl-tmp/3dataset/test/mask3/')

#     test_dir_img = Path('/root/autodl-tmp/3dataset/image/')
#     test_dir_mask = Path('/root/autodl-tmp/3dataset/mask3/')

#     args.pretrained_model_path = './checkpoints/checkpoint_epoch_1.pth'
#     try:
#         test_dataset = CarvanaDataset(test_dir_img, test_dir_mask, args.scale)
#     except (AssertionError, RuntimeError, IndexError):
#         test_dataset = BasicDataset(test_dir_img, test_dir_mask, args.scale)

#     final_test_score = test_model(test_dataset, model, device, args.amp)

#     with open('test_result.txt', 'w') as f:
#         f.write(f"Final Test Dice score: {final_test_score}")