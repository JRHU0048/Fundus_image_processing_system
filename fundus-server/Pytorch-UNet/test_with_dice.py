import argparse
import logging
import os
import torch
import csv  # 导入 CSV 模块
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

def test_model(dataset, model, device, amp):
    loader_args = dict(batch_size=1, num_workers=os.cpu_count(), pin_memory=True)
    test_loader = DataLoader(dataset, shuffle=False, **loader_args)

    # 存储每个样本的指标
    sample_metrics = []

    for batch in tqdm(test_loader, desc="Testing", unit="batch"):
        images, true_masks = batch['image'], batch['mask']
        images = images.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=torch.long)

        # 使用模型生成预测结果
        model.eval()  # 确保模型处于评估模式
        with torch.no_grad():
            preds = model(images)

        # 调用 evaluate 函数计算指标
        results = evaluate(preds, true_masks, device, amp)

        # 假设数据集中有 'name' 字段用于存储文件名
        sample_name = batch.get('name', ['unknown'])[0]  # 如果没有 'name'，使用 'unknown'

        # 记录当前样本的指标
        sample_metrics.append({
            'filename': sample_name,
            'dice': results['dice'],
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'specificity': results['specificity'],
            'iou': results['iou']
        })

    # 返回所有样本的指标
    return sample_metrics
    
if __name__ == '__main__':
    args = get_test_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    ########################################################################
    args.test_dir_img = "/home/tangzhiri/yanhanhu/dataset/dataset/trainB"
    args.test_dir_mask = "/home/tangzhiri/yanhanhu/dataset/dataset/PAPILA_disc_mask"
    args.pretrained_model_path = './checkpoints/ours_disc.pth'
    ########################################################################

    # 加载预训练模型
    model = UNet(n_channels=3, n_classes=args.classes)
    state_dict = torch.load(args.pretrained_model_path, map_location=device)

    if 'mask_values' in state_dict:
        del state_dict['mask_values']
    model.load_state_dict(state_dict)
    model.to(device=device)

    # 准备测试数据集
    try:
        test_dataset = CarvanaDataset(Path(args.test_dir_img), Path(args.test_dir_mask), args.scale)
    except (AssertionError, RuntimeError, IndexError):
        test_dataset = BasicDataset(Path(args.test_dir_img), Path(args.test_dir_mask), args.scale)

    # 执行测试并获取每个样本的指标
    sample_metrics = test_model(test_dataset, model, device, args.amp)

    # 将指标写入 CSV 文件
    csv_file = 'sample_metrics.csv'
    fieldnames = ['filename', 'dice', 'accuracy', 'precision', 'recall', 'f1_score', 'specificity', 'iou']

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_metrics)

    print(f"Sample metrics have been saved to {csv_file}")