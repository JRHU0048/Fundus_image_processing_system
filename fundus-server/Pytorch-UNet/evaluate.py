
    
    
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchmetrics import Accuracy, Precision, Recall, F1Score, Specificity, JaccardIndex
from utils.dice_score import multiclass_dice_coeff, dice_coeff

@torch.inference_mode()
def evaluate(net, dataloader, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    
    # Initialize metrics
    dice_metric = 0
    acc_metric = Accuracy(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes).to(device)
    prec_metric = Precision(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes, average='macro').to(device)
    recall_metric = Recall(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes, average='macro').to(device)
    f1_metric = F1Score(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes, average='macro').to(device)
    spec_metric = Specificity(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes, average='macro').to(device)
    iou_metric = JaccardIndex(task='binary' if net.n_classes == 1 else 'multiclass', num_classes=net.n_classes).to(device)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_metric += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
                
                # Update metrics for binary classification
                mask_true_float = mask_true.float()
                acc_metric.update(mask_pred, mask_true_float)
                prec_metric.update(mask_pred, mask_true_float)
                recall_metric.update(mask_pred, mask_true_float)
                f1_metric.update(mask_pred, mask_true_float)
                spec_metric.update(mask_pred, mask_true_float)
                iou_metric.update(mask_pred, mask_true_float)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes)'
                # convert to one-hot format
                mask_true_one_hot = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred_one_hot = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_metric += multiclass_dice_coeff(mask_pred_one_hot[:, 1:], mask_true_one_hot[:, 1:], reduce_batch_first=False)
                
                # Update metrics for multi-class case
                acc_metric.update(mask_pred.argmax(dim=1), mask_true)
                prec_metric.update(mask_pred.argmax(dim=1), mask_true)
                recall_metric.update(mask_pred.argmax(dim=1), mask_true)
                f1_metric.update(mask_pred.argmax(dim=1), mask_true)
                spec_metric.update(mask_pred.argmax(dim=1), mask_true)
                iou_metric.update(mask_pred.argmax(dim=1), mask_true)

    net.train()

    # Calculate the final scores
    avg_dice = dice_metric / max(num_val_batches, 1)
    avg_acc = acc_metric.compute()
    avg_prec = prec_metric.compute()
    avg_recall = recall_metric.compute()
    avg_f1 = f1_metric.compute()
    avg_spec = spec_metric.compute()
    avg_iou = iou_metric.compute()

    # Reset metrics for the next evaluation
    acc_metric.reset()
    prec_metric.reset()
    recall_metric.reset()
    f1_metric.reset()
    spec_metric.reset()
    iou_metric.reset()

    return {
        'dice': avg_dice,
        'accuracy': avg_acc,
        'precision': avg_prec,
        'recall': avg_recall,
        'f1_score': avg_f1,
        'specificity': avg_spec,
        'iou': avg_iou
    }