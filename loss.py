import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiClassDiceFocalLoss(nn.Module):
    """
    Combined Loss Architecture for Severe Class Imbalance (Background, Road, Fallback).
    Addresses the edge-case focus requirement in Problem Statement.
    """
    def __init__(self, alpha=0.5, gamma=2.0, ce_weight=0.5, dice_weight=0.5, num_classes=3):
        super(MultiClassDiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.num_classes = num_classes
        
        # We can use CrossEntropyLoss which automatically applies LogSoftmax
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets, smooth=1):
        # 1. CROSS ENTROPY & FOCAL LOSS
        # inputs: [B, C, H, W] logits
        # targets: [B, H, W] long integers
        
        CE_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * CE_loss
        focal_loss = focal_loss.mean()
        
        # 2. MULTI-CLASS DICE LOSS
        # Convert targets to one-hot: [B, C, H, W]
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        # Convert inputs to probabilities
        probs = F.softmax(inputs, dim=1)
        
        # Calculate dice per class and take the mean
        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        
        dice_score = (2. * intersection + smooth) / (union + smooth)
        dice_loss = 1 - dice_score.mean() # Average across all classes
        
        # Weighted Combination
        combined_loss = self.ce_weight * focal_loss + self.dice_weight * dice_loss
        return combined_loss

def calculate_miou(preds, labels, num_classes=3):
    """
    Calculate Mean Intersection over Union for multi-class inputs.
    preds: [B, H, W] (already argmaxed)
    labels: [B, H, W]
    """
    ious = []
    
    # Ignore background or include it? Problem statement usually implies mIoU over all classes.
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union == 0:
            ious.append(float('nan'))  # If there is no ground truth, do not include in mIoU
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            
    # Filter out nans
    valid_ious = [iou for iou in ious if not torch.isnan(torch.tensor(iou))]
    if not valid_ious:
        return 0.0
    return sum(valid_ious) / len(valid_ious)
