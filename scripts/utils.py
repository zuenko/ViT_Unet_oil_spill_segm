"""
Utility functions and classes for training and evaluation
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MetricsCalculator:
    """
    Complete metrics calculator implementing equations (2-7) from the paper.
    
    Equations:
        (2) IoU = TP / (TP + FP + FN)
        (3) Precision = TP / (TP + FP)
        (4) Recall = TP / (TP + FN)
        (5) F1 = 2 * TP / (2 * TP + FP + FN)
        (6) Accuracy = (TP + TN) / (TP + TN + FP + FN)
        (7) mIoU = (IoU_class0 + IoU_class1) / 2
    """
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0
        self.total_tn = 0
        self.valid_pixels = 0
        self.per_image_ious = []
    
    def update(self, preds, targets):
        """
        Update metrics with a batch of predictions and targets.
        
        Args:
            preds: Predicted probabilities [B, 1, H, W]
            targets: Ground truth masks [B, 1, H, W]
        """
        preds_binary = (preds >= self.threshold).float()
        
        preds_flat = preds_binary.view(-1)
        targets_flat = targets.view(-1)
        
        tp = torch.sum(preds_flat * targets_flat).item()
        fp = torch.sum(preds_flat * (1 - targets_flat)).item()
        tn = torch.sum((1 - preds_flat) * (1 - targets_flat)).item()
        fn = torch.sum((1 - preds_flat) * targets_flat).item()
        
        if tp + fp + fn > 0:
            self.total_tp += tp
            self.total_fp += fp
            self.total_tn += tn
            self.total_fn += fn
            self.valid_pixels += len(preds_flat)
            
            iou = tp / (tp + fp + fn + 1e-6)
            self.per_image_ious.append(iou)
    
    def get_metrics(self):
        """Calculate all metrics using equations (2-7)."""
        eps = 1e-6
        tp, fp, tn, fn = self.total_tp, self.total_fp, self.total_tn, self.total_fn
        
        # Equation (2): IoU
        iou = tp / (tp + fp + fn + eps)
        
        # Equation (3): Precision
        precision = tp / (tp + fp + eps)
        
        # Equation (4): Recall
        recall = tp / (tp + fn + eps)
        
        # Equation (5): F1-Score
        f1 = 2 * precision * recall / (precision + recall + eps)
        
        # Equation (6): Accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        
        # Equation (7): Mean IoU
        iou_bg = tn / (tn + fp + fn + eps)
        miou = (iou + iou_bg) / 2
        
        specificity = tn / (tn + fp + eps)
        
        return {
            'IoU': iou,
            'mIoU': miou,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'Accuracy': accuracy,
            'Specificity': specificity,
            'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn,
            'IoU_std': np.std(self.per_image_ious) if self.per_image_ious else 0
        }


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice Loss.
    
    Args:
        bce_weight: Weight for BCE loss
        dice_weight: Weight for Dice loss
        boundary_weight: Weight for boundary loss (for CBD-Net)
    """
    def __init__(self, bce_weight=0.8, dice_weight=0.2, boundary_weight=0.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = smp.losses.DiceLoss(mode="binary")
    
    def forward(self, logits, targets, boundary_logits=None):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(torch.sigmoid(logits), targets)
        loss = self.bce_weight * bce + self.dice_weight * dice
        
        if boundary_logits is not None and self.boundary_weight > 0:
            boundary_targets = self._get_boundary(targets)
            boundary_loss = self.bce_loss(boundary_logits, boundary_targets)
            loss += self.boundary_weight * boundary_loss
        
        return loss
    
    def _get_boundary(self, mask):
        """Extract boundary from mask using morphological gradient."""
        kernel = torch.ones((1, 1, 3, 3), device=mask.device)
        dilated = F.conv2d(mask, kernel, padding=1)
        eroded = F.conv2d(mask, kernel, padding=1)
        boundary = (dilated > 0).float() * (eroded < 9).float()
        return boundary
