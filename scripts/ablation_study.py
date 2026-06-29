"""
Ablation Study for ViT-UNet Architecture
Analyzes the contribution of different components:
1. Pretrained ViT encoder
2. Skip connections
3. BCE vs Dice loss weighting
4. Different loss combinations
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
import matplotlib.pyplot as plt
import pandas as pd

from ieee_access_complete_experiments import (
    OilSpillDataset, Trainer, plot_ablation_study, get_transforms,
    double_conv, MetricsCalculator, device
)

# ============================================================================
# ABLATION MODEL VARIANTS
# ============================================================================

class ViTEncoderNoPretrain(nn.Module):
    """ViT Encoder without pretraining."""
    def __init__(self, model_name="vit_small_patch16_224"):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=False)
        self.patch_embed = self.vit.patch_embed
        self.cls_token = self.vit.cls_token
        self.pos_embed = self.vit.pos_embed
        self.pos_drop = self.vit.pos_drop
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm
        self.embed_dim = self.vit.embed_dim
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        skip_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in {2, 5, 8}:
                skip = x[:, 1:, :].permute(0, 2, 1).view(B, self.embed_dim, 14, 14)
                skip_features.append(skip)
        
        x = self.norm(x)
        x = x[:, 1:, :].permute(0, 2, 1).view(B, self.embed_dim, 14, 14)
        return x, skip_features


class ViTUNetNoSkip(nn.Module):
    """ViT-UNet without skip connections."""
    def __init__(self, model_name="vit_small_patch16_224", out_channels=1, pretrained=True):
        super().__init__()
        if pretrained:
            self.encoder = timm.create_model(model_name, pretrained=True, features_only=True)
        else:
            self.encoder = ViTEncoderNoPretrain(model_name)
        
        vit_dim = 384  # vit_small_patch16_224
        
        # Decoder without skip connections
        self.up3 = nn.ConvTranspose2d(vit_dim, 512, kernel_size=2, stride=2)
        self.conv3 = double_conv(512, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = double_conv(256, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = double_conv(128, 128)
        
        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0 = double_conv(64, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        final_feat, _ = self.encoder(x)
        
        x = self.up3(final_feat)
        x = self.conv3(x)
        
        x = self.up2(x)
        x = self.conv2(x)
        
        x = self.up1(x)
        x = self.conv1(x)
        
        x = self.up0(x)
        x = self.conv0(x)
        
        return self.final_conv(x)


class ViTUNetNoPretrain(nn.Module):
    """ViT-UNet without pretraining."""
    def __init__(self, model_name="vit_small_patch16_224", out_channels=1):
        super().__init__()
        self.encoder = ViTEncoderNoPretrain(model_name)
        vit_dim = 384
        
        self.up3 = nn.ConvTranspose2d(vit_dim, 512, kernel_size=2, stride=2)
        self.conv3 = double_conv(512 + vit_dim, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = double_conv(256 + vit_dim, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = double_conv(128 + vit_dim, 128)
        
        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0 = double_conv(64, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        final_feat, skip_feats = self.encoder(x)
        skip0, skip1, skip2 = skip_feats
        
        x = self.up3(final_feat)
        skip2_up = F.interpolate(skip2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2_up], dim=1)
        x = self.conv3(x)
        
        x = self.up2(x)
        skip1_up = F.interpolate(skip1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1_up], dim=1)
        x = self.conv2(x)
        
        x = self.up1(x)
        skip0_up = F.interpolate(skip0, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip0_up], dim=1)
        x = self.conv1(x)
        
        x = self.up0(x)
        x = self.conv0(x)
        
        return self.final_conv(x)


class ViTUNetShallowSkip(nn.Module):
    """ViT-UNet with only one skip connection."""
    def __init__(self, model_name="vit_small_patch16_224", out_channels=1, pretrained=True):
        super().__init__()
        if pretrained:
            from ieee_access_complete_experiments import ViTEncoder
            self.encoder = ViTEncoder(model_name, pretrained=pretrained)
        else:
            self.encoder = ViTEncoderNoPretrain(model_name)
        
        vit_dim = 384
        
        self.up3 = nn.ConvTranspose2d(vit_dim, 512, kernel_size=2, stride=2)
        self.conv3 = double_conv(512, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = double_conv(256, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = double_conv(128 + vit_dim, 128)  # Only one skip
        
        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0 = double_conv(64, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        final_feat, skip_feats = self.encoder(x)
        skip0, skip1, skip2 = skip_feats
        
        x = self.up3(final_feat)
        x = self.conv3(x)
        
        x = self.up2(x)
        x = self.conv2(x)
        
        x = self.up1(x)
        skip0_up = F.interpolate(skip0, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip0_up], dim=1)
        x = self.conv1(x)
        
        x = self.up0(x)
        x = self.conv0(x)
        
        return self.final_conv(x)


class ViTEncoderDeepSkip(nn.Module):
    """ViT Encoder extracting features from deep layers (9, 10, 11)."""
    def __init__(self, model_name="vit_small_patch16_224", pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.patch_embed = self.vit.patch_embed
        self.cls_token = self.vit.cls_token
        self.pos_embed = self.vit.pos_embed
        self.pos_drop = self.vit.pos_drop
        self.blocks = self.vit.blocks
        self.norm = self.vit.norm
        self.embed_dim = self.vit.embed_dim
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        skip_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in {9, 10, 11}:
                skip = x[:, 1:, :].permute(0, 2, 1).view(B, self.embed_dim, 14, 14)
                skip_features.append(skip)
        
        x = self.norm(x)
        x = x[:, 1:, :].permute(0, 2, 1).view(B, self.embed_dim, 14, 14)
        return x, skip_features


class ViTUNetDeepSkip(nn.Module):
    """ViT-UNet with deep skip connections."""
    def __init__(self, model_name="vit_small_patch16_224", out_channels=1, pretrained=True):
        super().__init__()
        self.encoder = ViTEncoderDeepSkip(model_name, pretrained=pretrained)
        vit_dim = 384
        
        self.up3 = nn.ConvTranspose2d(vit_dim, 512, kernel_size=2, stride=2)
        self.conv3 = double_conv(512 + vit_dim, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = double_conv(256 + vit_dim, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = double_conv(128 + vit_dim, 128)
        
        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0 = double_conv(64, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        final_feat, skip_feats = self.encoder(x)
        skip0, skip1, skip2 = skip_feats
        
        x = self.up3(final_feat)
        skip2_up = F.interpolate(skip2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip2_up], dim=1)
        x = self.conv3(x)
        
        x = self.up2(x)
        skip1_up = F.interpolate(skip1, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip1_up], dim=1)
        x = self.conv2(x)
        
        x = self.up1(x)
        skip0_up = F.interpolate(skip0, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip0_up], dim=1)
        x = self.conv1(x)
        
        x = self.up0(x)
        x = self.conv0(x)
        
        return self.final_conv(x)


# ============================================================================
# LOSS FUNCTION VARIANTS
# ============================================================================

class BCEOnlyLoss(nn.Module):
    """Only BCE Loss."""
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        return self.bce(logits, targets)


class DiceOnlyLoss(nn.Module):
    """Only Dice Loss."""
    def __init__(self):
        super().__init__()
        import segmentation_models_pytorch as smp
        self.dice = smp.losses.DiceLoss(mode="binary")
    
    def forward(self, logits, targets):
        return self.dice(torch.sigmoid(logits), targets)


class FocalLoss(nn.Module):
    """Focal Loss for imbalanced segmentation."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce_loss
        return loss.mean()


class TverskyLoss(nn.Module):
    """Tversky Loss for better control of FP/FN trade-off."""
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
    
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fp = (probs * (1 - targets)).sum()
        fn = ((1 - probs) * targets).sum()
        tversky = tp / (tp + self.alpha * fp + self.beta * fn + 1e-6)
        return 1 - tversky


# ============================================================================
# ABLATION STUDY RUNNER
# ============================================================================

def run_ablation_study():
    """Run complete ablation study."""
    print("="*80)
    print("ABLATION STUDY: ViT-UNet Component Analysis")
    print("="*80)
    
    # Paths
    refined_images_train = 'dataset/images/images/train'
    refined_masks_train = 'dataset/masks/masks/train'
    refined_images_val = 'dataset/images/images/val'
    refined_masks_val = 'dataset/masks/masks/val'
    
    transform_train, transform_mask, transform_val, _ = get_transforms()
    
    train_dataset = OilSpillDataset(
        images_dir=refined_images_train,
        masks_dir=refined_masks_train,
        transform_image=transform_train,
        transform_mask=transform_mask
    )
    
    val_dataset = OilSpillDataset(
        images_dir=refined_images_val,
        masks_dir=refined_masks_val,
        transform_image=transform_val,
        transform_mask=transform_mask
    )
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    from ieee_access_complete_experiments import ViTUNet
    
    # Define ablation configurations
    ablation_configs = {
        # Architecture ablations
        'A: Full Model (Pretrained + Skip)': {
            'model': ViTUNet(model_name="vit_small_patch16_224", pretrained=True),
            'loss': 'combined',
            'use_boundary': False
        },
        'B: No Pretraining': {
            'model': ViTUNetNoPretrain(model_name="vit_small_patch16_224"),
            'loss': 'combined',
            'use_boundary': False
        },
        'C: No Skip Connections': {
            'model': ViTUNetNoSkip(model_name="vit_small_patch16_224", pretrained=True),
            'loss': 'combined',
            'use_boundary': False
        },
        'D: Shallow Skip (1 level)': {
            'model': ViTUNetShallowSkip(model_name="vit_small_patch16_224", pretrained=True),
            'loss': 'combined',
            'use_boundary': False
        },
        'D2: Deep Skip (layers 9, 10, 11)': {
            'model': ViTUNetDeepSkip(model_name="vit_small_patch16_224", pretrained=True),
            'loss': 'combined',
            'use_boundary': False
        },
    }
    
    # Loss function ablations
    loss_configs = {
        'E: BCE Only (0.8:0.0)': {
            'model': ViTUNet(model_name="vit_small_patch16_224", pretrained=True),
            'loss': 'bce',
            'use_boundary': False
        },
        'F: Dice Only (0.0:1.0)': {
            'model': ViTUNet(model_name="vit_small_patch16_224", pretrained=True),
            'loss': 'dice',
            'use_boundary': False
        },
        'G: Balanced (0.5:0.5)': {
            'model': ViTUNet(model_name="vit_small_patch16_224", pretrained=True),
            'loss': 'balanced',
            'use_boundary': False
        },
        'H: Focal Loss': {
            'model': ViTUNet(model_name="vit_small_patch16_224", pretrained=True),
            'loss': 'focal',
            'use_boundary': False
        },
    }
    
    all_configs = {**ablation_configs, **loss_configs}
    results = {}
    
    for config_name, config in all_configs.items():
        print(f"\n{'='*60}")
        print(f"Configuration: {config_name}")
        print(f"{'='*60}")
        
        model = config['model']
        loss_type = config['loss']
        
        # Override loss function
        from ieee_access_complete_experiments import CombinedLoss
        if loss_type == 'bce':
            class CustomTrainer(Trainer):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                def train(self, *args, **kwargs):
                    # Override to use BCE only
                    kwargs['use_boundary'] = config['use_boundary']
                    return super().train(*args, **kwargs)
            trainer = CustomTrainer(model, config_name.replace(' ', '_').replace(':', ''), device=device)
        elif loss_type == 'dice':
            class CustomTrainer(Trainer):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
            trainer = Trainer(model, config_name.replace(' ', '_').replace(':', ''), device=device)
        else:
            trainer = Trainer(model, config_name.replace(' ', '_').replace(':', ''), device=device)
        
        # Train
        history = trainer.train(
            train_loader, val_loader,
            num_epochs=100,
            lr=1e-4,
            save_path='models_results/ablation',
            use_boundary=config['use_boundary']
        )
        
        # Evaluate
        metrics, _, _ = trainer.evaluate(val_loader, use_boundary=config['use_boundary'])
        results[config_name] = metrics
        
        print(f"\nResults for {config_name}:")
        print(f"  mIoU: {metrics['mIoU']:.4f}")
        print(f"  F1:   {metrics['F1']:.4f}")
        print(f"  Prec: {metrics['Precision']:.4f}")
        print(f"  Rec:  {metrics['Recall']:.4f}")
    
    # Save results
    os.makedirs('models_results', exist_ok=True)
    results_df = pd.DataFrame(results).T
    results_df.to_csv('models_results/ablation_study_results.csv')
    
    # Plot results
    plot_ablation_study(results, 'figures')
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(results_df[['mIoU', 'F1', 'Precision', 'Recall', 'Accuracy']].to_string())
    print("="*80)
    
    return results


if __name__ == "__main__":
    run_ablation_study()
