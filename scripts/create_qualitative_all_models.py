"""
Create qualitative figures with ALL 9 models
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.ndimage import center_of_mass, label, distance_transform_edt
import torch
from PIL import Image

def find_blob_centers(binary_mask, min_size=150, max_blobs=3):
    labeled_mask, num_features = label(binary_mask)
    if num_features == 0:
        return []
    
    blobs = []
    for i in range(1, num_features + 1):
        size = np.sum(labeled_mask == i)
        if size >= min_size:
            blobs.append((size, i))
            
    blobs.sort(reverse=True, key=lambda x: x[0])
    
    centers = []
    for size, lbl in blobs[:max_blobs]:
        blob_mask = (labeled_mask == lbl)
        dist = distance_transform_edt(blob_mask)
        y, x = np.unravel_index(np.argmax(dist), dist.shape)
        centers.append((int(x), int(y)))
    return centers

from models import (
    ViTUNet, UNetBaseline, DeepLabV3Plus, CBDNet,
    TransUNet, DSUNet, LRAUNet, DAENet, SegFormerWrapper
)
from dataset import OilSpillDataset, get_transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_all_models():
    # Load all 9 trained models.
    models = {}

    # ViT-UNet
    model = ViTUNet(pretrained=False)
    model.load_state_dict(torch.load('models_results/vit_unet_best.pth', map_location=device))
    models['ViT-UNet'] = (model.to(device).eval(), False)
    print("Loaded ViT-UNet")

    # TransUNet
    model = TransUNet(pretrained=False)
    model.load_state_dict(torch.load('models_results/transunet_best.pth', map_location=device))
    models['TransUNet'] = (model.to(device).eval(), False)
    print("Loaded TransUNet")

    # U-Net
    model = UNetBaseline()
    model.load_state_dict(torch.load('models_results/unet_best.pth', map_location=device))
    models['U-Net'] = (model.to(device).eval(), False)
    print("Loaded U-Net")

    # DeepLabV3+
    model = DeepLabV3Plus(pretrained=False)
    model.load_state_dict(torch.load('models_results/deeplabv3plus_best.pth', map_location=device))       
    models['DeepLabV3+'] = (model.to(device).eval(), False)
    print("Loaded DeepLabV3+")

    # CBD-Net
    model = CBDNet(pretrained=False)
    model.load_state_dict(torch.load('models_results/cbdnet_best.pth', map_location=device))
    models['CBD-Net'] = (model.to(device).eval(), True)
    print("Loaded CBD-Net")

    # DS-UNet
    model = DSUNet()
    model.load_state_dict(torch.load('models_results/dsunet_best.pth', map_location=device))
    models['DS-UNet'] = (model.to(device).eval(), True)
    print("Loaded DS-UNet")

    # LRA-UNet
    model = LRAUNet()
    model.load_state_dict(torch.load('models_results/lraunet_best.pth', map_location=device))
    models['LRA-UNet'] = (model.to(device).eval(), False)
    print("Loaded LRA-UNet")
    
    # DAENet
    model = DAENet(pretrained=False)
    model.load_state_dict(torch.load('models_results/daenet_best.pth', map_location=device))
    models['DAENet'] = (model.to(device).eval(), False)
    print("Loaded DAENet")

    # SegFormer
    model = SegFormerWrapper(pretrained=False)
    model.load_state_dict(torch.load('models_results/segformer_best.pth', map_location=device))
    models['SegFormer'] = (model.to(device).eval(), False)
    print("Loaded SegFormer")

    return models

def calculate_iou(pred, target):
    """Calculate IoU."""
    pred_binary = (pred > 0.5).astype(np.float32)
    target_binary = (target > 0.5).astype(np.float32)
    intersection = np.logical_and(pred_binary, target_binary).sum()
    union = np.logical_or(pred_binary, target_binary).sum()
    return intersection / (union + 1e-6)

def add_annotations(ax, model_name, tp, fp, fn):
    if fn.sum() > 150:
        centers = find_blob_centers(fn, min_size=150, max_blobs=2)
        for (cx, cy) in centers:
            text_x = max(10, cx - 60) if cx > 112 else min(214, cx + 60)
            text_y = max(10, cy - 60) if cy > 112 else min(214, cy + 60)
            text_x += np.random.randint(-15, 15)
            text_y += np.random.randint(-15, 15)
            ax.annotate('Missed', xy=(cx, cy), xytext=(text_x, text_y),
                        arrowprops=dict(facecolor='yellow', shrink=0.05, width=2, headwidth=6),
                        color='yellow', fontweight='bold', fontsize=8,
                        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', pad=1))
    if fp.sum() > 150:
        centers = find_blob_centers(fp, min_size=150, max_blobs=1)
        for (cx, cy) in centers:
            text_x = max(10, cx - 60) if cx > 112 else min(214, cx + 60)
            text_y = max(10, cy - 60) if cy > 112 else min(214, cy + 60)
            text_x += np.random.randint(-15, 15)
            text_y += np.random.randint(-15, 15)
            ax.annotate('False\nAlarm', xy=(cx, cy), xytext=(text_x, text_y),
                        arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=6),
                        color='red', fontweight='bold', fontsize=8,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

def create_figure4_all_models():
    """Create figure4 with all 9 models."""
    print("\nCreating Figure 4 with all 9 models...")
    
    models = load_all_models()
    model_list = list(models.items())
    
    val_img_dir = 'dataset/images/images/val'
    val_mask_dir = 'dataset/masks/masks/val'
    
    _, _, transform_val, _ = get_transforms()
    val_dataset = OilSpillDataset(val_img_dir, val_mask_dir, transform_val, transform_val)
    
    # Select samples
    sample_indices = [0, 50, 100, 200]
    
    # 11 columns: Input + GT + 9 models
    fig = plt.figure(figsize=(26, 12))
    gs = GridSpec(len(sample_indices), 11, figure=fig, hspace=0.25, wspace=0.1)
    
    for row, idx in enumerate(sample_indices):
        img_filename = val_dataset.image_files[idx]
        original_img = np.array(Image.open(os.path.join(val_img_dir, img_filename)).convert("RGB"))
        img_tensor, mask_tensor = val_dataset[idx]
        true_mask = mask_tensor.squeeze(0).cpu().numpy()
        
        # Column 0: Input
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(original_img, cmap='gray')
        if row == 0:
            ax.set_title('Input SAR', fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # Column 1: Ground Truth
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(original_img, cmap='gray', alpha=0.6)
        gt_colored = np.zeros((*true_mask.shape, 4))
        gt_colored[true_mask > 0.5] = [0, 1, 0, 0.5]
        ax.imshow(gt_colored)
        if row == 0:
            ax.set_title('Ground Truth', fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # Columns 2-10: Model predictions
        for col, (model_name, (model, use_boundary)) in enumerate(model_list, start=2):
            ax = fig.add_subplot(gs[row, col])
            
            with torch.no_grad():
                img_batch = img_tensor.unsqueeze(0).to(device)
                if use_boundary:
                    output, _ = model(img_batch)
                else:
                    output = model(img_batch)
                pred = torch.sigmoid(output)[0, 0].cpu().numpy()
            
            iou = calculate_iou(pred, true_mask)
            
            # Overlay
            ax.imshow(original_img, cmap='gray', alpha=0.6)
            pred_binary = (pred > 0.5).astype(np.float32)
            true_binary = (true_mask > 0.5).astype(np.float32)
            
            overlay = np.zeros((*pred_binary.shape, 4))
            tp = (pred_binary == 1) & (true_binary == 1)
            fp = (pred_binary == 1) & (true_binary == 0)
            fn = (pred_binary == 0) & (true_binary == 1)
            
            overlay[tp] = [0, 1, 0, 0.6]
            overlay[fp] = [1, 0, 0, 0.6]
            overlay[fn] = [0, 0, 1, 0.6]
            ax.imshow(overlay)
            
            add_annotations(ax, model_name, tp, fp, fn)
            
            if row == 0:
                ax.set_title(f'{model_name}\nIoU:{iou:.3f}', fontsize=8, fontweight='bold')
            else:
                ax.set_title(f'{iou:.3f}', fontsize=8)
            ax.axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='True Positive (Correct Detection)'),
        Patch(facecolor='red', alpha=0.6, label='False Positive (False Alarm)'),
        Patch(facecolor='blue', alpha=0.6, label='False Negative (Missed Spill)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle('Qualitative Comparison - All 9 Models (100 Epochs)', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('figures/figure4_real_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: figure4_real_predictions.png")

def create_best_worst_all_models():
    """Create best/worst cases with all 9 models."""
    print("\nCreating best/worst cases with all 9 models...")
    
    models = load_all_models()
    model_list = list(models.items())
    
    val_img_dir = 'dataset/images/images/val'
    val_mask_dir = 'dataset/masks/masks/val'
    
    _, _, transform_val, _ = get_transforms()
    val_dataset = OilSpillDataset(val_img_dir, val_mask_dir, transform_val, transform_val)
    
    # Find best/worst cases using ViT-UNet
    print("Finding best/worst cases...")
    sample_ious = []
    
    for idx in range(min(50, len(val_dataset))):
        img_tensor, mask_tensor = val_dataset[idx]
        true_mask = mask_tensor.squeeze(0).cpu().numpy()
        
        model, use_boundary = models['ViT-UNet']
        with torch.no_grad():
            img_batch = img_tensor.unsqueeze(0).to(device)
            output = model(img_batch)
            pred = torch.sigmoid(output)[0, 0].cpu().numpy()
        
        iou = calculate_iou(pred, true_mask)
        sample_ious.append((idx, iou))
    
    sample_ious.sort(key=lambda x: x[1])
    
    worst_idx = sample_ious[0][0]
    best_idx = sample_ious[-1][0]
    mid_idx = len(sample_ious) // 2
    medium_idx = sample_ious[mid_idx][0]
    
    samples = [
        (best_idx, f'Best Case (IoU={sample_ious[-1][1]:.3f})'),
        (medium_idx, f'Medium Case (IoU={sample_ious[mid_idx][1]:.3f})'),
        (worst_idx, f'Worst Case (IoU={sample_ious[0][1]:.3f})')
    ]
    
    # Create figure with all 9 models
    fig = plt.figure(figsize=(26, 10))
    gs = GridSpec(3, 11, figure=fig, hspace=0.3, wspace=0.1)
    
    for row, (idx, title) in enumerate(samples):
        img_filename = val_dataset.image_files[idx]
        original_img = np.array(Image.open(os.path.join(val_img_dir, img_filename)).convert("RGB"))
        img_tensor, mask_tensor = val_dataset[idx]
        true_mask = mask_tensor.squeeze(0).cpu().numpy()
        
        # Input
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(original_img, cmap='gray')
        if row == 0:
            ax.set_title('Input', fontsize=9, fontweight='bold')
        ax.set_ylabel(title, fontsize=10, fontweight='bold', rotation=90, labelpad=10)
        ax.axis('off')
        
        # Ground Truth
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(original_img, cmap='gray', alpha=0.6)
        gt_colored = np.zeros((*true_mask.shape, 4))
        gt_colored[true_mask > 0.5] = [0, 1, 0, 0.5]
        ax.imshow(gt_colored)
        if row == 0:
            ax.set_title('Ground Truth', fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # All 9 models
        for col, (model_name, (model, use_boundary)) in enumerate(model_list, start=2):
            ax = fig.add_subplot(gs[row, col])
            
            with torch.no_grad():
                img_batch = img_tensor.unsqueeze(0).to(device)
                if use_boundary:
                    output, _ = model(img_batch)
                else:
                    output = model(img_batch)
                pred = torch.sigmoid(output)[0, 0].cpu().numpy()
            
            iou = calculate_iou(pred, true_mask)
            
            ax.imshow(original_img, cmap='gray', alpha=0.6)
            pred_binary = (pred > 0.5).astype(np.float32)
            true_binary = (true_mask > 0.5).astype(np.float32)
            
            overlay = np.zeros((*pred_binary.shape, 4))
            tp = (pred_binary == 1) & (true_binary == 1)
            fp = (pred_binary == 1) & (true_binary == 0)
            fn = (pred_binary == 0) & (true_binary == 1)
            
            overlay[tp] = [0, 1, 0, 0.6]
            overlay[fp] = [1, 0, 0, 0.6]
            overlay[fn] = [0, 0, 1, 0.6]
            ax.imshow(overlay)
            
            add_annotations(ax, model_name, tp, fp, fn)
            
            if row == 0:
                ax.set_title(model_name, fontsize=8, fontweight='bold')
            ax.set_xlabel(f'IoU:{iou:.3f}', fontsize=8)
            ax.axis('off')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='True Positive (Correct Detection)'),
        Patch(facecolor='red', alpha=0.6, label='False Positive (False Alarm)'),
        Patch(facecolor='blue', alpha=0.6, label='False Negative (Missed Spill)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, -0.02))
    
    plt.suptitle('Best/Medium/Worst Cases - All 9 Models', fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig('figures/best_worst_real_cases.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: best_worst_real_cases.png")

def create_qualitative_comparison_clean():
    """Create qualitative comparison figure."""
    print("\nCreating qualitative comparison...")
    
    models = load_all_models()
    model_list = list(models.items())
    
    val_img_dir = 'dataset/images/images/val'
    val_mask_dir = 'dataset/masks/masks/val'
    
    _, _, transform_val, _ = get_transforms()
    val_dataset = OilSpillDataset(val_img_dir, val_mask_dir, transform_val, transform_val)
    
    # Get samples
    samples = [0, 25, 50]
    
    fig = plt.figure(figsize=(26, 10))
    gs = GridSpec(len(samples), 11, figure=fig, hspace=0.2, wspace=0.1)
    
    for row, idx in enumerate(samples):
        img_filename = val_dataset.image_files[idx]
        original_img = np.array(Image.open(os.path.join(val_img_dir, img_filename)).convert("RGB"))
        img_tensor, mask_tensor = val_dataset[idx]
        true_mask = mask_tensor.squeeze(0).cpu().numpy()
        
        # Input
        ax = fig.add_subplot(gs[row, 0])
        ax.imshow(original_img, cmap='gray')
        if row == 0:
            ax.set_title('Input SAR', fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # Ground Truth
        ax = fig.add_subplot(gs[row, 1])
        ax.imshow(original_img, cmap='gray', alpha=0.6)
        gt_colored = np.zeros((*true_mask.shape, 4))
        gt_colored[true_mask > 0.5] = [0, 1, 0, 0.5]
        ax.imshow(gt_colored)
        if row == 0:
            ax.set_title('Ground Truth', fontsize=9, fontweight='bold')
        ax.axis('off')
        
        # All 9 models
        for col, (model_name, (model, use_boundary)) in enumerate(model_list, start=2):
            ax = fig.add_subplot(gs[row, col])
            
            with torch.no_grad():
                img_batch = img_tensor.unsqueeze(0).to(device)
                if use_boundary:
                    output, _ = model(img_batch)
                else:
                    output = model(img_batch)
                pred = torch.sigmoid(output)[0, 0].cpu().numpy()
            
            ax.imshow(original_img, cmap='gray', alpha=0.6)
            pred_binary = (pred > 0.5).astype(np.float32)
            true_binary = (true_mask > 0.5).astype(np.float32)
            
            overlay = np.zeros((*pred_binary.shape, 4))
            tp = (pred_binary == 1) & (true_binary == 1)
            fp = (pred_binary == 1) & (true_binary == 0)
            fn = (pred_binary == 0) & (true_binary == 1)
            
            overlay[tp] = [0, 1, 0, 0.6]
            overlay[fp] = [1, 0, 0, 0.6]
            overlay[fn] = [0, 0, 1, 0.6]
            ax.imshow(overlay)
            
            add_annotations(ax, model_name, tp, fp, fn)
            
            if row == 0:
                ax.set_title(model_name, fontsize=8, fontweight='bold')
            ax.axis('off')

    legend_elements = [
        Patch(facecolor='green', alpha=0.6, label='True Positive (Correct Detection)'),
        Patch(facecolor='red', alpha=0.6, label='False Positive (False Alarm)'),
        Patch(facecolor='blue', alpha=0.6, label='False Negative (Missed Spill)')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=3, fontsize=12,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle('Qualitative Comparison - All 9 Models', fontsize=16, fontweight='bold', y=0.98)
    plt.savefig('figures/qualitative_comparison_clean.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Created: qualitative_comparison_clean.png")

if __name__ == "__main__":
    print("Creating Qualitative Figures with ALL 9 Models")
    create_figure4_all_models()
    create_best_worst_all_models()
    create_qualitative_comparison_clean()
    print("Done! All qualitative figures created with 9 models.")
