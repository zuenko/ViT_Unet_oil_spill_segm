"""
Evaluation script for trained models

Usage:
    python scripts/evaluate.py --model vit_unet --model_path models_results/vit_unet_best.pth
    python scripts/evaluate.py --model cbdnet --model_path models_results/cbdnet_best.pth --save_predictions
"""
import os
import sys
import argparse

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import (
    ViTUNet, UNetBaseline, DeepLabV3Plus, CBDNet,
    TransUNet, DSUNet, LRAUNet, SegFormerWrapper
)
from dataset import OilSpillDataset, get_transforms
from utils import MetricsCalculator, set_seed


def get_args():
    parser = argparse.ArgumentParser(
        description='Evaluate oil spill segmentation models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--model', type=str, default='vit_unet',
                       choices=['vit_unet', 'unet', 'deeplabv3plus', 'cbdnet',
                               'transunet', 'dsunet', 'lraunet', 'segformer'],
                       help='Model architecture')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model weights')
    parser.add_argument('--data_path', type=str, default='./dataset',
                       help='Path to dataset')
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predicted masks as images')
    return parser.parse_args()


def evaluate_model(args):
    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(args.save_dir, exist_ok=True)
    if args.save_predictions:
        os.makedirs(os.path.join(args.save_dir, 'predictions'), exist_ok=True)
    
    # Load data
    train_img_dir = os.path.join(args.data_path, 'train', 'images')
    train_mask_dir = os.path.join(args.data_path, 'train', 'masks')
    test_img_dir = os.path.join(args.data_path, 'val', 'images')
    test_mask_dir = os.path.join(args.data_path, 'val', 'masks')
    
    # Check if directories exist, try alternative structure
    if not os.path.exists(test_img_dir):
        if os.path.exists(os.path.join(args.data_path, 'images', 'images', 'val')):
            test_img_dir = os.path.join(args.data_path, 'images', 'images', 'val')
            test_mask_dir = os.path.join(args.data_path, 'masks', 'masks', 'val')
        else:
            test_img_dir = os.path.join(args.data_path, 'images', 'val')
            test_mask_dir = os.path.join(args.data_path, 'masks', 'val')
    
    _, _, transform_test, transform_mask = get_transforms()
    
    test_dataset = OilSpillDataset(test_img_dir, test_mask_dir, 
                                   transform_test, transform_mask)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    if args.model == 'vit_unet':
        model = ViTUNet(model_name="vit_small_patch16_224", pretrained=False)
    elif args.model == 'unet':
        model = UNetBaseline()
    elif args.model == 'deeplabv3plus':
        model = DeepLabV3Plus(pretrained=False)
    elif args.model == 'cbdnet':
        model = CBDNet()
    elif args.model == 'transunet':
        model = TransUNet(pretrained=False)
    elif args.model == 'dsunet':
        model = DSUNet()
    elif args.model == 'lraunet':
        model = LRAUNet()
    elif args.model == 'segformer':
        model = SegFormerWrapper(pretrained=False)
    
    # Load weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from: {args.model_path}\n")
    
    # Evaluate
    metrics_calc = MetricsCalculator()
    all_ious = []
    all_f1s = []
    all_names = []
    
    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            images, masks = images.to(device), masks.to(device)
            
            if args.model == 'cbdnet':
                outputs, _ = model(images)
            else:
                outputs = model(images)
            
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            
            metrics_calc.update(probs, masks)
            
            # Per-image metrics
            for j in range(images.size(0)):
                pred = preds[j].cpu().numpy().squeeze()
                mask = masks[j].cpu().numpy().squeeze()
                
                intersection = np.sum(pred * mask)
                union = np.sum((pred + mask) > 0)
                iou = intersection / (union + 1e-6)
                
                tp = np.sum(pred * mask)
                fp = np.sum(pred * (1 - mask))
                fn = np.sum((1 - pred) * mask)
                f1 = 2 * tp / (2 * tp + fp + fn + 1e-6)
                
                all_ious.append(iou)
                all_f1s.append(f1)
                
                sample_name = test_dataset.image_files[i*8+j].split('.')[0]
                all_names.append(sample_name)
                
                # Save prediction if requested
                if args.save_predictions:
                    pred_img = (pred * 255).astype(np.uint8)
                    save_path = os.path.join(args.save_dir, 'predictions', f'{sample_name}_pred.png')
                    Image.fromarray(pred_img).save(save_path)
    
    # Get aggregate metrics
    metrics = metrics_calc.get_metrics()
    
    print("="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"mIoU:        {metrics['mIoU']:.4f}")
    print(f"F1-Score:    {metrics['F1']:.4f}")
    print(f"Precision:   {metrics['Precision']:.4f}")
    print(f"Recall:      {metrics['Recall']:.4f}")
    print(f"Accuracy:    {metrics['Accuracy']:.4f}")
    print(f"Specificity: {metrics['Specificity']:.4f}")
    print("="*60)
    
    # Save detailed results
    results_df = pd.DataFrame({
        'sample': all_names,
        'IoU': all_ious,
        'F1': all_f1s
    })
    results_df.to_csv(os.path.join(args.save_dir, 'per_sample_results.csv'), index=False)
    
    summary = pd.DataFrame([{
        'Model': args.model,
        'mIoU': metrics['mIoU'],
        'F1': metrics['F1'],
        'Precision': metrics['Precision'],
        'Recall': metrics['Recall'],
        'Accuracy': metrics['Accuracy'],
        'Specificity': metrics['Specificity']
    }])
    summary.to_csv(os.path.join(args.save_dir, 'summary_metrics.csv'), index=False)
    
    print(f"\nResults saved to: {args.save_dir}")
    
    return metrics


if __name__ == "__main__":
    args = get_args()
    evaluate_model(args)
