"""
Unified trainer for all nine segmentation architectures used in the paper.

One script, one protocol, one CSV schema, one checkpoint-selection rule.

Protocol (identical for every model):
  - Optimizer: AdamW, lr=1e-4, weight_decay=1e-4 (Adam variant standard for
    transformer + CNN segmentation; per-architecture lr-schedule recipes from
    the original papers were not adopted to keep the comparison apples-to-apples).
  - Loss: CombinedLoss(BCE 0.8 + Dice 0.2). Optional --pos_weight scales the
    BCE positive-class weight (>1 favours recall on rare oil-spill pixels).
  - LR schedule: ReduceLROnPlateau, mode='min' on val_loss, patience=5, factor=0.5.
  - Max 100 epochs.
  - Early stopping: patience=15 on validation loss. We deliberately do NOT
    select the checkpoint on the same metric we report (mIoU): selecting on
    val_loss is the model-selection criterion adopted by the eight published
    baselines (their public training scripts all save best-val_loss), and
    keeping the same criterion across all nine architectures avoids giving
    any single model an unfair selection-tuning advantage.
  - Checkpoint: saved at the epoch with the lowest validation loss.
  - 3 seeds (42, 123, 7); deterministic via set_seed.

Architectures: vit_unet (legacy 2/5/8 baseline), vit_unet_4skip (proposed), and
the eight published baselines (unet, deeplabv3plus, cbdnet, transunet,
dsunet, lraunet, daenet, segformer). The vit_unet_4skip id is retained for
checkpoint compatibility; architecturally it uses lateral skips from blocks
2/5/8 and fuses the block-11 bottleneck feature into the final decoder stage.
With --pos_weight 1.5 this is the configuration reported as our best model.

Datasets:
  --dataset refined  : --data_path ./dataset (single tree, 6455 train / 1615 val)
  --dataset original : --data_path ./dataset_orig (palsar+sentinel concat;
                       'train' for train, 'test' as val to match the original
                       Deep-SAR protocol)

Outputs:
  {save_dir}/{tag}_seed{S}_best.pth        — best-mIoU checkpoint
  {save_dir}/{tag}_seed{S}_history.csv     — per-epoch metrics
"""
import os
import sys
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import (
    ViTUNet, UNetBaseline, DeepLabV3Plus, CBDNet,
    TransUNet, DSUNet, LRAUNet, DAENet, SegFormerWrapper,
)
from vit_unet_plus import ViTUNet4Skip
from dataset import OilSpillDataset, get_transforms, load_dataset_orig
from utils import MetricsCalculator, CombinedLoss, set_seed


class WeightedCombinedLoss(nn.Module):
    """BCE(pos_weight) + Dice. Used when --pos_weight != 1.0."""
    def __init__(self, bce_weight=0.8, dice_weight=0.2, pos_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, logits, targets):
        pw = self.pos_weight.to(logits.device)
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw
        )
        probs = torch.sigmoid(logits)
        eps = 1e-6
        inter = (probs * targets).sum()
        dice = 1.0 - (2 * inter + eps) / (probs.sum() + targets.sum() + eps)
        return self.bce_weight * bce + self.dice_weight * dice


MODEL_FACTORY = {
    'vit_unet':           lambda: (ViTUNet(model_name="vit_small_patch16_224", pretrained=True), False),
    'vit_unet_4skip':     lambda: (ViTUNet4Skip(model_name="vit_small_patch16_224", pretrained=True), False),
    'unet':               lambda: (UNetBaseline(), False),
    'deeplabv3plus':      lambda: (DeepLabV3Plus(pretrained=True), False),
    'cbdnet':             lambda: (CBDNet(), True),
    'transunet':          lambda: (TransUNet(pretrained=True), False),
    'dsunet':             lambda: (DSUNet(), True),
    'lraunet':            lambda: (LRAUNet(), False),
    'daenet':             lambda: (DAENet(), False),
    'segformer':          lambda: (SegFormerWrapper(pretrained=True), False),
}


def get_args():
    p = argparse.ArgumentParser(
        description='Unified trainer for the SAR oil-spill segmentation paper',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--model', type=str, default='vit_unet_4skip',
                   choices=list(MODEL_FACTORY.keys()),
                   help='Architecture to train.')
    p.add_argument('--dataset', type=str, default='refined',
                   choices=['refined', 'original'],
                   help='refined: --data_path ./dataset (single tree); '
                        'original: --data_path ./dataset_orig (palsar+sentinel concat, val=test).')
    p.add_argument('--data_path', type=str, default='./dataset')
    p.add_argument('--epochs', type=int, default=100,
                   help='Maximum number of epochs (early stopping may end earlier).')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--bce_weight', type=float, default=0.8)
    p.add_argument('--dice_weight', type=float, default=0.2)
    p.add_argument('--pos_weight', type=float, default=1.0,
                   help='BCE positive-class weight (>1 favours recall). '
                        'Default 1.0 reproduces the unweighted CombinedLoss baseline.')
    p.add_argument('--lr_patience', type=int, default=5,
                   help='ReduceLROnPlateau patience (on val_loss).')
    p.add_argument('--early_stop_patience', type=int, default=15,
                   help='Early-stopping patience on validation loss. '
                        'Set to <=0 to disable.')
    p.add_argument('--save_dir', type=str, default='./models_results')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tag', type=str, default=None,
                   help='Output filename prefix. Defaults to --model.')
    p.add_argument('--device', type=str, default='cuda')
    return p.parse_args()


def make_loaders(args):
    transform_train, transform_mask, transform_val, _ = get_transforms()
    if args.dataset == 'original':
        train_ds = load_dataset_orig(args.data_path, split='train',
                                     transform_img=transform_train,
                                     transform_mask=transform_mask)
        val_ds = load_dataset_orig(args.data_path, split='test',
                                   transform_img=transform_val,
                                   transform_mask=transform_mask)
        if train_ds is None or val_ds is None:
            raise FileNotFoundError(
                f"--dataset original expects palsar+sentinel under "
                f"{args.data_path}/train and {args.data_path}/test")
    else:
        train_img = os.path.join(args.data_path, 'train', 'images')
        train_msk = os.path.join(args.data_path, 'train', 'masks')
        val_img = os.path.join(args.data_path, 'val', 'images')
        val_msk = os.path.join(args.data_path, 'val', 'masks')
        if not os.path.exists(train_img):
            if os.path.exists(os.path.join(args.data_path, 'images', 'images', 'train')):
                train_img = os.path.join(args.data_path, 'images', 'images', 'train')
                train_msk = os.path.join(args.data_path, 'masks', 'masks', 'train')
                val_img = os.path.join(args.data_path, 'images', 'images', 'val')
                val_msk = os.path.join(args.data_path, 'masks', 'masks', 'val')
            else:
                train_img = os.path.join(args.data_path, 'images', 'train')
                train_msk = os.path.join(args.data_path, 'masks', 'train')
                val_img = os.path.join(args.data_path, 'images', 'val')
                val_msk = os.path.join(args.data_path, 'masks', 'val')
        train_ds = OilSpillDataset(train_img, train_msk, transform_train, transform_mask)
        val_ds = OilSpillDataset(val_img, val_msk, transform_val, transform_mask)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, len(train_ds), len(val_ds)


def train_model(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader, n_train, n_val = make_loaders(args)
    print(f"[{args.model} seed {args.seed}] dataset={args.dataset} "
          f"train={n_train} val={n_val} max_epochs={args.epochs} "
          f"early_stop_patience={args.early_stop_patience} (on val_loss)")

    model, use_boundary = MODEL_FACTORY[args.model]()
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.2f} M")

    if args.pos_weight != 1.0:
        criterion = WeightedCombinedLoss(args.bce_weight, args.dice_weight, args.pos_weight)
    else:
        criterion = CombinedLoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=args.lr_patience
    )

    tag = args.tag if args.tag else args.model
    history_path = os.path.join(args.save_dir, f"{tag}_seed{args.seed}_history.csv")
    ckpt_path = os.path.join(args.save_dir, f"{tag}_seed{args.seed}_best.pth")
    with open(history_path, 'w') as f:
        f.write("epoch,train_loss,val_loss,mIoU,F1,Precision,Recall,Specificity,Accuracy,lr,sec\n")

    best_val_loss = float('inf')
    best_miou_at_best_loss = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            if use_boundary:
                out, boundary = model(images)
                loss = criterion(out, masks, boundary)
            else:
                out = model(images)
                loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        mc = MetricsCalculator()
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                if use_boundary:
                    out, _ = model(images)
                else:
                    out = model(images)
                vl = criterion(out, masks)
                val_loss += vl.item()
                preds = (torch.sigmoid(out) > 0.5).float()
                mc.update(preds, masks)
        val_loss /= len(val_loader)
        m = mc.get_metrics()
        scheduler.step(val_loss)
        cur_lr = optimizer.param_groups[0]['lr']
        dt = time.time() - t0

        with open(history_path, 'a') as f:
            f.write(f"{epoch},{train_loss:.6f},{val_loss:.6f},"
                    f"{m['mIoU']:.6f},{m['F1']:.6f},{m['Precision']:.6f},"
                    f"{m['Recall']:.6f},{m['Specificity']:.6f},{m['Accuracy']:.6f},"
                    f"{cur_lr:.2e},{dt:.1f}\n")

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_miou_at_best_loss = m['mIoU']
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            epochs_no_improve += 1

        flag = "*" if improved else " "
        print(f"[{args.model} seed {args.seed}] ep {epoch:3d}/{args.epochs} "
              f"tr={train_loss:.4f} vl={val_loss:.4f} "
              f"mIoU={m['mIoU']:.4f} F1={m['F1']:.4f} "
              f"P={m['Precision']:.4f} R={m['Recall']:.4f} "
              f"lr={cur_lr:.1e} ({dt:.0f}s) {flag}")

        if args.early_stop_patience > 0 and epochs_no_improve >= args.early_stop_patience:
            print(f"[{args.model} seed {args.seed}] early stopping at epoch {epoch} "
                  f"(no val_loss improvement for {args.early_stop_patience} epochs)")
            break

    print(f"\n[{args.model} seed {args.seed}] DONE. "
          f"best val_loss={best_val_loss:.4f} (mIoU={best_miou_at_best_loss:.4f}) "
          f"at epoch {best_epoch}")
    print(f"  ckpt:    {ckpt_path}")
    print(f"  history: {history_path}")


if __name__ == '__main__':
    train_model(get_args())
