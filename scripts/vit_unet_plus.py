"""
ViT-UNet variant with 2/5/8 lateral skips plus block-11 bottleneck fusion.

The public model id keeps the historical ``4skip`` suffix so existing
checkpoints and scripts remain compatible. Architecturally, blocks 2, 5, and
8 are lateral skip maps; block 11 is the deepest ViT representation fused into
the final decoder stage (up0/conv0), not a separate encoder stage.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models import double_conv


class ViTEncoder4Skip(nn.Module):
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
        self.skip_layers = {2, 5, 8, 11}

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        skips = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.skip_layers:
                t = x[:, 1:, :].permute(0, 2, 1)
                t = t.view(B, self.embed_dim, 14, 14)
                skips.append(t)

        x = self.norm(x)
        x = x[:, 1:, :].permute(0, 2, 1).view(B, self.embed_dim, 14, 14)
        # final feature is the normalized block-11 bottleneck representation
        return x, skips  # skips: [b2, b5, b8, b11]


class ViTUNet4Skip(nn.Module):
    """Compatibility name for ViT-UNet with 2/5/8 skips + block-11 bottleneck."""

    def __init__(self, model_name="vit_small_patch16_224", out_channels=1, pretrained=True):
        super().__init__()
        self.encoder = ViTEncoder4Skip(model_name=model_name, pretrained=pretrained)
        d = self.encoder.embed_dim

        self.up3 = nn.ConvTranspose2d(d, 512, kernel_size=2, stride=2)
        self.conv3 = double_conv(512 + d, 512)  # uses skip b8

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = double_conv(256 + d, 256)  # uses skip b5

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = double_conv(128 + d, 128)  # uses skip b2

        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0 = double_conv(64 + d, 64)  # fuses block-11 bottleneck

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        final_feat, skips = self.encoder(x)
        s_b2, s_b5, s_b8, s_b11 = skips

        x = self.up3(final_feat)
        s = F.interpolate(s_b8, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv3(torch.cat([x, s], dim=1))

        x = self.up2(x)
        s = F.interpolate(s_b5, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv2(torch.cat([x, s], dim=1))

        x = self.up1(x)
        s = F.interpolate(s_b2, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv1(torch.cat([x, s], dim=1))

        x = self.up0(x)
        s = F.interpolate(s_b11, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = self.conv0(torch.cat([x, s], dim=1))

        x = self.final_conv(x)
        return x
