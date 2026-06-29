"""
Model architectures for oil spill segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import segmentation_models_pytorch as smp

def double_conv(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True)
    )

class ViTEncoder(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", pretrained=True):
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
            if i in {2, 5, 8}:
                skip = x[:, 1:, :].permute(0, 2, 1)
                skip = skip.view(B, self.embed_dim, 14, 14)
                skip_features.append(skip)
        
        x = self.norm(x)
        x = x[:, 1:, :].permute(0, 2, 1).view(B, self.embed_dim, 14, 14)
        return x, skip_features

class ViTUNet(nn.Module):
    def __init__(self, model_name="vit_small_patch16_224", out_channels=1, pretrained=True):
        super().__init__()
        self.encoder = ViTEncoder(model_name=model_name, pretrained=pretrained)
        vit_dim = self.encoder.embed_dim
        
        self.up3 = nn.ConvTranspose2d(vit_dim, 512, kernel_size=2, stride=2)
        self.conv3 = double_conv(512 + vit_dim, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = double_conv(256 + vit_dim, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = double_conv(128 + vit_dim, 128)
        
        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0 = double_conv(64, 64)
        
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
        x = self.final_conv(x)
        return x

class UNetBaseline(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.bottleneck = double_conv(512, 1024)
        
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = double_conv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = double_conv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = double_conv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(128, 64)
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels, 1),
            nn.Sigmoid()
        )
        self.sSE = nn.Sequential(
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)

class CBDNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()
        # Backbone ResNet-50
        self.cnn = timm.create_model("resnet50", pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        # resnet50 outputs: [B, 64], [B, 256], [B, 512], [B, 1024], [B, 2048]
        
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.att4 = scSE(1024)
        self.dec4 = double_conv(1024 + 1024, 1024)
        
        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att3 = scSE(512)
        self.dec3 = double_conv(512 + 512, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = scSE(256)
        self.dec2 = double_conv(256 + 256, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.att1 = scSE(64)
        self.dec1 = double_conv(64 + 64, 64)
        
        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec0 = double_conv(64 + 64, 64)
        
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        features = self.cnn(x)
        # e0 is [B, 64] (1/2 res), e1 is [B, 256] (1/4 res), e2 is [B, 512] (1/8 res), e3 is [B, 1024] (1/16 res), e4 is [B, 2048] (1/32 res)
        e0, e1, e2, e3, e4 = features[0], features[1], features[2], features[3], features[4]
        
        d4 = self.up4(e4)
        d4 = self.att4(d4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.att3(d3)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.att2(d2)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.att1(d1)
        # Resize e0 to match d1 (e0 is 1/2 res, d1 is 1/2 res but dimensions might slightly off due to pooling)
        e0 = F.interpolate(e0, size=d1.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e0], dim=1))
        
        # Final upsample to original resolution (from 1/2 to 1)
        d0 = self.up0(d1)
        # We don't have a full resolution skip from ResNet, just upsample
        d0_skip = F.interpolate(features[0], size=d0.shape[-2:], mode='bilinear', align_corners=False)
        d0 = self.dec0(torch.cat([d0, d0_skip], dim=1))
        
        seg_out = self.final(d0)
        boundary_out = self.boundary_conv(d0)
        return seg_out, boundary_out

class DeepLabV3Plus(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_channels,
            classes=out_channels
        )
    def forward(self, x):
        return self.model(x)

class TransUNet(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", out_channels=1, pretrained=True):
        super().__init__()
        # CNN Backbone (ResNet-50)
        self.cnn = timm.create_model("resnet50", pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3))
        # CNN outputs:
        # idx 0: [B, 64, 112, 112]
        # idx 1: [B, 256, 56, 56]
        # idx 2: [B, 512, 28, 28]
        # idx 3: [B, 1024, 14, 14]

        # ViT Encoder (takes CNN features as tokens)
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim
        
        # Projection from CNN 1024 to ViT embed_dim (768)
        self.patch_proj = nn.Conv2d(1024, self.embed_dim, kernel_size=1)
        
        # Decoder with multi-scale skip connections
        self.up3 = nn.ConvTranspose2d(self.embed_dim, 512, kernel_size=2, stride=2)
        self.conv3 = double_conv(512 + 512, 512) # +512 from CNN layer 2
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = double_conv(256 + 256, 256) # +256 from CNN layer 1
        
        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.conv1 = double_conv(64 + 64, 64) # +64 from CNN layer 0
        
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv0 = double_conv(32, 32)
        
        self.final_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        B = x.shape[0]
        
        # CNN Feature Extraction
        cnn_features = self.cnn(x)
        skip1 = cnn_features[0] # [B, 64, 112, 112]
        skip2 = cnn_features[1] # [B, 256, 56, 56]
        skip3 = cnn_features[2] # [B, 512, 28, 28]
        cnn_out = cnn_features[3] # [B, 1024, 14, 14]
        
        # Prepare for ViT
        v_tokens = self.patch_proj(cnn_out) # [B, 768, 14, 14]
        v_tokens = v_tokens.flatten(2).transpose(1, 2) # [B, 196, 768]
        
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        v_tokens = torch.cat((cls_token, v_tokens), dim=1) # [B, 197, 768]
        
        # Interpolate positional embedding to match patch count
        pos_embed = self.vit.pos_embed
        v_tokens = v_tokens + pos_embed
        v_tokens = self.vit.pos_drop(v_tokens)
        
        for blk in self.vit.blocks:
            v_tokens = blk(v_tokens)
            
        v_tokens = self.vit.norm(v_tokens)
        v_tokens = v_tokens[:, 1:] # Remove CLS token
        v_tokens = v_tokens.transpose(1, 2).view(B, self.embed_dim, 14, 14)
        
        # Cascaded Upsampler (CUP) with Skip Connections
        d3 = self.up3(v_tokens)
        d3 = torch.cat([d3, skip3], dim=1)
        d3 = self.conv3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, skip2], dim=1)
        d2 = self.conv2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.conv1(d1)
        
        d0 = self.up0(d1)
        d0 = self.conv0(d0)
        
        return self.final_conv(d0)

class DSUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = double_conv(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = double_conv(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = double_conv(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = double_conv(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        self.edge_conv1 = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.edge_conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.bottleneck = double_conv(256, 512)
        
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = double_conv(256 + 256 + 16, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = double_conv(128 + 128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = double_conv(32 + 32, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        self.edge_final = nn.Conv2d(16, 1, kernel_size=1)
    
    def forward(self, x):
        edge = F.relu(self.edge_conv1(x))
        edge = self.edge_conv2(edge)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        d4 = self.up4(b)
        edge_up = F.interpolate(edge, size=d4.shape[-2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4, edge_up], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1), self.edge_final(edge)

class SimAM(nn.Module):
    def __init__(self, e_lambda=1e-4):
        super().__init__()
        self.act = nn.Sigmoid()
        self.e_lambda = e_lambda
    def forward(self, x):
        b, c, h, w = x.shape
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class LRAResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv_block = nn.Sequential(
            DepthwiseSeparableConv(in_ch, out_ch),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(out_ch, out_ch),
            nn.BatchNorm2d(out_ch)
        )
        self.simam = SimAM()
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.simam(out)
        out += self.shortcut(x)
        return F.relu(out, inplace=True)

class LRAUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = LRAResBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = LRAResBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = LRAResBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = LRAResBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2)
        
        self.bottleneck = LRAResBlock(256, 512)
        
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = LRAResBlock(256 + 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = LRAResBlock(128 + 128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = LRAResBlock(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = LRAResBlock(32 + 32, 32)
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))
        
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)

class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class CAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out

class DAEM(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.pam = PAM_Module(in_dim)
        self.cam = CAM_Module(in_dim)
    def forward(self, x):
        return self.pam(x) + self.cam(x)

class DAENet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, pretrained=True):
        super().__init__()
        # Backbone ResNet-34
        self.cnn = timm.create_model("resnet34", pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3, 4))
        # resnet34 outputs: [B, 64], [B, 64], [B, 128], [B, 256], [B, 512]
        
        self.daem1 = DAEM(64)
        self.daem2 = DAEM(128)
        self.daem3 = DAEM(256)
        self.daem4 = DAEM(512)
        
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4 = double_conv(256 + 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = double_conv(128 + 128, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = double_conv(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.dec1 = double_conv(64 + 64, 64)
        self.up0 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec0 = double_conv(32, 32)
        
        self.final = nn.Conv2d(32, out_channels, kernel_size=1)
        
    def forward(self, x):
        features = self.cnn(x)
        # e0 is 1/2 res [B, 64], e1 is 1/4 res [B, 64], e2 is 1/8 [B, 128], e3 is 1/16 [B, 256], e4 is 1/32 [B, 512]
        e0 = features[0]
        e1 = self.daem1(features[1])
        e2 = self.daem2(features[2])
        e3 = self.daem3(features[3])
        e4 = self.daem4(features[4])
        
        d4 = self.dec4(torch.cat([self.up4(e4), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e2], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e1], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e0], dim=1))
        d0 = self.dec0(self.up0(d1))
        
        return self.final(d0)

class SegFormerWrapper(nn.Module):
    def __init__(self, out_channels=1, pretrained=True):
        super().__init__()
        from transformers import SegformerForSemanticSegmentation, SegformerConfig
        if pretrained:
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                "nvidia/mit-b0", 
                num_labels=out_channels, 
                ignore_mismatched_sizes=True
            )
        else:
            config = SegformerConfig(num_labels=out_channels)
            self.model = SegformerForSemanticSegmentation(config)
            
    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        # Upsample to original image size
        logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits

class ViTUNetSingleSkip(nn.Module):
    def __init__(self, model_name="vit_small_patch16_224", out_channels=1, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim
        self.up3 = nn.ConvTranspose2d(self.embed_dim, 512, kernel_size=2, stride=2)
        self.conv3 = double_conv(512, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = double_conv(256, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = double_conv(128, 128)
        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0 = double_conv(64, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        x = x[:, 1:, :].permute(0, 2, 1).view(B, self.embed_dim, 14, 14)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up0(x)
        x = self.conv0(x)
        x = self.final_conv(x)
        return x

class ViTUNetNoSkip(nn.Module):
    def __init__(self, model_name="vit_small_patch16_224", out_channels=1, pretrained=True):
        super().__init__()
        self.vit = timm.create_model(model_name, pretrained=pretrained)
        self.embed_dim = self.vit.embed_dim
        self.up3 = nn.ConvTranspose2d(self.embed_dim, 512, kernel_size=2, stride=2)
        self.conv3 = double_conv(512, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = double_conv(256, 256)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = double_conv(128, 128)
        self.up0 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv0 = double_conv(64, 64)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = blk(x)
        x = self.vit.norm(x)
        x = x[:, 1:, :].permute(0, 2, 1).view(B, self.embed_dim, 14, 14)
        x = self.up3(x)
        x = self.conv3(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up0(x)
        x = self.conv0(x)
        x = self.final_conv(x)
        return x


class ViTEncoderDeepSkip(nn.Module):
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
