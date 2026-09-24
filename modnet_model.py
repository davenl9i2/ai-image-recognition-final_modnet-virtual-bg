"""
modnet_model.py — MODNet 模型架構定義
來源：https://github.com/ZHKKKe/MODNet (MIT License)
這裡保留最小可執行版本，供期末報告使用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── MobileNetV2 backbone (簡化版) ─────────────────────────────────────────────

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers += [
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        input_channel = 32
        inverted_residual_setting = [
            [1, 16,  1, 1],
            [6, 24,  2, 2],
            [6, 32,  3, 2],
            [6, 64,  4, 2],
            [6, 96,  3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        features = [ConvBNReLU(3, input_channel, stride=2)]
        for t, c, n, s in inverted_residual_setting:
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, c, stride, t))
                input_channel = c
        self.features = nn.Sequential(*features)
        self._initialize_weights()

    def forward(self, x):
        enc2x  = self.features[0:2](x)
        enc4x  = self.features[2:4](enc2x)
        enc8x  = self.features[4:7](enc4x)
        enc16x = self.features[7:14](enc8x)
        enc32x = self.features[14:19](enc16x)
        return enc2x, enc4x, enc8x, enc16x, enc32x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


# ── MODNet 組件 ───────────────────────────────────────────────────────────────

class IBNorm(nn.Module):
    """Instance-Batch Normalization"""
    def __init__(self, in_channels):
        super().__init__()
        in_channels = in_channels
        self.bnorm_channels = int(in_channels / 2)
        self.inorm_channels = in_channels - self.bnorm_channels
        self.bnorm = nn.BatchNorm2d(self.bnorm_channels, affine=True)
        self.inorm = nn.InstanceNorm2d(self.inorm_channels, affine=False)

    def forward(self, x):
        bn_x  = self.bnorm(x[:, :self.bnorm_channels, ...].contiguous())
        in_x  = self.inorm(x[:, self.bnorm_channels:,  ...].contiguous())
        return torch.cat((bn_x, in_x), 1)


class Conv2dIBNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 with_ibn=True, with_relu=True):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size,
                            stride=stride, padding=padding,
                            dilation=dilation, groups=groups, bias=False)]
        if with_ibn:
            layers.append(IBNorm(out_channels))
        if with_relu:
            layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reduction=1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Sequential(
            nn.Linear(in_channels, int(in_channels // reduction), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(int(in_channels // reduction), out_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        n, c, _, _ = x.size()
        y = self.pool(x).view(n, c)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


# ── Low-Resolution Branch (LR Branch) ────────────────────────────────────────

class LRBranch(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        enc_channels = [32, 16, 24, 32, 96, 1280]
        self.backbone = backbone
        self.se_block  = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x  = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr    = Conv2dIBNormRelu(enc_channels[2], 1, 3, stride=2, padding=1, with_ibn=False, with_relu=False)

    def forward(self, img, inference):
        enc2x, enc4x, enc8x, enc16x, enc32x = self.backbone(img)
        enc32x = self.se_block(enc32x)
        lr16x  = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x  = self.conv_lr16x(lr16x)
        lr8x   = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x   = self.conv_lr8x(lr8x)
        pred_semantic = None
        if not inference:
            lr     = self.conv_lr(lr8x)
            pred_semantic = torch.sigmoid(lr)
        return pred_semantic, lr8x, [enc2x, enc4x, enc8x, enc16x, enc32x]


# ── High-Resolution Branch (HR Branch) ───────────────────────────────────────

class HRBranch(nn.Module):
    def __init__(self, hr_channels, enc_channels):
        super().__init__()
        self.tohr_enc2x  = Conv2dIBNormRelu(enc_channels[0], hr_channels, 1, stride=1, padding=0)
        self.conv_enc2x  = Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=2, padding=1)
        self.tohr_enc4x  = Conv2dIBNormRelu(enc_channels[1], hr_channels, 1, stride=1, padding=0)
        self.conv_enc4x  = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_hr4x   = nn.Sequential(
            Conv2dIBNormRelu(3 * hr_channels + 3, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )
        self.conv_hr2x   = nn.Sequential(
            Conv2dIBNormRelu(2 * hr_channels, 2 * hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1),
        )
        self.conv_hr     = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, hr_channels, 3, stride=1, padding=1),
            Conv2dIBNormRelu(hr_channels, 1, 1, stride=1, padding=0, with_ibn=False, with_relu=False),
        )

    def forward(self, img, enc2x, enc4x, lr8x, inference):
        img2x  = F.interpolate(img,   scale_factor=1/2, mode='bilinear', align_corners=False)
        img4x  = F.interpolate(img,   scale_factor=1/4, mode='bilinear', align_corners=False)
        enc2x  = self.tohr_enc2x(enc2x)
        hr4x   = self.conv_enc2x(torch.cat((img2x, enc2x), dim=1))
        enc4x  = self.tohr_enc4x(enc4x)
        hr4x   = self.conv_enc4x(torch.cat((hr4x, enc4x), dim=1))
        lr4x   = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        hr4x   = self.conv_hr4x(torch.cat((hr4x, lr4x, img4x), dim=1))
        hr2x   = F.interpolate(hr4x, scale_factor=2, mode='bilinear', align_corners=False)
        hr2x   = self.conv_hr2x(torch.cat((hr2x, enc2x), dim=1))
        pred_detail = None
        if not inference:
            hr     = F.interpolate(hr2x, scale_factor=2, mode='bilinear', align_corners=False)
            pred_detail = torch.sigmoid(self.conv_hr(torch.cat((hr, img), dim=1)))
        return pred_detail, hr2x


# ── Fusion Branch ─────────────────────────────────────────────────────────────

class FusionBranch(nn.Module):
    def __init__(self, hr_channels, enc_channels):
        super().__init__()
        self.conv_lr4x = Conv2dIBNormRelu(enc_channels[2], hr_channels, 5, stride=1, padding=2)
        self.conv_f2x  = Conv2dIBNormRelu(2 * hr_channels, hr_channels, 3, stride=1, padding=1)
        self.conv_f    = nn.Sequential(
            Conv2dIBNormRelu(hr_channels + 3, int(hr_channels / 2), 3, stride=1, padding=1),
            Conv2dIBNormRelu(int(hr_channels / 2), 1, 1, stride=1, padding=0,
                             with_ibn=False, with_relu=False),
        )

    def forward(self, img, lr8x, hr2x):
        lr4x = F.interpolate(lr8x, scale_factor=2, mode='bilinear', align_corners=False)
        lr4x = self.conv_lr4x(lr4x)
        f2x  = F.interpolate(lr4x, scale_factor=2, mode='bilinear', align_corners=False)
        f2x  = self.conv_f2x(torch.cat((f2x, hr2x), dim=1))
        f    = F.interpolate(f2x, scale_factor=2, mode='bilinear', align_corners=False)
        pred_matte = torch.sigmoid(self.conv_f(torch.cat((f, img), dim=1)))
        return pred_matte


# ── MODNet 主體 ───────────────────────────────────────────────────────────────

class MODNet(nn.Module):
    """
    MODNet: Real-Time Trimap-Free Portrait Matting via Objective Decomposition
    Paper: https://arxiv.org/abs/2011.11961
    """
    def __init__(self, in_channels=3, hr_channels=32, backbone_arch='mobilenetv2',
                 backbone_pretrained=True):
        super().__init__()
        self.in_channels       = in_channels
        self.hr_channels       = hr_channels
        self.backbone_arch     = backbone_arch
        self.backbone_pretrained = backbone_pretrained

        self.backbone = MobileNetV2()
        enc_channels  = [32, 16, 24, 32, 96, 1280]

        self.lr_branch = LRBranch(self.backbone)
        self.hr_branch = HRBranch(self.hr_channels, enc_channels)
        self.f_branch  = FusionBranch(self.hr_channels, enc_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:   # affine=False 時 weight/bias 是 None
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, img, inference=False):
        pred_semantic, lr8x, [enc2x, enc4x, *_] = self.lr_branch(img, inference)
        pred_detail,   hr2x                      = self.hr_branch(img, enc2x, enc4x, lr8x, inference)
        pred_matte                               = self.f_branch(img, lr8x, hr2x)
        return pred_semantic, pred_detail, pred_matte
