import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
    ConvRep5,
    ConvRep3,
    ConvRepPoint,
    QuadraticConnectionUnit,
    QuadraticConnectionUnitS,
    AdditionFusion,
    AdditionFusionS,
    ResBlock,
    ResBlockS
)


class PrePyramidL1(nn.Module):
    def __init__(self, num_feat, rep_scale=4):
        super(PrePyramidL1, self).__init__()
        self.conv_first = ConvRep3(num_feat, num_feat, rep_scale=rep_scale)
        self.resblock = ResBlock(num_feat=num_feat, rep_scale=rep_scale)

    def forward(self, x):
        feat_l1 = self.conv_first(x)
        feat_l1 = self.resblock(feat_l1)
        return feat_l1


class PrePyramidL1S(nn.Module):
    def __init__(self, num_feat):
        super(PrePyramidL1S, self).__init__()
        self.conv_first = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.resblock = ResBlockS(num_feat=num_feat)

    def forward(self, x):
        feat_l1 = self.conv_first(x)
        feat_l1 = self.resblock(feat_l1)
        return feat_l1


class PrePyramidL2(nn.Module):
    def __init__(self, num_feat, rep_scale=4):
        super(PrePyramidL2, self).__init__()
        self.conv_first = ConvRep3(num_feat, num_feat, rep_scale=rep_scale)
        self.resblock = ResBlock(num_feat=num_feat, rep_scale=rep_scale)

    def forward(self, x):
        feat_l2 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        feat_l2 = self.conv_first(feat_l2)
        feat_l2 = self.resblock(feat_l2)
        _, _, h, w = x.size()
        feat_l2 = nn.Upsample((h, w), mode='bilinear', align_corners=False)(feat_l2)
        feat_l2 = self.resblock(feat_l2)
        return feat_l2


class PrePyramidL2S(nn.Module):
    def __init__(self, num_feat):
        super(PrePyramidL2S, self).__init__()
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.resblock = ResBlockS(num_feat=num_feat)

    def forward(self, x):
        feat_l2 = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        feat_l2 = self.conv_first(feat_l2)
        feat_l2 = self.resblock(feat_l2)
        _, _, h, w = x.size()
        feat_l2 = nn.Upsample((h, w), mode='bilinear', align_corners=False)(feat_l2)
        feat_l2 = self.resblock(feat_l2)
        return feat_l2


class SYESRX4Net(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(SYESRX4Net, self).__init__()
        img_range = 255.
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.channels = channels
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.headpre = AdditionFusion(
            PrePyramidL1(3, rep_scale=rep_scale),
            PrePyramidL2(3, rep_scale=rep_scale),
            3
        )
        self.resblock = ResBlock(num_feat=3, rep_scale=rep_scale)
        self.head = QuadraticConnectionUnit(
            nn.Sequential(
                ConvRep5(3, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                ConvRep3(channels, channels, rep_scale=rep_scale)
            ),
            ConvRep5(3, channels, rep_scale=rep_scale),
            channels
        )
        self.body = QuadraticConnectionUnit(
            ConvRep3(channels, channels, rep_scale=rep_scale),
            ConvRepPoint(channels, channels, rep_scale=rep_scale),
            channels
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvRepPoint(channels, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
            ConvRepPoint(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.tail = nn.Sequential(
            ConvRep3(channels, 48, rep_scale=rep_scale),
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            ConvRep3(3, 3, rep_scale=rep_scale)
        )

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        inp = x
        x = self.headpre(x)
        x = self.resblock(x)
        x = self.head(x)
        x = self.body(x)
        x = self.att(x) * x
        base = F.interpolate(inp, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.tail(x)+base
        return x / self.img_range + self.mean

    def slim(self):
        net_slim = SYESRX4NetS(self.channels)
        weight_slim = net_slim.state_dict()
        for name, mod in self.named_modules():
            if isinstance(mod, ConvRep3) or isinstance(mod, ConvRep5) or isinstance(mod, ConvRepPoint):
                if '%s.weight' % name in weight_slim:
                    w, b = mod.slim()
                    weight_slim['%s.weight' % name] = w
                    weight_slim['%s.bias' % name] = b
                    if 'block2' in name:
                        weight_slim['%s.weight' % name] = weight_slim['%s.weight' % name] * 0.1
                        weight_slim['%s.bias' % name] = weight_slim['%s.bias' % name] * 0.1
            elif isinstance(mod, QuadraticConnectionUnit):
                weight_slim['%s.bias' % name] = mod.bias
            elif isinstance(mod, AdditionFusion):
                weight_slim['%s.bias' % name] = mod.bias
            elif isinstance(mod, nn.PReLU):
                weight_slim['%s.weight' % name] = mod.weight

        net_slim.load_state_dict(weight_slim)
        return net_slim


class SYESRX4NetS(nn.Module):
    def __init__(self, channels):
        super(SYESRX4NetS, self).__init__()
        img_range = 255.
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.headpre = AdditionFusionS(PrePyramidL1S(3), PrePyramidL2S(3), 3)
        self.resblock = ResBlockS(num_feat=3)
        self.head = QuadraticConnectionUnitS(
            nn.Sequential(
                nn.Conv2d(3, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            nn.Conv2d(3, channels, 5, 1, 2),
            channels
        )
        self.body = QuadraticConnectionUnitS(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 1, ),
            channels
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels,  channels, 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )

        self.tail = nn.Sequential(
            nn.Conv2d(channels, 48, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PixelShuffle(2),
            nn.Conv2d(3, 3, 3, 1, 1)
        )

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        inp = x
        x = self.headpre(x)
        x = self.resblock(x)
        x = self.head(x)
        x = self.body(x)
        x = self.att(x) * x
        base = F.interpolate(inp, scale_factor=4, mode='bilinear', align_corners=False)
        x = self.tail(x) + base
        return x / self.img_range + self.mean
