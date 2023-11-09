import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (
    ConvRep5,
    ConvRep3,
    ConvRepPoint,
    QuadraticConnectionUnit,
    QuadraticConnectionUnitS,
)


class FEBlock(nn.Module):
    def __init__(self, num_feat, rep_scale=4):
        super(FEBlock, self).__init__()
        self.num_feat = num_feat
        self.conv_first = ConvRep3(3, num_feat, rep_scale=rep_scale)
        self.conv_up = ConvRep3(num_feat, num_feat*4, rep_scale=rep_scale)
        self.conv_last = ConvRep3(2*num_feat, 3, rep_scale=rep_scale)
        self.downsample = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv_first(x))
        base = x
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.lrelu(self.conv_up(x))
        x = self.downsample(x)
        x = torch.cat((x, base), 1)
        x = self.conv_last(x)
        return x


class FEBlockS(nn.Module):
    def __init__(self, num_feat):
        super(FEBlockS, self).__init__()
        self.num_feat = num_feat
        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.conv_up = nn.Conv2d(num_feat, num_feat*4, 3, 1, 1)
        self.conv_last = nn.Conv2d(2*num_feat, 3, 3, 1, 1)
        self.downsample = nn.PixelShuffle(2)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        x = self.lrelu(self.conv_first(x))
        base = x
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        x = self.lrelu(self.conv_up(x))
        x = self.downsample(x)
        x = torch.cat((x, base), 1)
        x = self.conv_last(x)
        return x


class SYESRX2Net(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(SYESRX2Net, self).__init__()
        img_range = 255.
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.channels = channels
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.feature_extraction = FEBlock(8, rep_scale=rep_scale)
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
        self.tail = nn.Sequential(nn.PixelShuffle(2), ConvRep3(3, 3, rep_scale=rep_scale))

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        inp = x
        x = self.feature_extraction(x)
        x = self.head(x)
        x = self.body(x)
        x = self.att(x) * x
        base = F.interpolate(inp, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.tail(x)+base
        return x / self.img_range + self.mean

    def slim(self):
        net_slim = SYESRX2NetS(self.channels)
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
            elif isinstance(mod, nn.PReLU):
                weight_slim['%s.weight' % name] = mod.weight
        net_slim.load_state_dict(weight_slim)
        return net_slim


class SYESRX2NetS(nn.Module):
    def __init__(self, channels=12):
        super(SYESRX2NetS, self).__init__()
        img_range = 255.
        rgb_mean = (0.4488, 0.4371, 0.4040)
        self.img_range = img_range
        self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        self.feature_extraction = FEBlockS(8)
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
            nn.Conv2d(channels, channels, 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.tail = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(3, 3, 3, 1, 1))

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        inp = x
        x = self.feature_extraction(x)
        x = self.head(x)
        x = self.body(x)
        x = self.att(x) * x
        base = F.interpolate(inp, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.tail(x) + base
        return x / self.img_range + self.mean
