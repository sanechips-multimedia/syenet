import torch.nn as nn
from .utils import (
    ConvRep5,
    ConvRep3,
    ConvRepPoint,
    DropBlock,
    QuadraticConnectionUnit,
    QuadraticConnectionUnitS,
)


class SYEISPNet(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(SYEISPNet, self).__init__()
        self.channels = channels
        self.head = QuadraticConnectionUnit(
            nn.Sequential(
                ConvRep5(4, channels, rep_scale=rep_scale),
                nn.PReLU(channels),
                ConvRep3(channels, channels, rep_scale=rep_scale)
            ),
            ConvRep5(4, channels, rep_scale=rep_scale),
            channels
        )
        self.body = QuadraticConnectionUnit(
            ConvRep3(channels, 12, rep_scale=rep_scale),
            ConvRepPoint(channels, 12, rep_scale=rep_scale),
            12
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvRepPoint(12, 12, rep_scale=rep_scale),
            nn.PReLU(12),
            ConvRepPoint(12, 12, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.tail = nn.Sequential(nn.PixelShuffle(2), ConvRep3(3, 3, rep_scale=rep_scale))
        self.tail_warm = ConvRep3(12, 4, rep_scale=rep_scale)
        self.drop = DropBlock(3)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.att(x) * x
        return self.tail(x)

    def forward_warm(self, x):
        x = self.drop(x)
        x = self.head(x)
        x = self.body(x)
        return self.tail(x), self.tail_warm(x)

    def slim(self):
        net_slim = SYEISPNetS(self.channels)
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


class SYEISPNetS(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(SYEISPNetS, self).__init__()
        self.head = QuadraticConnectionUnitS(
            nn.Sequential(
                nn.Conv2d(4, channels, 5, 1, 2),
                nn.PReLU(channels),
                nn.Conv2d(channels, channels, 3, 1, 1)
            ),
            nn.Conv2d(4, channels, 5, 1, 2),
            channels
        )
        self.body = QuadraticConnectionUnitS(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.Conv2d(channels, channels, 1, ),
            12
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(12, 12, 1),
            nn.PReLU(12),
            nn.Conv2d(12, 12, 1),
            nn.Sigmoid()
        )
        self.tail = nn.Sequential(nn.PixelShuffle(2), nn.Conv2d(3, 3, 3, 1, 1))

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.att(x) * x
        return self.tail(x)
