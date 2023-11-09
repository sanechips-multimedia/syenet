import torch.nn as nn
from .utils import (
    ConvRep5,
    ConvRep3,
    ConvRepPoint,
    DropBlock,
    QuadraticConnectionUnit,
    QuadraticConnectionUnitS,
)


class SYELLENet(nn.Module):
    def __init__(self, channels, rep_scale=4):
        super(SYELLENet, self).__init__()
        self.channels = channels
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
            12
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvRepPoint(channels, channels, rep_scale=rep_scale),
            nn.PReLU(channels),
            ConvRepPoint(channels, channels, rep_scale=rep_scale),
            nn.Sigmoid()
        )
        self.tail = ConvRep3(channels, 3, rep_scale=rep_scale)

        self.tail_warm = ConvRep3(channels, 3, rep_scale=rep_scale)
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
        net_slim = SYELLENetS(self.channels)
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


class SYELLENetS(nn.Module):
    def __init__(self, channels):
        super(SYELLENetS, self).__init__()
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
            12
        )
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.PReLU(channels),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid()
        )
        self.tail = nn.Conv2d(channels, 3, 3, 1, 1)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.att(x) * x
        return self.tail(x)
