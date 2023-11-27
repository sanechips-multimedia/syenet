import torch
from torch import nn
from model import SYEISPNetS


def export_onnx(pretrained):
    net = SYEISPNetS(channels=12)
    checkpoint = torch.load(pretrained)
    net.load_state_dict(checkpoint)
    net.eval()
    net = net.slim().eval()
    net.body.block1.weight = nn.Parameter(
        net.body.block1.weight.reshape((3, 4, 12, 3, 3)).permute([1, 0, 2, 3, 4]).reshape((12, 12, 3, 3))
    )
    net.body.block2.weight = nn.Parameter(
        net.body.block2.weight.reshape((3, 4, 12, 1, 1)).permute([1, 0, 2, 3, 4]).reshape((12, 12, 1, 1))
    )
    net.body.block1.bias = nn.Parameter(net.body.block1.bias.reshape((3, 4)).permute([1, 0]).reshape(12))
    net.body.block2.bias = nn.Parameter(net.body.block2.bias.reshape((3, 4)).permute([1, 0]).reshape(12))
    net.body.bias = nn.Parameter(net.body.bias.reshape((3, 4)).permute([1, 0]).reshape(1, 12, 1, 1))
    net.att[1].weight = nn.Parameter(
        net.att[1].weight.reshape((12, 3, 4, 1, 1)).permute([0, 2, 1, 3, 4]).reshape((12, 12, 1, 1))
    )
    net.att[3].weight = nn.Parameter(
        net.att[3].weight.reshape((3, 4, 12, 1, 1)).permute([1, 0, 2, 3, 4]).reshape((12, 12, 1, 1))
    )
    net.att[3].bias = nn.Parameter(net.att[3].bias.reshape((3, 4)).permute([1, 0]).reshape(12))
    x = torch.rand(1, 4, 544, 960)
    torch.onnx.export(net, x, 'model.onnx', opset_version=11)
    torch.onnx.export(net, x, 'model_none.onnx', opset_version=11, dynamic_axes={'input': [2, 3], 'output': [2, 3]})


if __name__ == "__main__":
    export_onnx(r'D:\Data\AIModel\LowLevel\SYENet\experiments\train_isp\models\model_best_slim.pkl')
