import torch
from importlib import import_module

from .isp import SYEISPNet, SYEISPNetS
from .lle import SYELLENet, SYELLENetS
from .srx2 import SYESRX2Net, SYESRX2NetS
from .srx4 import SYESRX4Net, SYESRX4NetS


__all__ = {
    'SYEISPNet',
    'SYEISPNetS',

    'SYELLENet',
    'SYELLENetS',

    'SYESRX2Net',
    'SYESRX2NetS',
    'SYESRX4Net',
    'SYESRX4NetS',

    'import_model'
}


def import_model(opt):
    model_name = 'SYE'+opt.model_task.upper()
    if opt.model_task == 'sr':
        model_name += 'X{}'.format(opt.config['model']['scale'])
    kwargs = {'channels': opt.config['model']['channels']}

    if opt.config['model']['type'] == 're-parameterized':
        model_name += 'NetS'
    elif opt.config['model']['type'] == 'original':
        model_name += 'Net'
        kwargs['rep_scale'] = opt.config['model']['rep_scale']
    else:
        raise ValueError('unknown model type, please choose from [original, re-parameterized]')

    model = getattr(import_module('model'), model_name)(**kwargs)
    model = model.to(opt.device)

    if opt.config['model']['pretrained']:
        model.load_state_dict(torch.load(opt.config['model']['pretrained']))

    if opt.config['model']['type'] == 'original' and opt.config['model']['need_slim'] is True:
        model = model.slim().to(opt.device)
    return model
