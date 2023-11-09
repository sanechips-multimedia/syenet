from torch.utils import data
from importlib import import_module

from .ispdata import ISPData
from .lledata import LLEData
from .srdata import SRData

__all__ = {
    'ISPData',
    'LLEData',
    'SRData',
    'import_loader'
}


def import_loader(opt):
    dataset_name = opt.model_task.upper()+'Data'
    dataset = getattr(import_module('data'), dataset_name)

    if opt.task == 'train':
        train_inp_path = opt.config['train']['train_inp']
        train_gt_path = opt.config['train']['train_gt']
        valid_inp_path = opt.config['train']['valid_inp']
        valid_gt_path = opt.config['train']['valid_gt']

        train_data = dataset(opt, train_inp_path, train_gt_path)
        if opt.model_task == 'sr':
            valid_data = dataset(opt, valid_inp_path, valid_gt_path, 'valid')
        else:
            valid_data = dataset(opt, valid_inp_path, valid_gt_path)
        train_loader = data.DataLoader(
            train_data,
            batch_size=opt.config['train']['batch_size'],
            shuffle=True,
            num_workers=opt.config['train']['num_workers'],
            drop_last=True,
        )
        valid_loader = data.DataLoader(
            valid_data,
            batch_size=1,
            shuffle=False,
            num_workers=opt.config['train']['num_workers'],
            drop_last=False,
        )
        return train_loader, valid_loader

    elif opt.task == 'test':
        inp_test_path = opt.config['test']['test_inp']
        gt_test_path = opt.config['test']['test_gt']

        test_data = dataset(opt, inp_test_path, gt_test_path)
        test_loader = data.DataLoader(
            test_data,
            batch_size=1,
            shuffle=False,
            num_workers=opt.config['test']['num_workers'],
            drop_last=False,
        )
        return test_loader

    elif opt.task == 'demo':
        inp_demo_path = opt.config['demo']['demo_inp']
        demo_data = dataset(opt, inp_demo_path)
        demo_loader = data.DataLoader(
            demo_data,
            batch_size=1,
            shuffle=False,
            num_workers=opt.config['demo']['num_workers'],
            drop_last=False,
        )
        return demo_loader

    else:
        raise ValueError('unknown task, please choose from [train, test, demo]')
