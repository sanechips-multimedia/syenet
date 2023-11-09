import os
import argparse
import yaml
from datetime import datetime


def get_option():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-task',
        default='train',
        type=str,
        choices=['train', 'test', 'demo'],
        help='choose the task for running the model'
    )
    parser.add_argument(
        '-model_task',
        default='isp',
        type=str,
        choices=['isp', 'lle', 'sr'],
        help='the model of the task'
    )
    parser.add_argument(
        '-device',
        default='cuda',
        type=str,
        help='choose the device to run the model'
    )
    opt = parser.parse_args()
    opt = opt_format(opt)
    return opt


def load_yaml(path):
    with open(path, 'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)
    return model_config


def save_yaml(path, file_dict):
    with open(path, 'w') as f:
        f.write(yaml.dump(file_dict, allow_unicode=True))


def opt_format(opt):
    opt.root = os.getcwd()
    opt.config = r'{}\config\{}.yaml'.format(opt.root, opt.model_task)
    opt.config = load_yaml(opt.config)

    proper_time = str(datetime.now()).split('.')[0].replace(':', '-')

    opt.config['exp_name'] = '{}_{}'.format(opt.task, opt.config['exp_name'])

    opt.experiments = r'{}\experiments\{}'.format(opt.root, '{} {}'.format(proper_time, opt.config['exp_name']))
    if not os.path.exists(opt.experiments):
        os.mkdir(opt.experiments)

    config_path = r'{}\config.yaml'.format(opt.experiments)
    save_yaml(config_path, opt.config)

    if opt.task == 'demo' or (opt.task == 'test' and opt.config['test']['save'] != False):
        opt.save_image = True
        opt.save_image_dir = r'{}\{}'.format(opt.experiments, 'images')
        if not os.path.exists(opt.save_image_dir):
            os.mkdir(opt.save_image_dir)

    opt.log_path = r'{}\logger.log'.format(opt.experiments)

    if opt.task == 'train':
        opt.save_model = True
        opt.save_model_dir = r'{}\{}'.format(opt.experiments, 'models')
        if not os.path.exists(opt.save_model_dir):
            os.mkdir(opt.save_model_dir)

    return opt
