import torch
import numpy as np
import random
import os
from PIL import Image


class SRData(torch.utils.data.Dataset):
    def __init__(self, opt, lr_path, hr_path=None, training_task='train'):
        super(SRData, self).__init__()
        self.opt = opt
        self.img_li = [path for path in os.listdir(lr_path)]
        self.inp_path = lr_path
        self.gt_path = hr_path
        self.training_task = training_task

    def __getitem__(self, index):
        inp = Image.open(os.path.join(self.inp_path, self.img_li[index]))
        inp = np.array(inp).transpose([2, 0, 1])
        inp = inp.astype(np.float32) / 255

        inp = torch.Tensor(np.array(inp))
        inp = inp.to(self.opt.device)

        if self.gt_path: # gt_path -> train/test not demo
            gt = Image.open(os.path.join(self.gt_path, self.img_li[index]))
            gt = np.array(gt).transpose([2, 0, 1])
            gt = gt.astype(np.float32) / 255

            gt = torch.Tensor(np.array(gt))
            gt = gt.to(self.opt.device)

            if self.opt.task == 'train' and self.training_task != 'valid':
                inp, gt = self.get_patch_pair(inp, gt)
            return inp, gt, self.img_li[index].split('.')[0]

        return inp, self.img_li[index].split('.')[0]

    def __len__(self):
        return len(self.img_li)

    def get_patch_pair(self, inp, gt):
        scale = self.opt.config['model']['scale']
        lr_h, lr_w = inp.shape[1:] # c h w

        lr_patch_size = self.opt.config['train']['patch_size']
        lr_h_start = random.randint(0, lr_h - lr_patch_size)
        lr_w_start = random.randint(0, lr_w - lr_patch_size)
        lr_patch = inp[:, lr_h_start: lr_h_start + lr_patch_size, lr_w_start: lr_w_start + lr_patch_size]

        hr_patch_size = lr_patch_size * scale
        hr_h_start = lr_h_start * scale
        hr_w_start = lr_w_start * scale
        hr_patch = gt[:, hr_h_start: hr_h_start + hr_patch_size, hr_w_start: hr_w_start + hr_patch_size]

        return lr_patch, hr_patch
