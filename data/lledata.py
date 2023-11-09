import torch
import numpy as np
import os
from PIL import Image


class LLEData(torch.utils.data.Dataset):
    def __init__(self, opt, inp_path, gt_path=None):
        super(LLEData, self).__init__()
        self.img_li = [path for path in os.listdir(inp_path)]
        self.inp_path = inp_path
        self.gt_path = gt_path
        self.opt = opt

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

            return inp, gt, self.img_li[index].split('.')[0]
        return inp, self.img_li[index].split('.')[0]

    def __len__(self):
        return len(self.img_li)
