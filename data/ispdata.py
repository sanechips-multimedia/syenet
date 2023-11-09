import torch
import numpy as np
import os
from PIL import Image


class ISPData(torch.utils.data.Dataset):
    def __init__(self, opt, raw_path, rgb_path=None):
        super(ISPData, self).__init__()
        self.img_li = [path for path in os.listdir(raw_path)]
        self.raw_path = raw_path
        self.rgb_path = rgb_path
        self.opt = opt

    def __getitem__(self, index):
        raw = Image.open(os.path.join(self.raw_path, self.img_li[index]))
        raw = np.array(raw)
        raw = self.bayer2rggb(raw)
        raw = raw.astype(np.float32) / 4095

        raw = torch.Tensor(np.array(raw))
        raw = raw.to(self.opt.device)

        if self.rgb_path: # gt_path -> train/test not demo
            rgb = Image.open(os.path.join(self.rgb_path, self.img_li[index]))
            rgb = np.array(rgb).transpose([2, 0, 1])
            rgb = rgb.astype(np.float32) / 255

            rgb = torch.Tensor(np.array(rgb))
            rgb = rgb.to(self.opt.device)

            return raw, rgb, self.img_li[index].split('.')[0]

        return raw, self.img_li[index].split('.')[0]

    def __len__(self):
        return len(self.img_li)

    def bayer2rggb(self, img_bayer):
        h, w = img_bayer.shape
        img_bayer = img_bayer.reshape(h // 2, 2, w // 2, 2)
        img_bayer = img_bayer.transpose([1, 3, 0, 2]).reshape([-1, h // 2, w // 2])
        return img_bayer
