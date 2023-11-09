import torch
import numpy as np
import cv2
from tqdm import tqdm

from logger import Logger
from option import get_option
from data import import_loader
from loss import import_loss
from model import import_model


def train(opt, logger):
    logger.info('task: {}, model task: {}'.format(opt.task, opt.model_task))

    train_loader, valid_loader = import_loader(opt)
    lr = float(opt.config['train']['lr'])
    lr_warmup = float(opt.config['train']['lr_warmup'])

    loss_warmup = import_loss('warmup')
    loss_training = import_loss(opt.model_task)
    net = import_model(opt)
    # logger.info(net)

    net.train()
    # Phase Warming-up
    if opt.config['train']['warmup']:
        logger.info('start warming-up')

        optim_warm = torch.optim.Adam(net.parameters(), lr_warmup, weight_decay=0)
        epochs = opt.config['train']['warmup_epoch']
        for epo in range(epochs):
            loss_li = []
            for img_inp, img_gt, _ in tqdm(train_loader, ncols=80):
                optim_warm.zero_grad()
                warmup_out1, warmup_out2 = net.forward_warm(img_inp)
                loss = loss_warmup(img_inp, img_gt, warmup_out1, warmup_out2)
                loss.backward()
                optim_warm.step()
                loss_li.append(loss.item())

            logger.info('epoch: {}, train_loss: {}'.format(epo+1, sum(loss_li)/len(loss_li)))
            torch.save(net.state_dict(), r'{}\model_pre.pkl'.format(opt.save_model_dir))
        logger.info('warming-up phase done')

    # Phase Training
    best_psnr = 0
    epochs = int(opt.config['train']['epoch'])
    optim = torch.optim.Adam(net.parameters(), lr, weight_decay=0)
    lr_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, 50, 2, 1e-7)

    logger.info('start training')
    for epo in range(epochs):
        loss_li = []
        test_psnr = []
        net.train()
        for img_inp, img_gt, _ in tqdm(train_loader, ncols=80):
            out = net(img_inp)
            loss = loss_training(out, img_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_li.append(loss.item())
        lr_sch.step()

        # Validation
        net.eval()
        for img_inp, img_gt, _ in tqdm(valid_loader, ncols=80):
            with torch.no_grad():
                out = net(img_inp)
                mse = ((out - img_gt)**2).mean((2, 3))
                psnr = (1 / mse).log10().mean() * 10
            test_psnr.append(psnr.item())
        mean_psnr = sum(test_psnr)/len(test_psnr)

        if (epo+1) % int(opt.config['train']['save_every']) == 0:
            torch.save(net.state_dict(), r'{}\model_{}.pkl'.format(opt.save_model_dir, epo+1))

        logger.info('epoch: {}, training loss: {}, validation psnr: {}'.format(
            epo+1, sum(loss_li) / len(loss_li), sum(test_psnr) / len(test_psnr)
        ))

        if mean_psnr > best_psnr:
            best_psnr = mean_psnr
            torch.save(net.state_dict(), r'{}\model_best.pkl'.format(opt.save_model_dir))
            if opt.config['train']['save_slim']:
                net_slim = net.slim().to(opt.device)
                torch.save(net_slim.state_dict(), r'{}\model_best_slim.pkl'.format(opt.save_model_dir))
                logger.info('best model saved and re-parameterized in epoch {}'.format(epo+1))
            else:
                logger.info('best model saved in epoch in epoch {}'.format(epo+1))

    logger.info('training done')


def test(opt, logger):
    test_loader = import_loader(opt)
    net = import_model(opt)
    net.eval()
    psnr_list = []
    logger.info('start testing')
    for (img_inp, img_gt, img_name) in test_loader:

        with torch.no_grad():
            out = net(img_inp)
            mse = ((out - img_gt)**2).mean((2, 3))
            psnr = (1 / mse).log10().mean() * 10

        if opt.config['test']['save']:
            out_img = (out.clip(0, 1)[0] * 255).permute([1, 2, 0]).cpu().numpy().astype(np.uint8)[..., ::-1]
            cv2.imwrite(r'{}\{}.png'.format(opt.save_image_dir, img_name[0]), out_img)

        psnr_list.append(psnr.item())
        logger.info('image name: {}, test psnr: {}'.format(img_name[0], psnr))

    logger.info('testing done, overall psnr: {}'.format(sum(psnr_list) / len(psnr_list)))


def demo(opt, logger):
    demo_loader = import_loader(opt)
    net = import_model(opt)
    net.eval()
    logger.info('start demonstration')
    for img_inp, img_name in demo_loader:

        with torch.no_grad():
            out = net(img_inp)
        out_img = (out.clip(0, 1)[0] * 255).permute([1, 2, 0]).cpu().numpy().astype(np.uint8)[..., ::-1]
        cv2.imwrite(r'{}\{}.png'.format(opt.save_image_dir, img_name[0]), out_img)
        logger.info('image name: {} output generated'.format(img_name[0]))
    logger.info('demonstration done')


if __name__ == "__main__":
    opt = get_option()
    logger = Logger(opt)

    if opt.task == 'train':
        train(opt, logger)
    elif opt.task == 'test':
        test(opt, logger)
    elif opt.task == 'demo':
        demo(opt, logger)
    else:
        raise ValueError('unknown task, please choose from [train, test, demo].')
