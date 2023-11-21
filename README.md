# SYENet

This repository contains the official implementation for ICCV 2023 paper:

[SYENet: A Simple Yet Effective Network for Multiple Low-Level Vision Tasks with Real-Time Performance on Mobile Device](https://openaccess.thecvf.com/content/ICCV2023/papers/Gou_SYENet_A_Simple_Yet_Effective_Network_for_Multiple_Low-Level_Vision_ICCV_2023_paper.pdf)

Weiran Gou, Ziyao Yi, Yan Xiang, Shaoqing Li, Zibin Liu, Dehui Kong, Ke Xu. [[arxiv]](https://arxiv.org/abs/2308.08137)

SYENet is an efficient network that could handle multiple low-level vision (isp, lle and sr) tasks. SYENet utilises re-parameterization for fast inference and got the highest score in [MAI 2022 Learned Smartphone Challenge](https://arxiv.org/abs/2211.03885).

### Citation

If you find our work useful in your research, please cite:

```
@InProceedings{Gou_2023_ICCV,
    author    = {Gou, Weiran and Yi, Ziyao and Xiang, Yan and Li, Shaoqing and Liu, Zibin and Kong, Dehui and Xu, Ke},
    title     = {SYENet: A Simple Yet Effective Network for Multiple Low-Level Vision Tasks with Real-Time Performance on Mobile Device},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {12182-12195}
}
```

### Environment

- python 3.8
- pytorch == 1.12.1
- numpy == 1.23.3
- cv2 == 4.7.0
- PIL == 9.2.0
- tqdm == 4.64.1
- yaml == 6.0

### Configuration

Edit the yaml files (isp.yaml, lle.yaml, sr.yaml) in ./config.

You are recommended to use [basicsr](https://github.com/XPixelGroup/BasicSR) to train our sr models. We put the train/test configuration files for training/testing our sr models using basicsr in ./config, which are sr_basicsr_train.yaml and sr_basicsr_test.yaml.

### Train

If you want to re-parameterize the model and save it, please set 'train $\rightarrow$ save_slim' parameter in the configuration yaml file to be 'true'. And hence, the re-parameterized small model for fast inference will be saved.

For isp and lle tasks, we utilise a warmup phase which is a self-supervised training stage. This phase could be cancalled by setting 'train $\rightarrow$ warmup' parameter in the configuration yaml file to be false.

```bash
python main.py -task train -model_type original -model_task isp/lle/sr -device cuda
```

### Test

Set the 'model $\rightarrow$ type' parameter in the configuration yaml file to be original if you are loading an original pretrained model, otherwise, set it to be re-parameterized for loading a re-parameterized model.

If you are loading an original pretrained model, but you want to re-parameterize it before inference, you could set 'model $rightarrow$ need_slim' parameter in the configuration yaml file to be true. Notice that you cannot re-parameterize a re-parameterized model.

You could save the images generated in the test by setting 'test $\rightarrow$ save' parameter in the configuration yaml file to be true.

```bash
python main.py -task test -model_type original -model_task isp/lle/sr -device cuda
```

### Demo

You could save the images generated in the demonstration by setting 'demo $\rightarrow$ save' parameter in the configuration yaml file to be true.

```bash
python main.py -task demo -model_type original -model_task isp/lle/sr -device cuda
```
