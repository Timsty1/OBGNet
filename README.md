# OBGNet
Code for ACM MM 2021 paper "Occlusion-aware Bi-directional Guided Network for Light Field Salient Object Detection"

## Overall Architecture

![Image text](https://github.com/Timsty1/OBGNet/raw/main/figure/network.png)

## Datasets

We train and evaluate our model on [DUTLF-v2] (https://github.com/OIPLab-DUT/DUTLF-V2) dataset. 

## Requirements

- Ubuntu 18.04
- torch 1.7.1
- python 3.8
- opencv-python 4.5.3.56
- imageio 2.4.1

## Train

We train our model on multiple GPUs.

If you want to retrain our model,  the process is as follows:

1. Please adjust hyperparameters in 'train_multiGPUs.py' , such as 'root', 'batch_size', and so on. 

2. Please set CUDA devices in train_start.sh.

3. cd to the code path, then start training.

   `nohup sh train_start.sh > log/xxx.txt 2>&1 &`

**Please note that the 'batch_size' should be greater than or equal to 16 for model generalization.**

## Test

You can download the checkpoint we provided ([Baidu Pan](https://pan.baidu.com/s/15bg13_g0XF0tOSdWjEjqAA), code:uk24).

Adjust paths in 'test.py' and run it to obtain predictions.

We also provide results of our model ([Baidu Pan](https://pan.baidu.com/s/1NhHQnLTFYDt3rSuSXP7gnQ), code:dy78).

## Contact us

If you have any questions, please contact us (20120370@bjtu.edu.cn).
