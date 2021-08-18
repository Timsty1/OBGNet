import torch
import numpy as np
from data.dutlfv2 import DUTLF_V2
from net.OBGNet import OBGNet
import imageio
import os
import os.path as osp

os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # set GPU Number

root = '/data/Timsty/dataset/DUTLF-V2'  # dataset path
edge_outdir = ''  # set save path of salient edge predictions
sa1_outdir = ''  # set save path of initial salient predictions
sa2_outdir = ''  # set save path of final salient predictions
checkpoint_dir = ''  # set checkpoint path

if osp.exists(sa1_outdir) == False:
    os.makedirs(sa1_outdir)
if osp.exists(edge_outdir) == False:
    os.makedirs(edge_outdir)
if osp.exists(sa2_outdir) == False:
    os.makedirs(sa2_outdir)

dutlfv2_test = DUTLF_V2(root, type='test')
testloader = torch.utils.data.DataLoader(dutlfv2_test, batch_size=16)

model = OBGNet()
model.load_state_dict(torch.load(checkpoint_dir)['state_dict'])


def save_img(res, save_path):
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    res_copy = np.asarray(res)
    imageio.imwrite(save_path, res_copy)


if torch.cuda.is_available():
    model.cuda()
    model.eval()
    print(len(testloader))
    with torch.no_grad():
        for i, sample in enumerate(testloader):
            u, v, cvi = sample['u'].cuda(), sample['v'].cuda(), sample['cvi'].cuda()
            name = sample['name']
            outs = model(u, v, cvi)
            for j in range(outs[0].size(0)):
                save_img(outs[0][j, ...], osp.join(edge_outdir, name[j]))
                save_img(outs[1][j, ...], osp.join(sa1_outdir, name[j]))
                save_img(outs[2][j, ...], osp.join(sa2_outdir, name[j]))
