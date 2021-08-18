import torch
from torch.utils.data import Dataset
import numpy as np
import os
import os.path as osp
import cv2


class DUTLF_V2(Dataset):
    def __init__(self, root, cropsize=(256, 256), tmpsize=(280, 280), type='train'):

        assert cropsize[0] <= tmpsize[0] and cropsize[1] <= tmpsize[1]

        self.cropsize = cropsize
        self.tmpsize = tmpsize
        self.type = type

        # print('Current root_dir: ' + root)

        # file types:
        # array; cvi: .jpg    depth; mask: .png    focal: .mat
        self.jpgnames = None
        self.pngnames = None
        if type == 'train' or type == 'val':
            self.lfdir = osp.join(root, 'Train/train_array')
            self.cvidir = osp.join(root, 'Train/train_images')
            self.maskdir = osp.join(root, 'Train/train_masks')
            self.depthdir = osp.join(root, 'Train/train_depth')
        elif type == 'test':
            self.lfdir = osp.join(root, 'Test/test_array')
            self.cvidir = osp.join(root, 'Test/test_images')
            self.maskdir = osp.join(root, 'Test/test_masks')
            self.depthdir = osp.join(root, 'Test/test_depth')

        else:
            print('Wrong Dataset Type! Please Check!!')

        self.jpgnames = sorted(os.listdir(self.cvidir))
        self.pngnames = sorted(os.listdir(self.maskdir))

        if type == 'val':
            self.jpgnames = self.jpgnames[len(self.jpgnames) * 9 // 10:]
            self.pngnames = self.pngnames[len(self.pngnames) * 9 // 10:]

    def __len__(self):
        return len(self.pngnames)

    def __getitem__(self, item):
        jpg_name = self.jpgnames[item]
        png_name = self.pngnames[item]

        lf = cv2.imread(osp.join(self.lfdir, jpg_name))  # BGR
        cvi = cv2.imread(osp.join(self.cvidir, jpg_name))
        mask = cv2.cvtColor(cv2.imread(osp.join(self.maskdir, png_name)), cv2.COLOR_BGR2GRAY).astype(np.float32)

        u, v = self.get_uv(lf, angular_size=9)
        del lf

        if self.type == 'train':
            # data augmentation:
            # 1. change contrast and brightness
            # 2. resize to a certain size larger than crop size
            # 3. crop
            rand = np.random.rand(2)
            if rand[0] >= 0.3:
                u, v, cvi, mask = self.check_size(u, v, cvi, mask, size=self.tmpsize, augment=True)
                u, v, cvi, mask = self.crop(self.cropsize, u, v, cvi, mask)
            else:
                if rand[1] >= 0.8:
                    u, v, cvi, mask = self.check_size(u, v, cvi, mask, size=self.cropsize, augment=True)
                else:
                    u, v, cvi, mask = self.check_size(u, v, cvi, mask, size=self.cropsize)
        elif self.type == 'val' or self.type == 'test':
            u, v, cvi, mask = self.check_size(u, v, cvi, mask, size=self.cropsize)
        edge = self.get_edge(mask, kernel_size=2)
        sample = self.transform(u, v, cvi, mask, edge, png_name)
        return sample

    def get_uv(self, lf, angular_size=9):
        lf_5d = np.stack(np.split(np.stack(np.split(lf, angular_size, axis=1), axis=0), angular_size, axis=1), axis=0)
        return lf_5d[angular_size // 2, ...], lf_5d[:, angular_size // 2, ...]

    def get_edge(self, img, kernel_size):
        [gy, gx] = np.gradient(img)
        edge = gy * gy + gx * gx
        edge[edge != 0.] = 1.
        kernel = np.ones((kernel_size, kernel_size), np.float32)
        edge = cv2.dilate(edge, kernel, iterations=1)
        return edge

    def check_size(self, u, v, cvi, mask, size=(280, 280), augment=False):
        if augment:
            contrast = np.random.rand(1)[0] * 0.6 + 0.7
            brightness = np.random.randint(-20, 20, dtype=int)
        # lf
        tmp = []
        for i in range(u.shape[0]):
            if augment:
                tmp.append(np.clip(
                    cv2.convertScaleAbs(cv2.resize(u[i], size, interpolation=cv2.INTER_LINEAR), alpha=contrast,
                                        beta=brightness), 0., 255.))
            else:
                tmp.append(cv2.resize(u[i], size, interpolation=cv2.INTER_LINEAR))
        u = np.stack(tmp, axis=0)
        tmp = []
        for i in range(v.shape[0]):
            if augment:
                tmp.append(np.clip(
                    cv2.convertScaleAbs(cv2.resize(v[i], size, interpolation=cv2.INTER_LINEAR), alpha=contrast,
                                        beta=brightness), 0., 255.))
            else:
                tmp.append(cv2.resize(v[i], size, interpolation=cv2.INTER_LINEAR))
        v = np.stack(tmp, axis=0)
        cvi = cv2.resize(cvi, size, interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        return u, v, cvi, mask

    def padding(self, lf_5d, cvi, mask):
        h, w = cvi.shape[0:2]
        dif_h, dif_w = ((h // 32) + 1) * 32 - h, ((w // 32) + 1) * 32 - w

        padded_lf_5d = np.zeros([*lf_5d.shape[0:2], h + dif_h, w + dif_w, 3], dtype=np.float32)
        padded_cvi = np.zeros([h + dif_h, w + dif_w, 3], dtype=np.float32)
        padded_mask = np.zeros([h + dif_h, w + dif_w], dtype=np.float32)

        padded_lf_5d[:, :, dif_h // 2:dif_h // 2 + h, dif_w // 2:dif_w // 2 + w, :] = lf_5d
        padded_cvi[dif_h // 2:dif_h // 2 + h, dif_w // 2:dif_w // 2 + w, :] = cvi
        padded_mask[dif_h // 2:dif_h // 2 + h, dif_w // 2:dif_w // 2 + w] = mask

        del lf_5d, cvi, mask
        return padded_lf_5d, padded_cvi, padded_mask

    def crop(self, t_size, u, v, cvi, mask):
        h, w = cvi.shape[0:2]
        if t_size == (h, w):
            return u, v, cvi, mask
        elif t_size[0] > h or t_size[1] > w:
            print('Crop size too large.')
        else:
            start_h, start_w = np.random.randint(0, h - t_size[0]), np.random.randint(0, w - t_size[1])
            return u[:, start_h:t_size[0] + start_h, start_w:t_size[1] + start_w, :], \
                   v[:, start_h:t_size[0] + start_h, start_w:t_size[1] + start_w, :], \
                   cvi[start_h:t_size[0] + start_h, start_w:t_size[1] + start_w, :], \
                   mask[start_h:t_size[0] + start_h, start_w:t_size[1] + start_w]

    def transform(self, u, v, cvi, mask, edge, name):
        u = u.transpose(0, 3, 1, 2) / 255.
        v = v.transpose(0, 3, 1, 2) / 255.
        cvi = cvi.transpose(2, 0, 1) / 255.
        mask = mask / 255.

        return {'u': torch.from_numpy(u[:, ::-1, ...].astype(np.float32)),
                'v': torch.from_numpy(v[:, ::-1, ...].astype(np.float32)),
                'cvi': torch.from_numpy(cvi.astype(np.float32)),
                'mask': torch.from_numpy(mask[np.newaxis, ...].astype(np.float32)),
                'edge': torch.from_numpy(edge[np.newaxis, ...].astype(np.float32)),
                'name': name
                }
