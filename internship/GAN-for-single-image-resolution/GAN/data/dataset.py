
import torch.utils.data as data
import skimage.color as sc
import random
import scipy.misc as misc
import numpy as np
import torch
import torchvision

class dataset(data.Dataset):
    def __init__(self,args,train=True):
        self.args = args
        self.train = train
        self.scale = args.scale
        self.idx_scale = 0
        self._set_filesystem(args.dir_data,train)

        self.images_hr, self.images_lr = self._scan()


    def _set_filesystem(self,dir_data,train):
        raise NotImplementedError
    def _scan(self):
        raise NotImplementedError

    def __getitem__(self, item):
        if self.args.all_data:
            lr = self.images_lr[self.idx_scale][item]
            hr = self.images_hr[item]
            filename = str(item+1)
            lr, hr = self._get_patch(lr, hr)
            lr, hr = self.set_channel((lr, hr), self.args.n_colors)
            lr_tensor, hr_tensor = self.np2Tensor([lr, hr], self.args.rgb_range)
            return lr_tensor, hr_tensor, filename
        else:
            lr,hr,filename = self._load_file(item)
            lr,hr = self._get_patch(lr,hr)
            lr, hr = self.set_channel((lr, hr), self.args.n_colors)
            lr_tensor, hr_tensor = self.np2Tensor([lr, hr], self.args.rgb_range)
            return lr_tensor, hr_tensor, filename

    def __len__(self):
        return len(self.images_hr)

    def  _load_file(self,idx):
        lr = misc.imread(self.images_lr[self.idx_scale][idx])
        hr = misc.imread(self.images_hr[idx])
        filename = self.images_hr[idx].split('/')[-1]
        return lr,hr,filename
    def _get_patch(self,lr,hr):
        patch_size = self.args.patch_size
        scale = self.scale[self.idx_scale]
        multi_scale = len(self.scale) > 1
        if self.train:
            lr,hr = self._patch(
                lr, hr, patch_size, scale, multi_scale=multi_scale
            )
            lr, hr = self.augment([lr, hr])
            # lr = self.add_noise(lr, self.args.noise)
        else:
            ih, iw = lr.shape[0:2]
            hr = hr[0:ih * scale, 0:iw * scale]
        return lr, hr

    def set_channel(self,l,n_channel):
        def _set_channel(img):
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)

            c = img.shape[2]
            if n_channel == 1 and c == 3:
                img = np.expand_dims(sc.rgb2ycbcr(img)[:, :, 0], 2)#取Y通道并扩展一个维�?
            elif n_channel == 3 and c == 1:
                img = np.concatenate([img] * n_channel, 2)

            return img

        return [_set_channel(_l) for _l in l]
    def np2Tensor(self,l,rgb_range):
        def _np2Tensor(img):
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
            tensor = torch.from_numpy(np_transpose).float()
            tensor.mul_(rgb_range / 255.0)

            return tensor

        return [_np2Tensor(_l) for _l in l]
    def _patch(self,img_in, img_tar, patch_size, scale, multi_scale=False):
        ih, iw = img_in.shape[:2]

        p = scale if multi_scale else 1
        tp = p * patch_size
        ip = tp // scale

        ix = random.randrange(0, iw - ip + 1)
        iy = random.randrange(0, ih - ip + 1)
        tx, ty = scale * ix, scale * iy

        img_in = img_in[iy:iy + ip, ix:ix + ip, :]
        img_tar = img_tar[ty:ty + tp, tx:tx + tp, :]

        return img_in, img_tar

    def add_noise(self,x, noise='.'):
        if noise is not '.':
            noise_type = noise[0]
            noise_value = int(noise[1:])
            if noise_type == 'G':
                noises = np.random.normal(scale=noise_value, size=x.shape)
                noises = noises.round()
            elif noise_type == 'S':
                noises = np.random.poisson(x * noise_value) / noise_value
                noises = noises - noises.mean(axis=0).mean(axis=0)

            x_noise = x.astype(np.int16) + noises.astype(np.int16)
            x_noise = x_noise.clip(0, 255).astype(np.uint8)
            return x_noise
        else:
            return x

    def augment(self,l, hflip=True, rot=True):
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            if hflip: img = img[:, ::-1, :]
            if vflip: img = img[::-1, :, :]
            if rot90: img = img.transpose(1, 0, 2)

            return img

        return [_augment(_l) for _l in l]


