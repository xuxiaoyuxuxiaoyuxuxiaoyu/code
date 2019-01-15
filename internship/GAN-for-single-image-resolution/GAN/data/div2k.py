import os
from data import dataset
import numpy as np
import scipy.misc as misc
from tqdm import tqdm
from option import args

class DIV2K(dataset.dataset):
    def __init__(self, args, train=True):
        super(DIV2K, self).__init__(args, train)
        self.all_data = args.all_data

    def _scan(self):
        if not self.args.all_data:
            list_hr = []
            list_lr = [[] for _ in self.scale]
            if self.train:
                idx_begin = 0
                idx_end = self.args.n_train
            else:
                idx_begin = self.args.n_train
                idx_end = self.args.offset_val + self.args.n_val
            for i in range(idx_begin + 1, idx_end + 1):
                filename = '{:0>4}'.format(i)
                list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
                for si, s in enumerate(self.scale):
                    list_lr[si].append(os.path.join(
                        self.dir_lr,
                        'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                    ))

            return list_hr, list_lr
        else:
            if self.train:
                pbar = tqdm(total=self.args.n_train,desc='train_data')
            else:
                pbar = tqdm(total=self.args.n_val,desc='evaluate_data')
            list_hr = []
            list_lr = [[] for _ in self.scale]
            if self.train:
                idx_begin = 0
                idx_end = self.args.n_train
            else:
                idx_begin = 800
                idx_end = self.args.offset_val + self.args.n_val
            for i in range(idx_begin + 1, idx_end + 1):
                pbar.update()
                filename = '{:0>4}'.format(i)
                path_hr = os.path.join(self.dir_hr, filename + self.ext)
                hr = misc.imread(path_hr)
                list_hr.append(hr)
                for si, s in enumerate(self.scale):
                    path_lr = os.path.join(
                        self.dir_lr,
                        'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                    )
                    lr = misc.imread(path_lr)
                    list_lr[si].append(lr)
            pbar.close()

            return list_hr, list_lr

    def _set_filesystem(self, dir_data, train=True):
        self.apath = dir_data + '/DIV2K'
        if train:
            self.dir_hr = os.path.join(self.apath, 'DIV2K_train_HR')
            self.dir_lr = os.path.join(self.apath, 'DIV2K_train_LR_bicubic')
        else:
            self.dir_hr = os.path.join(self.apath, 'DIV2K_valid_HR')
            self.dir_lr = os.path.join(self.apath, 'DIV2K_valid_LR_bicubic')
        self.ext = '.png'

    def _pack_npz(self):
        def _pack_hr(idx_begin, idx_end):
            name = 'train_data_hr.npz' if self.train else 'valid_data_hr.npz'
            path = os.path.join(self.dir_hr,name)
            if not os.path.exists(path):
                data_hr = []
                for i in range(idx_begin + 1, idx_end + 1):
                    filename = '{:0>4}'.format(i)
                    hr = misc.imread(os.path.join(self.dir_hr, filename + self.ext))
                    data_hr.append(hr)
                data_hr = np.array(data_hr)
                np.savez(path, data_hr)
            else:
                print(path + ' exists')

        def _pack_lr(idx_begin, idx_end):
            for si, s in enumerate(self.scale):
                name = 'train_data_lrX{}.npz'.format(s) if self.train else 'valid_data_lrX{}.npz'.format(s)
                path = os.path.join(self.dir_lr, 'X{}'.format(s), name)
                if not os.path.exists(path):
                    data_lr = []
                    for i in range(idx_begin + 1, idx_end + 1):
                        filename = '{:0>4}'.format(i)
                        lr = misc.imread(os.path.join(
                            self.dir_lr,
                            'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                        ))
                        data_lr.append(lr)
                    data_lr = np.array(data_lr)
                    np.savez(path, data_lr)
                else:
                    print(path + ' exists')
        if self.train:
            idx_begin = 0
            idx_end = self.args.n_train//2

            _pack_hr(idx_begin, idx_end)
            _pack_lr(idx_begin, idx_end)
        else:
            idx_begin = self.args.n_train
            idx_end = self.args.offset_val + self.args.n_val

            _pack_hr(idx_begin, idx_end)
            _pack_lr(idx_begin, idx_end)


if __name__ == '__main__':
    dat = DIV2K(args)
    dat._pack_npz()
