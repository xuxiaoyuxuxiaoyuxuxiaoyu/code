import torch
import data
from option import args
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import utility as util
import torch.nn as nn
import PIL.Image as Image
from option import args
from torch.autograd import Variable




if __name__ == '__main__':
    model = util.load_mdoel(args.result_dir,'dsnet')
    data_ = data.Data(args)
    model.eval()
    x=0
    with torch.no_grad():
        for n_batch, (lr, hr, _) in enumerate(data_.loader_test):
            if not args.cpu:
                lr = lr.cuda()
                hr = hr.cuda()
            lr = Variable(lr, requires_grad=True)
            hr = Variable(hr, requires_grad=True)
            out = model(hr)

            out_ = out.cpu().detach().numpy()
            lr_ = lr.cpu().detach().numpy()
            psnr = util.calc_psnr(lr,out,scale=4, rgb_range=1)
            x+=psnr
            print('psnr{:0>2},{:.4f}'.format(n_batch+1,psnr))
            # img = np.transpose(out_[0, :, :, :], [1, 2, 0])
            # img_lr = np.transpose(lr_[0, :, :, :], [1, 2, 0])
            # plt.figure(1)
            # plt.imshow(img-img_lr)  #
            # plt.axis('off')  #
            # plt.show()
            # input()
        print('average psnr,{:.4f}'.format(x / 14))
