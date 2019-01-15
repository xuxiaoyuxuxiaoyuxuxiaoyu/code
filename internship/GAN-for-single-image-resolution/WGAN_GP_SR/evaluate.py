import torch
import data
from option import args
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import utility as util
import torch.nn as nn
import PIL.Image as Image


class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            print(module)
            x = module(x)
            print(name)
            img = x.cpu().detach().numpy()
            while(True):
                print('input number of {}, max num is {}'.format(name,x.shape[1]-1))
                n=input()
                if (n==''):break
                img1 = img[0, int(n), :, :]
                plt.figure(1)
                plt.imshow(img1)  #
                plt.axis('off')  #
                plt.show()
        return outputs


if __name__ == '__main__':
    print(args.all_data)
    print(args.test_only)
    switch = 'loss'
    md = 'all'
    print('please select mode:\n1->model\n2->weight\n3->loss\n4->psnr\n5->feature')
    a = input()
    if a == '1':
        switch = 'model'
    elif a == '2':
        switch = 'weight'  # loss model weight feature
        print('please select mode of weight\n1->single\n2->all')
        a_ = input()
        if a_ == '1':
            md = 'single'
        elif a_ == '2':
            md = 'all'
        else:
            print('Error loss mode')
            exit()
    elif a == '3':
        switch = 'loss'
    elif a=='4':
        switch = 'psnr'
    elif a == '5':
        switch = 'feature'
    else:
        print('Error mode')
        exit()

    if switch == 'model':
        # model = util.load_mdoel(args.result_dir, 'generator')
        model = util.load_mdoel(args.result_dir, 'single_generator')
        data_ = data.Data(args)
        with torch.no_grad():
            model.eval()
            for n_batch, (lr, hr, _) in enumerate(data_.loader_test if args.test_only else data_.loader_train2):
                lr = lr.cuda()
                hr = hr.cuda()
                out = model(lr)
                if True:
                    img = out.cpu().detach().numpy()
                    print('rock')
                    img = np.transpose(img[0, :, :, :], [1, 2, 0])
                    plt.figure(1)
                    plt.imshow(img)  #
                    plt.axis('off')  #
                    #plt.show()
                    # img = Image.fromarray((img*255).astype('uint8')).convert('RGB')
                    # img.save('baboon_sr.png')
                    img2 = hr.cpu().detach().numpy()
                    img2 = np.transpose(img2[0, :, :, :], [1, 2, 0])
                    plt.figure(2)
                    plt.imshow(img2)  #
                    plt.axis('off')  #
                    #plt.show()

                    img3 = lr.cpu().detach().numpy()
                    img3 = np.transpose(img3[0, :, :, :], [1, 2, 0])
                    plt.figure(3)
                    plt.imshow(img3)  #
                    plt.axis('off')  #
                    plt.show()

                    input()

    if switch == 'loss':
        loss_d = util.load_loss(args.result_dir, 'discriminator')
        loss_g = util.load_loss(args.result_dir, 'generator')
        util.plot_d_loss(loss_d)
        util.plot_g_loss(loss_g)
        # util.save_d_loss(loss_d, args.result_dir)
        # util.save_g_loss(loss_g, args.result_dir)

    if switch == 'weight':
        model = util.load_mdoel(args.result_dir, 'generator')
        # state = model.state_dict()
        util.visualize_weiht(model, md)

    if switch == 'psnr':
        model = util.load_mdoel(args.result_dir, 'single_generator')
        model.eval()
        # args.data_test=='Urban'
        args.test_only = True
        data_ = data.Data(args)
        with torch.no_grad():
            sum = 0
            for n_batch, (lr, hr, _) in enumerate(data_.loader_test if args.test_only else data_.loader_train2):
                lr = lr.cuda()
                hr = hr.cuda()
                out = model(lr)
                psnr = util.calc_psnr(out, hr, scale=4, rgb_range=1.0, benchmark=True)
                print('num:{},psnr:{:0.6f}'.format(n_batch+1,psnr))
                sum += psnr
            sum /= (n_batch+1)
            print('sum = {:0.8f}'.format(sum))

    if switch == 'feature':
        model = util.load_mdoel(args.result_dir, 'generator')
        layer_feature = ['head', 'body', 'upsampler', 'body2', 'tail']
        f_e = FeatureExtractor(model, layer_feature)
        data_ = data.Data(args)
        for n_batch, (lr, hr, _) in enumerate(data_.loader_test):
            lr = lr.cuda()
            # hr = hr.cuda()
            if n_batch == 0:
                out = f_e(lr)
        # print(model._modules.keys())
