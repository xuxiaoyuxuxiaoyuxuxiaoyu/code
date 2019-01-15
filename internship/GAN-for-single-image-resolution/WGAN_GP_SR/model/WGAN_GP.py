import torch.nn as nn
import math

class WGAN_GP(nn.Module):
    def __init__(self,args):
        super(WGAN_GP,self).__init__()
        if args.cpu:
            self.g = Generator(args)
            self.d = Discriminator(args)
        else:
            self.g = Generator(args).cuda()
            self.d = Discriminator(args).cuda()

class Generator(nn.Module):
    def __init__(self,args):
        super(Generator,self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors,16,3,1,1),nn.ReLU(),nn.Conv2d(16,args.n_feats,3,1,1)
        )
        self.body = ResBlock(depth=8,n_feats=args.n_feats,kernel_size=3)
        self.upsampler = Upsampler(scale=args.scale[0],n_feats=args.n_feats)
        self.body2 = ResBlock(depth=4,n_feats=args.n_feats,kernel_size=3,act=nn.Sigmoid())
        self.tail = nn.Sequential(
            nn.Conv2d(args.n_feats,16,3,1,1),nn.ReLU(),nn.Conv2d(16,args.n_colors,3,1,1)
        )

    def forward(self, x):
        buf = self.head(x)
        lr_ = self.body(buf)
        hr_raw = self.upsampler(lr_)
        hr_raw = self.body2(hr_raw)
        out = self.tail(hr_raw)
        return out



class ResBlock(nn.Module):
    def __init__(
        self, depth,n_feats, kernel_size,
        bias=True, bn=False, act=nn.LeakyReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        num=depth
        m = []
        for i in range(num):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size,1,kernel_size//2, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feats))
            if i < num-1: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, scale, n_feats, bn=False, act='relu', bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(
                    nn.ConvTranspose2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1, output_padding=1),
                )
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(nn.Conv2d(n_feats, 9 * n_feats, kernel_size=3, stride=1, padding=1, bias=bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class Discriminator(nn.Module):
    def __init__(self,args):
        super(Discriminator, self).__init__()
        self.DIM = args.n_dims
        self.patch_size = args.patch_size
        body = nn.Sequential(
            nn.Conv2d(3, self.DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(self.DIM, 2 * self.DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.DIM, 4 * self.DIM, 3, 2, padding=1),
            nn.LeakyReLU(),
        )

        self.body = body
        self.linear = nn.Linear((self.patch_size//8)*(self.patch_size//8)*4*self.DIM, 1)

    def forward(self, input):
        output = self.body(input)
        # the parameter '-1' for view means the first dimensions depends on the other dims
        #  in the condition that thier multipications equals all elements
        output = output.view(-1, (self.patch_size//8)*(self.patch_size//8)*4*self.DIM)

        output = self.linear(output)
        return output

