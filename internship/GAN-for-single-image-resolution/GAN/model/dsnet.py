import torch.nn as nn
# import torch

def make_model(args):
    return DSNet(args)

class DSNet(nn.Module):
    def __init__(self, args):
        super(DSNet, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(args.n_colors, 16, 3, 1, 1, ), nn.ReLU(), nn.Conv2d(16, args.n_feats, 3, 1, 1)
        )
        self.res1 = ResBlock(n_feats=args.n_feats,kernel_size=3,num=4)
        self.res2 = ResBlock(n_feats=args.n_feats, kernel_size=3, num=2)
        # self.downsampler = Downsampler(args.n_feats)
        self.res3 = ResBlock(n_feats=args.n_feats, kernel_size=3, num=2)
        self.down_X2_1 = nn.Sequential(
            nn.Conv2d(args.n_feats, args.n_feats, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.down_X2_2 = nn.Sequential(
            nn.Conv2d(args.n_feats, args.n_feats, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.tail = nn.Sequential(
            nn.Conv2d(args.n_feats,args.n_feats,3,1,1),nn.ReLU(),nn.Conv2d(args.n_feats,args.n_colors,3,1,1)
        )

    def forward(self, x):
        out = self.head(x)
        # out = self.res1(out)
        out = self.down_X2_1(out)
        # out = self.res2(out)
        out = self.down_X2_2(out)
        # out = self.downsampler(out)
        out = self.res3(out)
        out = self.tail(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size,bias=True, bn=False, act=nn.LeakyReLU(True),num=8):
        super(ResBlock, self).__init__()
        m = []
        for i in range(num):
            # in_channels = (2**i)*n_feats
            # out_channels = 2*in_channels
            in_channels = n_feats
            out_channels = in_channels
            m.append(nn.Conv2d(in_channels, out_channels, kernel_size,1,kernel_size//2, bias=bias))
            if bn: m.append(nn.BatchNorm2d(out_channels))
            if i < num-1: m.append(act)
        self.body = nn.Sequential(*m)
        self.tail = nn.Sequential(
            nn.Conv2d(out_channels, n_feats, 1, 1, 0, bias=bias)
        )

    def forward(self, x):
        res = self.body(x)
        res = self.tail(res)
        # res += x

        return res

class Downsampler(nn.Sequential):
    def __init__(self, n_feats):
        m = []
        for _ in range(2):
            m.append(
                nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=2, padding=1),
                )
            m.append(nn.ReLU())
        super(Downsampler, self).__init__(*m)