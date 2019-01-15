import torch
import data
from option import args
import numpy as np
import scipy.misc as misc
import matplotlib.pyplot as plt
import utility as util
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import grad
import struct

class model(nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.body = nn.Conv2d(3,1,kernel_size=3,stride=1,padding=1)
        self.act = nn.Sigmoid()


    def forward(self, x):
        out = self.body(x)
        out = self.act(out)
        return out

if __name__ == '__main__':
    a=b'\x00\x00\x00\x01'
    b=bytes(a)
    c=struct.unpack('!f',b)[0]
    d=np.ones((1,3,10,10),dtype=np.float32)
    e=torch.Tensor(d)
    x = Variable(e).cuda()
    x.requires_grad = True
    model_ = model().cuda()
    for m in model_.modules():
        if isinstance(m,nn.Conv2d):
            nn.init.xavier_normal(m.weight)
            # nn.init.xavier_normal(m.bias)
    # data_ = data.Data(args)
    # for n_batch, (lr, hr, _) in enumerate(data_.loader_test if args.test_only else data_.loader_train2):
    #     lr = lr.cuda()
    #     hr = hr.cuda()
    out = model_(x)
    gradients = grad(
        outputs=out, inputs=x,
        grad_outputs=torch.ones(out.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    d_g = grad(
        outputs=gradients, inputs=x,
        grad_outputs=torch.ones(gradients.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    print(out)
    print(gradients)
    print(d_g)



