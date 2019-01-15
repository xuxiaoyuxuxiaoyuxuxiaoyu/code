import torch
from model import model
from torch.autograd import Variable
from torch.autograd import grad

from option import args
import numpy as np
class loss():
    def __init__(self,args,g_input,real_data,model_):
        self.args = args
        self.model = model_
        self.real_data = real_data
        self.g_input = g_input
        self.pixel_loss = torch.nn.L1Loss()
        # print(self.args.pixel_loss)


    def get_d_loss(self):
        d_real_loss = self.model.d(self.real_data)
        d_real_loss = d_real_loss.mean()
        d_fake_loss = self.model.d(self.model.g(self.g_input)).mean()

        return d_real_loss,d_fake_loss,

    def get_g_loss(self):
        g_fake = self.model.g(self.g_input)
        g_loss = self.model.d(g_fake).mean()
        p_loss = self.pixel_loss(self.model.g(self.g_input), self.real_data)
        return g_loss,p_loss

    def get_gradients_penalty(self):
        dim = self.real_data.data.shape[0]
        alpha = torch.rand(dim,1)
        alpha = alpha.expand(
            dim, int(self.real_data.nelement() / dim))\
            .contiguous().view(dim, self.args.n_colors,\
                               self.args.patch_size, self.args.patch_size)
        alpha = alpha.cuda() if not self.args.cpu else alpha
        fake_data = self.model.g(self.g_input)
        intplts = alpha * self.real_data + (1 - alpha) * fake_data
        if not self.args.cpu:
            intplts = intplts.cuda()
        intplts = Variable(intplts,requires_grad=True)
        disc_intplt = self.model.d(intplts)
        gradients = grad(
            outputs = disc_intplt,inputs = intplts,
            grad_outputs = torch.ones(disc_intplt.size()).cuda()
            if not self.args.cpu else torch.ones(disc_intplt.size()),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        gradients = gradients.view(gradients.size(0), -1)
        grad_penalty = ((gradients.norm(2,dim=1)-1)**2).mean()*self.args.loss_lambda

        return grad_penalty


if __name__=='__main__':
    r_d = torch.rand(args.batch_size,3,64,64).cuda()
    g_i = torch.rand(args.batch_size,3,16,16).cuda()
    loss = loss(args,r_d,g_i)
    g_p = loss.get_gradients_penalty()
    d_r,d_f = loss.get_d_loss()
    g_loss = loss.get_g_loss()
    print(torch.cuda.is_available())
