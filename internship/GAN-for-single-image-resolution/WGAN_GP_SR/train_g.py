'''
trainer includes function
train(args,model,data,loss)
'''
import torch.optim as optim
import time
from torch.autograd import Variable
from loss import loss
import torch
import utility as util
import numpy as np
from option import args
import data
from model import model


class trainer():
    def __init__(self, args, model, data):
        print('creat a trainer for training')
        self.args = args
        self.model = model
        self.data = data
        self.optimizerD = optim.Adam(self.model.d.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.optimizerG = optim.Adam(self.model.g.parameters(), lr=1e-4, betas=(0.5, 0.9), weight_decay=args.weight_decay)
        self.one = torch.tensor(1.0).cuda() if not self.args.cpu else torch.FloatTensor([1.0])
        self.minus_one = self.one * -1
        # self.save_d_loss = torch.tensor([1.0,1.0,1.0,1.0]).cuda()
        self.save_d_loss = np.ones([1,4],dtype=np.float32)
        self.save_g_loss = np.ones([1],dtype=np.float32)

    def update_d(self, loss_):
        # updata D network
        #print(self.model.d.parameters())
        for p in self.model.d.parameters():
            p.requires_grad = True

        self.model.d.zero_grad()
        self.optimizerD.zero_grad()
        d_real_loss, d_fake_loss = loss_.get_d_loss()
        d_loss_penalty = loss_.get_gradients_penalty()
        d_real_loss.backward(self.minus_one)
        d_fake_loss.backward(self.one)
        d_loss_penalty.backward()
        # print(loss_.g_input.grad)

        self.d_cost = -d_real_loss + d_fake_loss + d_loss_penalty
        self.wasserstein = d_real_loss - d_fake_loss
        self.d_r_loss = d_real_loss
        self.d_f_loss = d_fake_loss
        self.optimizerD.step()

    def update_g(self, loss_):
        # for p in self.model.d.parameters():
        #     p.requires_grad = False

        self.model.g.zero_grad()
        g_loss,p_loss = loss_.get_g_loss()
        # g_loss.backward(self.minus_one)
        # if (self.args.pixel_loss):
        p_loss.backward(self.one)
        # p_loss = 0
        # self.g_cost = g_loss
        self.p_loss = p_loss
        self.optimizerG.step()

    def train(self):
        thresd = 100
        # if self.args.load_model:
        #     print('load model')
        #     self.model.d = util.load_mdoel(self.args.result_dir,'discriminator')
        #     self.model.g = util.load_mdoel(self.args.result_dir,'best_generator')
        #     self.save_d_loss = util.load_loss(self.args.result_dir,'discriminator')
        #     self.save_g_loss = util.load_loss(self.args.result_dir,'generator')
        for epoch in range(self.args.epochs):
            start_time = time.time()
            d_cnt = 0
            for n_batch, (lr, hr, _) in enumerate(self.data.loader_train):
                if not self.args.cpu:
                    lr = lr.cuda()
                    hr = hr.cuda()

                lr = Variable(lr,requires_grad = True)
                hr = Variable(hr,requires_grad = True)
                loss_ = loss(self.args, lr, hr,self.model)

                # if d_cnt < self.args.d_count:
                #     self.update_d(loss_)
                #     a = [self.d_r_loss.cpu().view(-1),self.d_f_loss.cpu().view(-1),self.d_cost.cpu().view(-1),self.wasserstein.cpu().view(-1)]
                #     a = np.array([[l.detach().numpy()[0] for l in a]])
                #     self.save_d_loss = util.add_loss(self.save_d_loss,a)
                #     #print(
                #     #    'batch:{}/{}--d_real_loss = {:0.6f}, d_fake_loss = {:0.6f},d_cost = {:0.6f}, wasserstein = {:0.6f}\n' \
                #     #        .format(n_batch, self.args.n_train // self.args.batch_size + 1, self.d_r_loss,
                #     #                self.d_f_loss, self.d_cost, self.wasserstein)
                #     #)
                # else:
                #     d_cnt = 0
                self.update_g(loss_)
                # a=self.g_cost.cpu().view(-1).detach().numpy()
                # self.save_g_loss = util.add_loss(self.save_g_loss,a)
                # self.save_g_loss = torch.cat([self.save_g_loss, a], 0)
                # del(a)
                print('p_loss={:0.6f}'.format(self.p_loss))
                # d_cnt += 1
            util.save_mdoel(self.args.result_dir, self.model.g, 'single_generator')

if __name__ =='__main__':
    data_ = data.Data(args)
    model_ = model(args)

    trainer_ = trainer(args,model_,data_)
    trainer_.train()
