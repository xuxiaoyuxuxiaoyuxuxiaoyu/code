import torch.optim as optim
import time
import torch.nn as nn
from torch.autograd import Variable
from loss import loss
import torch
import utility as util
import numpy as np
import torch.optim.lr_scheduler as lrs
class trainer():
    def __init__(self, args, model, data):
        print('creat a trainer for training')
        self.args = args
        self.scale = args.scale[0]
        if not self.args.not_load_model:
            self.model = util.load_mdoel(self.args.result_dir,'dsnet')
        else:
            self.model = model
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal(m.weight.data)
        '''attention, pretrained models have to be loaded here 
        or optimizer will not capture model parameters'''
        if not self.args.not_load_model:
            print('load model dsnet')
            self.model= util.load_mdoel(self.args.backup_dir,'dsnet')
        self.data = data
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.5, 0.9))
        self.loss = nn.L1Loss()
        self.save_loss = np.array([0.0])
        self.save_psnr = np.array([0.0])
        self.scheduler = lrs.StepLR(
            self.optimizer,
            step_size=args.lr_decay,
            gamma=args.gamma
        )

    def train(self):
        thresd = 1000
        sum_loss=0
        if self.args.load_loss:
            print('load loss')
            self.save_loss = util.load_loss(self.args.result_dir,'dsnet')
        self.model.train()
        for epoch in range(self.args.epochs):
            start_time = time.time()
            self.scheduler.step()
            for n_batch, (lr, hr, _) in enumerate(self.data.loader_train):
                if not self.args.cpu:
                    lr = lr.cuda()
                    hr = hr.cuda()
                lr = Variable(lr,requires_grad = True)
                hr = Variable(hr,requires_grad = True)
                out = self.model(hr)
                loss_ = self.loss(lr,out)
                sum_loss += loss_
                # tmp = util.calc_psnr(out, lr, self.scale, self.args.rgb_range, benchmark=False)
                # np.concatenate([self.save_loss,np.expand_dims(loss_.detach().cpu().numpy(),0)],0)
                loss_.backward()
                self.optimizer.step()
                print('epoch:{:0>5},batch:{},loss:{:.4f},lr = {:.4f}'.format(epoch,n_batch+1,loss_,self.scheduler.get_lr()[0]))

            util.save_mdoel(self.args.result_dir, self.model, 'dsnet')
            if self.args.save_loss:
                self.save_loss = util.add_loss(self.save_loss,np.expand_dims(sum_loss.detach().cpu().numpy()/800,0))
                util.save_loss(self.args.result_dir,self.save_loss,'dsnet')

            if sum_loss/800<thresd:
                if epoch>5:
                    thresd = sum_loss/800
                    util.save_mdoel(self.args.result_dir, self.model, 'best_dsnet')
                    print('best epoch {}'.format(epoch))
            print('epoch:{} takes {:.2f} seconds,average loss={:.4f}'.format(epoch,time.time()-start_time,sum_loss/800))
            sum_loss=0
            if(epoch%100==0):
                self.test()


    def test(self):
        epoch = self.scheduler.last_epoch+1
        self.model.eval()
        if not self.args.test_only:
            num_eval = 20
            psnr=0
            with torch.no_grad():
                for n_batch, (lr, hr, _) in enumerate(self.data.loader_eval):
                    if n_batch==20:
                        print('at epoch:{},{} images were evaluated. the average psnr is {:.4f}'.format(epoch,num_eval,psnr/num_eval))
                        if self.args.save_psnr:
                            self.save_psnr = util.add_psnr(self.save_psnr,
                                                           np.expand_dims(np.array(psnr/num_eval), 0))
                            util.save_psnr(self.args.result_dir, self.save_psnr, 'dsnet')
                        break
                    if not self.args.cpu:
                        lr = lr.cuda()
                        hr = hr.cuda()
                    lr = Variable(lr, requires_grad=True)
                    hr = Variable(hr, requires_grad=True)
                    out = self.model(hr)
                    tmp = util.calc_psnr(out, lr, self.scale, self.args.rgb_range, benchmark=False)
                    print('psnr of No.{:0>2} image in evaluation dataset is {:.4f}'.format(n_batch+1,tmp))
                    psnr+=tmp
