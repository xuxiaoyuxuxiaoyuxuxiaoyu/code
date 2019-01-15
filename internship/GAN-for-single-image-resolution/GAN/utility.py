'''
save model
plot loss

calculate psnr
inception score
'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import math


def save_mdoel(result_dir, model, name):
    path = os.path.join(result_dir, 'model_{}.pkl'.format(name))
    torch.save(model, path)


def load_mdoel(result_dir,name):
    path = os.path.join(result_dir, 'model_{}.pkl'.format(name))
    model = torch.load(path)
    return model


def add_loss(orig_loss, new_loss):
    orig_loss = np.concatenate([orig_loss, new_loss], 0)
    return orig_loss


def save_loss(result_dir, loss, name):
    path = os.path.join(result_dir, 'loss_{}.npy'.format(name))
    np.save(path,loss)


def load_loss(result_dir, name):
    path = os.path.join(result_dir, 'loss_{}.npy'.format(name))
    loss = np.load(path)
    return loss

def add_psnr(orig_psnr, new_psnr):
    orig_loss = np.concatenate([orig_psnr, new_psnr], 0)
    return orig_loss


def save_psnr(result_dir, psnr, name):
    path = os.path.join(result_dir, 'psnr_{}.npy'.format(name))
    np.save(path,psnr)


def load_psnr(result_dir, name):
    path = os.path.join(result_dir, 'psnr_{}.npy'.format(name))
    psnr = np.load(path)
    return psnr


def calc_psnr(sr, hr, scale, rgb_range, benchmark=False):
    sr = sr.detach().cpu()
    hr = hr.detach().cpu()
    diff = (sr - hr).data.div(rgb_range)
    if benchmark:
        shave = scale
        if diff.size(1) > 1:
            convert = diff.new(1, 3, 1, 1)
            convert[0, 0, 0, 0] = 65.738
            convert[0, 1, 0, 0] = 129.057
            convert[0, 2, 0, 0] = 25.064
            diff.mul_(convert).div_(256)
            diff = diff.sum(dim=1, keepdim=True)
    else:
        shave = scale + 6

    valid = diff[:, :, shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)


def save_d_loss(loss,result_dir):
    x = loss.shape[0]
    # loss = loss.cpu().detach().numpy()
    axis = np.linspace(1, x, x)
    fig = plt.figure()
    plt.title('d_loss')
    plt.plot(axis, loss[:, 0], color='red', label='d_r_loss',alpha=1)
    plt.plot(axis, loss[:, 1], color='green', label='d_f_loss',alpha=0.5)
    plt.plot(axis, loss[:, 2], color='blue', label='d_cost',alpha=0.5)
    plt.plot(axis, loss[:, 3], color='purple', label='wasserstein',alpha=0.5)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(os.path.join(result_dir,'d_loss.png'))
    plt.close(fig)

def save_g_loss(loss,result_dir):
    x = loss.shape[0]
    # loss = loss.cpu().detach().numpy()
    axis = np.linspace(1, x, x)
    fig = plt.figure()
    plt.title('g_loss')
    plt.plot(axis, loss, color='red', label='d_r_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.savefig(os.path.join(result_dir,'g_loss.png'))
    plt.close(fig)

def plot_d_loss(loss):
    x = loss.shape[0]
    # loss = loss.cpu().detach().numpy()
    axis = np.linspace(1, x, x)
    fig = plt.figure()
    plt.title('d_loss')
    plt.plot(axis, loss[:, 0], color='red', label='d_r_loss')
    plt.plot(axis, loss[:, 1], color='green', label='d_f_loss')
    plt.plot(axis, loss[:, 2], color='blue', label='d_cost')
    plt.plot(axis, loss[:, 3], color='purple', label='wasserstein')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.show()
    plt.close(fig)


def plot_g_loss(loss):
    x = loss.shape[0]
    # loss = loss.cpu().detach().numpy()
    axis = np.linspace(1, x, x)
    fig = plt.figure()
    plt.title('g_loss')
    plt.plot(axis, loss, color='red', label='g_loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.show()
    plt.close(fig)


class log_file():
    def __init__(self):
        self.dir = './result'
        self.path = os.path.join(self.dir, 'log.txt')
        self.open_type = 'a' if os.path.exists(self.path) else 'w'
        self.log_file = open(self.path, self.open_type)

    def write_log(self, log, refresh=False):
        self.log_file.writelines(log)
        if refresh:
            self.log_file.close()
            self.log_file = open(self.path, 'a')

def visualize_weiht(model,mode = 'all'):
    if mode=='single':
        for layer, data_ in model.state_dict().items():
            data_ = data_.cpu().detach().view(-1).numpy()
            # hist,bin_edges = np.histogram(weight)
            if layer.find('bias') != -1:
                n, bins, patches = plt.hist(x=data_, bins=10, color='#0504aa',
                                            alpha=0.7, rwidth=0.85)
            else:
                n, bins, patches = plt.hist(x=data_, bins=200, color='#0504aa',
                                            alpha=0.7, rwidth=0.85)
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('{} Histogram'.format(layer))
            plt.show()
            print('print enter to continue..')
            key = input()
            print(key)
    if mode=='all':
        w = np.ones([1], dtype=np.float32)
        b = np.ones([1], dtype=np.float32)
        for layer, data_ in model.state_dict().items():
            data_ = data_.cpu().detach().view(-1).numpy()
            # hist,bin_edges = np.histogram(weight)
            if layer.find('bias') != -1:
                b = np.concatenate([b,data_],0)

            else:
                w=np.concatenate([w,data_],0)
                plt.hist(x=data_, bins=200, color='#0504aa',
                                            alpha=0.7, rwidth=0.85)
        w=w[1:]
        b=b[1:]
        plt.hist(x=w, bins=200, color='#0504aa',
                 alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('weight Histogram')
        plt.show()

        plt.hist(x=b, bins=10, color='#0504aa',
                 alpha=0.7, rwidth=0.85)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('bias Histogram')
        plt.show()
def visualize_feature(model,layer,lr):
    a=1
if __name__ == '__main__':
    epoch = 1000
    loss = torch.Tensor(np.random.rand(1000, 1))
    # plot_loss(1000, loss, 'd_loss')
