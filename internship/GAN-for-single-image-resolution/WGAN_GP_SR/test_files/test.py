import torch
from option import args
import data
import train
from model import model
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import os

def main():
    path = os.path.join('/home/public/PycharmProjects/WGAN_GP_SR/result/','model_best_generator.pkl')
    print(path)
    model = torch.load(path)
    print(model)


if __name__ == '__main__':
    main()