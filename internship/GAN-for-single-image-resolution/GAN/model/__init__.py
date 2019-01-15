from model import wgan_gp
import torch.nn as nn
from importlib import import_module
class model():
    def __init__(self,args):
        print('Making model')
        module = import_module('model.' + args.model.lower())
        self.model = module.make_model(args)
        if not args.cpu:
            self.model.cuda()
        print(self.model)