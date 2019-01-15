from model import WGAN_GP

class model(WGAN_GP.WGAN_GP):
    def __init__(self,args):
        super(model,self).__init__(args)