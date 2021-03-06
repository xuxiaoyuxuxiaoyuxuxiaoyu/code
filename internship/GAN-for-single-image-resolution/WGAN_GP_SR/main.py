import torch
from option import args
import data
import train
from model import model

def main():
    data_ = data.Data(args)
    model_ = model(args)

    trainer_ = train.trainer(args,model_,data_)
    trainer_.train()


if __name__ == '__main__':
    main()