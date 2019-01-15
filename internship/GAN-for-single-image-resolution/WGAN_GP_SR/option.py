import argparse

parser = argparse.ArgumentParser(description='WGAN_GP for SuperResolution')


parser.add_argument('--cpu', action='store_true',default=False,
                    help='use cpu only')
#model specifications
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--n_dims', type=int, default=16,
                    help='dimensions in intermidiate layer of discriminator')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--result_dir', type=str, default='./result',
                    help='directory for saving model')
parser.add_argument('--backup_dir', type=str, default='./backup',
                    help='directory for backup model')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')


#data specifications
parser.add_argument('--dir_data', type=str, default='/home/public/xuxiaoyu/code/dataset/',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='/home/public/xuxiaoyu/code/dataset/',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='Set14',
                    help='test dataset name')
parser.add_argument('--patch_size', type=int, default=64,
                    help='output patch size')
parser.add_argument('--scale', default='4',
                    help='super resolution scale')

parser.add_argument('--n_train', type=int, default=800,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=100,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--rgb_range', type=int, default=1,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')

parser.add_argument('--not_load_model', action='store_true',default=False,
                    help='load pretrained model ')

parser.add_argument('--all_data', action='store_true', default=False,
                    help='load dataset once ')
#train specification

parser.add_argument('--test_only', action='store_true',default=False,
                    help='set this option to test the model')
parser.add_argument('--batch_size', type=int, default=60,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=500000,
                    help='epochs for training')
parser.add_argument('--d_count', type=int, default=5,
                    help='update steps of discriminator per generator step ')


#loss specification
parser.add_argument('--loss_lambda', type=int, default=10,
                    help='lambda for loss penalty')
parser.add_argument('--pixel_loss', action='store_true',default=False,
                    help='lambda for loss penalty')
parser.add_argument('--load_loss', action='store_true',default=False,
                    help='load loss of npz format ')
parser.add_argument('--save_loss', action='store_true',default=False,
                    help='load loss of npz format ')

#edsr specification
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')

args = parser.parse_args()


args.scale = list(map(lambda x: int(x), args.scale.split('+')))